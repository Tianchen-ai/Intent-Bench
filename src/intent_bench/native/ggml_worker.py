import base64
import ctypes as C
from pathlib import Path
import subprocess
import time
import numpy as np
from cnnl_worker import serve


class Worker:
    def __init__(self,request):
        self._repeat=None
        root=Path(__file__).parent;source=Path(request["source"]);libraries=Path(request["libraries"])
        start=time.perf_counter()
        build=subprocess.run([request.get("cxx","c++"),"-O3","-std=c++17","-shared","-fPIC",str(root/"ggml_ops.cpp"),
            "-I"+str(source/"ggml/include"),"-L"+str(libraries),"-Wl,-rpath,"+str(libraries),
            "-lggml","-lggml-cpu","-lggml-base","-o",str(root/"ops.so")],text=True,capture_output=True)
        if build.returncode:raise RuntimeError(build.stdout+build.stderr)
        self.compile_seconds=time.perf_counter()-start
        self.lib=C.CDLL(str(root/"ops.so"))
        self.lib.bench_create.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int]
        self.lib.bench_create.restype=C.c_void_p
        self.lib.bench_run.argtypes=[C.c_void_p];self.lib.bench_run.restype=C.c_int
        self.lib.bench_output.argtypes=[C.c_void_p,C.c_void_p,C.c_int];self.lib.bench_output.restype=None
        self.lib.bench_measure.argtypes=[C.c_void_p,C.c_int];self.lib.bench_measure.restype=C.c_double
        self.lib.bench_destroy.argtypes=[C.c_void_p];self.lib.bench_destroy.restype=None
        self.lib.bench_error.argtypes=[];self.lib.bench_error.restype=C.c_char_p
        self.lib.bench_packed_weight.argtypes=[C.c_void_p,C.c_void_p];self.lib.bench_packed_weight.restype=None
        self.lib.bench_packed_activation.argtypes=[C.c_void_p,C.c_void_p];self.lib.bench_packed_activation.restype=None
        self.lib.bench_gated_delta_create.argtypes=[C.POINTER(C.c_void_p)]+[C.c_int]*6
        self.lib.bench_gated_delta_create.restype=C.c_void_p
        self.lib.bench_state_output.argtypes=[C.c_void_p,C.c_void_p];self.lib.bench_state_output.restype=None
        if request["operation"] in ("softmax", "weighted_rms_norm", "relu", "transpose", "layer_norm"):
            x=np.load(root/"input0.npy")
            if x.dtype not in (np.float32,np.float16) or x.ndim!=2:raise ValueError("expected FP16 or FP32 row input")
            affine=request["operation"]=="layer_norm"
            weighted=request["operation"]=="weighted_rms_norm" or affine
            w=np.load(root/"input1.npy") if weighted else None
            bias=np.load(root/"input2.npy") if affine else None
            if weighted and (x.dtype!=np.float32 or w.dtype!=x.dtype or w.shape!=(x.shape[1],)):raise ValueError("gain signature mismatch")
            if affine and (bias.shape!=w.shape or bias.dtype!=w.dtype):raise ValueError("bias signature mismatch")
            entry=self.lib.bench_row_create
            entry.argtypes=[C.c_int,C.c_bool,C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_float,C.c_int]
            entry.restype=C.c_void_p
            operation=("softmax","weighted_rms_norm","relu","transpose","layer_norm").index(request["operation"])
            self.context=entry(operation,x.dtype==np.float16,x.ctypes.data,w.ctypes.data if weighted else None,
                               bias.ctypes.data if affine else None,*x.shape,request.get("epsilon",0),request["threads"])
            if not self.context:raise RuntimeError(self.lib.bench_error().decode())
            self.kind=9 if weighted else 8
            self.output=np.empty(x.T.shape if request["operation"]=="transpose" else x.shape,dtype=x.dtype);self.reference=False
            return
        if request["operation"] == "gated_delta":
            self.kind=6
            if request.get("reference",False):
                raise ValueError("GGML gated delta is a library baseline, not a numerical oracle")
            self.inputs=[np.load(root/f"input{i}.npy") for i in range(5)]
            q,k,v,gate,beta=self.inputs
            if any(value.dtype!=np.uint16 or not value.flags.c_contiguous for value in self.inputs):
                raise ValueError("GGML gated delta binding requires contiguous BF16 storage bits")
            if q.ndim!=4 or k.shape!=q.shape or v.ndim!=4:
                raise ValueError("GGML gated delta requires Q/K/V in [B,T,H,D] order")
            batch,tokens,key_heads,dimension=q.shape
            value_heads=v.shape[2]
            if (v.shape!=(batch,tokens,value_heads,dimension) or value_heads%key_heads or
                    gate.shape!=(batch,tokens,value_heads) or beta.shape!=gate.shape):
                raise ValueError("GGML gated delta requires equal key/value dimensions and grouped heads")
            pointers=(C.c_void_p*5)(*(value.ctypes.data for value in self.inputs))
            self.context=self.lib.bench_gated_delta_create(pointers,batch,tokens,key_heads,value_heads,
                                                         dimension,request["threads"])
            if not self.context:raise RuntimeError(self.lib.bench_error().decode())
            self.output=np.empty(v.shape,dtype=np.uint16)
            self.state=np.empty((batch,value_heads,dimension,dimension),dtype=np.float32)
            return
        self.a=np.load(root/"input0.npy");self.b=np.load(root/"input1.npy")
        self.kind={"gemm":0,"q4_projection":1,"rms_norm":2,"q4_dot":3,"q8_quantize":4,
                   "q4_projection_packed":5,"q4_dot_packed":7,"q6_projection_packed":8}[request["operation"]]
        if self.kind in (5,7,8):
            expected_bytes = 210 if self.kind == 8 else 144
            if self.a.dtype!=np.uint8 or self.a.ndim!=3 or self.a.shape[-1]!=expected_bytes or not self.a.flags.c_contiguous:
                raise ValueError(f'packed K-quant weights require contiguous [N,G,{expected_bytes}] bytes')
            n,groups,_=self.a.shape
            k=groups*256
            if not self.b.flags.c_contiguous:
                raise ValueError('the GGML activation must be contiguous')
            if self.kind==7:
                if self.b.dtype!=np.uint8 or self.b.shape!=(groups,292):
                    raise ValueError('packed Q8_K dot input must be [G,292] bytes')
                m=1
            else:
                if self.b.dtype!=np.float32 or self.b.ndim!=2 or not (
                    self.b.shape==(groups,256) or self.b.shape[1]==k):
                    raise ValueError('FP32 activation must match the packed Q4_K reduction extent')
                self.b=self.b.reshape(-1,k)
                m=self.b.shape[0]
        else:
            m,k=self.b.shape;n=self.a.shape[0]
        self.context=self.lib.bench_create(self.kind,self.a.ctypes.data,self.b.ctypes.data,m,n,k,request["threads"])
        if not self.context:raise RuntimeError(self.lib.bench_error().decode())
        self.output=np.empty((m,n),dtype="float32") if self.kind!=4 else np.empty(m*k//256*292,dtype="uint8")
        self.reference=request.get("reference",False)

    def run(self):
        if self.lib.bench_run(self.context):raise RuntimeError(self.lib.bench_error().decode())

    def result(self):
        if self.kind==6:
            self.lib.bench_output(self.context,self.output.ctypes.data,0)
            self.lib.bench_state_output(self.context,self.state.ctypes.data)
            return {"items":[
                {"shape":list(self.output.shape),"torch_dtype":"bfloat16",
                 "data":base64.b64encode(self.output.tobytes()).decode()},
                {"shape":list(self.state.shape),"dtype":"float32",
                 "data":base64.b64encode(self.state.tobytes()).decode()}]}
        self.lib.bench_output(self.context,self.output.ctypes.data,int(self.reference))
        return {"shape":list(self.output.shape),"dtype":str(self.output.dtype),"data":base64.b64encode(self.output.tobytes()).decode()}

    def packed(self):
        if self.kind not in (1,3):raise ValueError('only Q4_K weights have this packed-input contract')
        weights=np.empty((self.a.shape[0],self.a.shape[1]//256,144),dtype='uint8')
        self.lib.bench_packed_weight(self.context,weights.ctypes.data)
        def record(data):
            return {'shape':list(data.shape),'dtype':'uint8','data':base64.b64encode(data.tobytes()).decode()}
        if self.kind==1:return record(weights)
        activation=np.empty((self.b.shape[1]//256,292),dtype='uint8')
        self.lib.bench_packed_activation(self.context,activation.ctypes.data)
        return {'items':[record(weights),record(activation)]}

    def measure(self,samples,warmup,cache):
        min_ms=0.05
        max_repeat=65536
        repeat=self._repeat or 1
        for _ in range(warmup):
            if self.lib.bench_measure(self.context,repeat)<0:raise RuntimeError(self.lib.bench_error().decode())
        eviction=np.empty(64*1024*1024,dtype=np.uint8) if cache=='flush' else None
        values=[]
        for _ in range(samples):
            if self._repeat is None:
                while True:
                    if eviction is not None:eviction.fill(0)
                    elapsed=self.lib.bench_measure(self.context,repeat)
                    if elapsed<0:raise RuntimeError(self.lib.bench_error().decode())
                    if elapsed>0 and elapsed*repeat>=min_ms:
                        self._repeat=repeat
                        break
                    if repeat>=max_repeat:
                        if elapsed<=0:raise RuntimeError('ggml timer returned non-positive duration')
                        self._repeat=repeat
                        break
                    repeat=min(max_repeat,repeat*2 if elapsed<=0 else max(repeat*2,int(min_ms/elapsed)+1))
            else:
                if eviction is not None:eviction.fill(0)
                elapsed=self.lib.bench_measure(self.context,repeat)
                if elapsed<0:raise RuntimeError(self.lib.bench_error().decode())
                if elapsed<=0:raise RuntimeError('ggml timer returned non-positive duration')
            values.append(elapsed)
        return values

    def close(self):self.lib.bench_destroy(self.context)


if __name__=="__main__":serve(Worker)
