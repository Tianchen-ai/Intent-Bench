"""The original six RVV cases with independent Weft and Intent preparation."""
import random
import struct
import numpy as np
from ..model import Case,EnvironmentUnavailable,Implementation,Workload


def check_execution(environment, requires_ime):
    if environment.threads != 1:
        raise EnvironmentUnavailable('the original Weft cases specify a single CPU worker')
    if requires_ime:
        from triton.backends.weft.driver import WeftDriver
        target = WeftDriver().get_current_target()
        if target.arch != 'riscv64' or target.matrix_extension in (None, 'none'):
            raise EnvironmentUnavailable('the registered IME case requires a native RISC-V matrix extension on the selected CPUs')


def q4_projection_reference(weights,activation):
    n,g,_=weights.shape
    d=weights[...,:2].copy().view('<f2').reshape(n,g).astype('float32')
    dmin=weights[...,2:4].copy().view('<f2').reshape(n,g).astype('float32')
    packed=weights[...,4:16].astype('int32');qs=weights[...,16:]
    scales=np.empty((n,g,8),dtype='float32');minimum=np.empty_like(scales)
    for j in range(8):
        if j<4:scales[...,j]=packed[...,j]&63;minimum[...,j]=packed[...,j+4]&63
        else:
            scales[...,j]=(packed[...,j+4]&15)|((packed[...,j-4]>>6)<<4)
            minimum[...,j]=(packed[...,j+4]>>4)|((packed[...,j]>>6)<<4)
    dequant=np.empty((n,g,256),dtype='float32')
    for j in range(8):
        values=(qs[...,j//2*32:(j//2+1)*32] >> (4*(j%2))) & 15
        dequant[...,j*32:(j+1)*32]=d[...,None]*scales[...,j,None]*values-dmin[...,None]*minimum[...,j,None]
    from ..references.quantization import q8_fields
    scale, codes = q8_fields(activation)
    quantized = codes * scale[:, None]
    return np.einsum('ngk,gk->n',dequant,quantized,dtype='float64').astype('float32')


def build(row):
    name=row['kernel'];projection=name.startswith('q4_k');matrix=name.startswith('i8_')
    linear=name=='causal_linear_attention_f32';ime=matrix or name.endswith('_ime')
    def inputs(case,seed):
        rng=np.random.default_rng(seed)
        if projection:
            rng=random.Random(seed);n=k=4096;g=k//256
            packed=bytearray(rng.randbytes(n*g*144))
            for offset in range(0,len(packed),144):struct.pack_into('<ee',packed,offset,rng.uniform(.001,.004),rng.uniform(.002,.008))
            return np.frombuffer(packed,dtype='uint8').reshape(n,g,144),np.array([rng.uniform(-1,1) for _ in range(k)],dtype='float32').reshape(g,256)
        if matrix:
            m=1 if name=='i8_gemv_bias' else 128
            return rng.integers(-128,128,(m,4096),dtype='int8'),rng.integers(-128,128,(4096,4096),dtype='int8'),rng.integers(-1024,1024,(4096,),dtype='int32')
        data=tuple(rng.normal(size=(8,128,32)).astype('float32') for _ in range(3))
        return data if linear else (*data,32**-.5)
    def source(case,args,env):
        from ..implementations.triton_weft_source import prepare
        check_execution(env, ime)
        prefix='corpus/cpu/baselines/triton_weft/'
        if projection:
            n,g,_=args[0].shape
            return prepare((prefix+'q4_k_projection.py','production_mul_mat_q4_k'),args,
                [{'shape':[n],'dtype':'float32'}],
                [{'buffer':0,'shape':[n,g*256],'encoding':'Q4_K'},
                 {'buffer':1,'shape':[1,g*256]},
                 {'workspace':g*292,'shape':[1,g*256],'encoding':'Q8_K'},
                 {'buffer':2,'shape':[1,n]}],env)
        if matrix:
            a,b,bias=args;m,k=a.shape;n=b.shape[1]
            return prepare((prefix+'i8.py','gemm_bias'),args,[{'shape':[m,n],'dtype':'int32'}],
                [{'buffer':i} for i in range(4)],env,use_matrix=True,meta=(('MR',1 if m==1 else 4),))
        b,s,d=args[0].shape;dv=args[2].shape[-1]
        family='linear' if linear else 'causal';arguments=args if linear else (*args[:3],np.array([args[3]],dtype='float32'))
        outputs=[{'shape':[b,s,dv],'dtype':'float32'}]
        if linear:outputs.append({'shape':[b,d,dv],'dtype':'float32'})
        bindings=[{'buffer':i} for i in ([0,1,2,3,4] if linear else [0,1,2,4,3])]
        return prepare((prefix+f'{family}.py','causal_linear_attention' if linear else 'causal_attention'),
            arguments,outputs,bindings,env)
    def generated(case,args,env):
        from ..implementations.weft_intent import prepare
        check_execution(env, ime)
        reference=('intent.examples.kernels.quantization.quantized_projection:quantized_projection' if projection else
                   'intent.examples.kernels.contraction.gemm:gemm_i8' if matrix else 'intent.examples.kernels.streaming.attention_f32:'+name)
        return prepare(reference,args,env)
    def oracle(case,args):
        if projection:return q4_projection_reference(*args)
        # Every i8 product and partial sum fits exactly in a float64 integer.
        if matrix:return (args[0].astype('float64')@args[1].astype('float64')+args[2]).astype('int32')
        q,k,v=args[:3]
        if linear:
            states=np.einsum('bsd,bsv->bsdv',k,v).cumsum(1)
            return np.einsum('bsd,bsdv->bsv',q,states),states[:,-1]
        scores=(q@k.swapaxes(-1,-2))*args[3]
        scores=np.where(np.triu(np.ones((q.shape[1],k.shape[1]),dtype=bool),1),-np.inf,scores)
        probabilities=np.exp(scores-scores.max(-1,keepdims=True))
        return probabilities/probabilities.sum(-1,keepdims=True)@v
    implementations={'weft':Implementation('weft',source),'intent:weft':Implementation('intent:weft',generated)}
    from functools import partial
    from ..implementations.v2_program import bind
    from .programs import _device
    v2_symbol = ("q4_projection" if projection else
                 "int8_gemm" if matrix else
                 "causal_linear_attention" if linear else
                 "causal_attention_f32")
    def weft_v2_call(backend,case,args,env):
        device=_device(env,backend)
        if device=="cpu":
            check_execution(env,ime)
        else:
            import torch
            if device.startswith("mlu"):import torch_mlu
            args=tuple(torch.from_numpy(v).to(device) if isinstance(v,np.ndarray) else v for v in args)
        return bind(v2_symbol,backend,args,env,device)
    if projection and not ime:
        from ..references.aten_rows import registrations
        implementations.update(registrations(row["name"]))
    v2_backends = ("triton","cutile","mojo","weft","bangc") if (projection and not ime) else ("weft",)
    for backend in v2_backends:
        key="intent-v2:"+backend
        implementations[key]=Implementation(key,partial(weft_v2_call,backend))
    return Workload(row['name'],[Case(row['case'],{'profile':'original'})],inputs,
        implementations,oracle=oracle,
        atol=0 if matrix else 1e-4 if projection else 2e-4,rtol=0 if matrix else 2e-3 if projection else 1e-5,
        family=row['family'],parameter_keys=())
