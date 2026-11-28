import numpy as np
from ..model import Case,Implementation,Workload

NAMES=("gemm","q4_projection","rms_norm","q4_dot","q8_quantize")


def q8_reference(case,inputs):
    from ..references.quantization import q8_records
    return q8_records(inputs[1]).ravel()


def build(name):
    if name not in NAMES:raise KeyError(name)
    cases=[Case("decode",{"m":1,"n":128,"k":1024}),Case("prefill",{"m":128,"n":128,"k":1024})]
    if name=="rms_norm":cases=[Case("rows",{"m":32,"n":1024,"k":1024})]
    if name in {"q4_dot","q8_quantize"}:cases=[Case("vector",{"m":1,"n":1,"k":4096})]
    def inputs(case,seed):
        rng=np.random.default_rng(seed);p=case.parameters
        return (rng.normal(size=(p["n"],p["k"])).astype("float32"),rng.normal(size=(p["m"],p["k"])).astype("float32"))
    def prepare(case,data,env,reference=False):
        from ..implementations.native_libraries import ggml
        operation = name + "_packed" if name in {"q4_dot","q4_projection"} else name
        return ggml(operation,case,data,env,reference=reference)
    if name=="q8_quantize":oracle=q8_reference
    elif name=="rms_norm":oracle=lambda case,x:x[1]/np.sqrt((x[1]**2).mean(-1,keepdims=True)+1e-5)
    elif name=="gemm":oracle=lambda case,x:x[1].astype("float64")@x[0].astype("float64").T
    else:oracle=Implementation("dequantized_scalar_oracle",lambda case,data,env:prepare(case,data,env,True))
    def generated(case,data,env):
        from ..implementations.weft_intent import prepare
        if name=='gemm':
            weights, activations = data
            return prepare('intent.examples.kernels.contraction.gemm:linear_f32',
                           (activations, weights), env)
        if name=='rms_norm':
            return prepare('intent.examples.kernels.normalization.rms_norm:rms_norm_f32',(data[1],1/data[1].shape[-1],1e-5),env)
        prefix='intent.examples.kernels.quantization.quantized_projection:'
        if name=='q8_quantize':
            call=prepare(prefix+name,(data[1].reshape(-1,256),),env)
            result=call.result;call.result=lambda:result().reshape(-1)
            return call
        weights, activation = data
        if name=='q4_projection':args=(weights,activation.reshape(activation.shape[0],-1,256))
        else:args=(weights[0],activation.reshape(-1,292))
        return prepare(prefix+name,args,env)
    def generated_v2(case, data, env):
        from ..implementations.v2_program import bind
        if name == 'gemm':
            weights, activations = data
            return bind('linear_f32', 'weft', (activations, weights), env, 'cpu')
        if name == 'rms_norm':
            return bind('rms_norm_f32', 'weft', (data[1], 1.0 / data[1].shape[-1], 1e-5), env, 'cpu')
        if name == 'q8_quantize':
            call = bind('q8_quantize', 'weft', (data[1].reshape(-1, 256),), env, 'cpu')
            result = call.result
            call.result = lambda: result().reshape(-1)
            return call
        weights, activation = data
        if name == 'q4_projection':
            return bind('q4_projection_batched', 'weft', (weights, activation.reshape(activation.shape[0], -1, 256)), env, 'cpu')
        call = bind('q4_dot', 'weft', (weights, activation.reshape(-1, 292)), env, 'cpu')
        result = call.result
        call.result = lambda: result().reshape(1, -1)
        return call
    def encode(case, data, env):
        from ..implementations.native_libraries import pack_q4
        return pack_q4(data,env,packed_activation=name=="q4_dot")
    return Workload("rvv/"+name,cases,inputs,{"ggml":Implementation("ggml",prepare),
                    "intent:weft":Implementation("intent:weft",generated),
                    "intent-v2:weft":Implementation("intent-v2:weft",generated_v2)},oracle=oracle,
                    atol=0 if name=="q8_quantize" else 2e-3,
                    rtol=0 if name=="q8_quantize" else 2e-3,family="rvv/"+name,
                    prepare_inputs=encode if name in {"q4_dot","q4_projection"} else None)
