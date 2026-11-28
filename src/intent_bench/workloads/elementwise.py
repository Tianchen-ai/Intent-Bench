from __future__ import annotations

import numpy as np

from ..model import Case, Implementation, PreparedCall, Workload, ExpressionGap, EnvironmentUnavailable
from ..loading import intent_environment


def _numpy(name, x):
    if name == "softmax":
        value = x.astype(np.float32)
        exp = np.exp(value - value.max(axis=-1, keepdims=True))
        return (exp / exp.sum(axis=-1, keepdims=True)).astype(x.dtype)
    if name == "relu":
        return np.maximum(x, 0)
    raise ValueError(name)


def _prepare(name, implementation, case, inputs, env):
    x, = inputs
    if implementation == "numpy":
        state = {}
        def run():
            state["output"] = _numpy(name, x)
        return PreparedCall(run, lambda: state["output"],scopes=("prepared_call","host_call"))
    import torch
    target = implementation.partition(":")[2]
    gpu = implementation in {"triton", "cutile"} or target in {"triton", "cutile"}
    device = env.device
    if device == "auto":
        device = "cuda:0" if gpu or (implementation == "torch" and torch.cuda.is_available()) else "cpu"
    if gpu and not device.startswith("cuda"):
        raise EnvironmentUnavailable(f"{implementation} requires --device cuda:N")
    x = torch.from_numpy(x).to(device)
    if implementation == "torch":
        state = {}
        def run():
            state["output"] = torch.softmax(x, dim=-1) if name == "softmax" else torch.relu(x)
        return PreparedCall(run, lambda: state["output"], device=device,scopes=("prepared_call","host_call"))
    if implementation in {"triton", "cutile"}:
        from ..implementations.gpu import prepare
        return prepare(name, implementation, x)
    if implementation.startswith("intent-v2:"):
        from intentdsl.examples import relu, relu_f32, softmax, softmax_f32
        from ..implementations.intent_v2 import prepare
        if name == "softmax":
            definition = softmax_f32 if x.dtype == torch.float32 else softmax
        else:
            definition = relu_f32 if x.dtype == torch.float32 else relu
        return prepare(definition, (x,), env, backend=target)
    intent = intent_environment(env)
    if name == "softmax":
        from intent.examples.kernels.normalization.softmax import stable_softmax, stable_softmax_f16
        definition = stable_softmax if x.dtype == torch.float32 else stable_softmax_f16
    else:
        from intent.examples.kernels.activation.pointwise import relu_forward, relu_forward_f32
        definition = relu_forward_f32 if x.dtype == torch.float32 else relu_forward
    from ..implementations.intent import make_kernel
    from ..implementations.prepared import intent_call
    artifact = make_kernel(definition, env, target)
    return intent_call(artifact, (x,), device=device)


def build(name="softmax"):
    cases = [Case(f"{m}x{n}-{dtype}", {"rows":m,"columns":n,"dtype":dtype})
             for m,n in ((1,128),(64,1024),(512,4096)) for dtype in ("float16", "float32")]
    def inputs(case, seed):
        p = case.parameters
        return (np.random.default_rng(seed).normal(size=(p["rows"],p["columns"])).astype(p["dtype"]),)
    impls = {}
    for impl in ("numpy", "torch", "triton", "cutile", "intent:triton", "intent:cutile", "intent:mojo"):
        impls[impl] = Implementation(impl, lambda case, args, env, impl=impl: _prepare(name, impl, case, args, env))
    from ..implementations.cpu_rows import register
    impls.update(register(name))
    if name == "softmax":
        from ..references.aten_rows import registrations
        impls.update(registrations(name))
        from ..implementations.mlu_rows import register as mlu_rows
        impls.update(mlu_rows(name))
        for key in ("intent-v2:triton","intent-v2:cutile"):
            impls[key] = Implementation(key,lambda c,d,e,k=key: _prepare(name,k,c,d,e))
    return Workload(name, cases, inputs, impls, oracle=lambda case,args: _numpy(name,*args),
                    atol=2e-3, rtol=2e-3, family="reduction" if name=="softmax" else "elementwise",
                    work=lambda case: {"bytes":2*case.parameters["rows"]*case.parameters["columns"]*np.dtype(case.parameters["dtype"]).itemsize})
