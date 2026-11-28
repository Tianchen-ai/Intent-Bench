"""Weighted row normalization with an explicit inverse width and epsilon."""

import numpy as np
from ..model import Case, Implementation, Workload, PreparedCall
from ..implementations.prepared import intent_call
from ..implementations.intent import make_kernel


def inputs(case, seed):
    p = case.parameters
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((p["rows"], p["columns"]), dtype=np.float32),
        rng.standard_normal(p["columns"], dtype=np.float32),
        1.0 / p["columns"],
        1e-6,
    )


def tensors(data, device):
    from importlib.util import find_spec

    if device == "cpu" and find_spec("torch") is None:
        return tuple(
            np.ascontiguousarray(x) if isinstance(x, np.ndarray) else x for x in data
        )
    import torch

    return tuple(
        torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x for x in data
    )


def reference(case, data):
    x, w, inv, eps = data
    return (x * np.reciprocal(np.sqrt((x * x).sum(-1, keepdims=True) * inv + eps))) * w


def build(name):
    def prepare(impl, case, data, env):
        args = tensors(data, env.device)
        x, w, inv, eps = args
        if impl == "torch":
            import torch
            import torch.nn.functional as F

            # The declared case binds inv=1/N, as required by native RMSNorm.
            if inv != 1 / x.size(-1):
                raise ValueError("native RMSNorm requires inverse width")
            state = {}

            def launch():
                state["out"] = F.rms_norm(x, (x.size(-1),), w, eps)

            return PreparedCall(launch, lambda: state["out"], device=env.device)
        if impl in ("intent:triton","intent:cutile"):
            from intent.examples.kernels.normalization.rms_norm import weighted_rms_norm

            return intent_call(
                make_kernel(weighted_rms_norm, env, impl.partition(":")[2]), args, device=env.device
            )
        from intentdsl.examples import rms_norm
        from ..implementations.intent_v2 import prepare as prepare_v2

        return prepare_v2(rms_norm, args, env, backend=impl.partition(":")[2])

    impls = {
        k: Implementation(k, lambda c, d, e, k=k: prepare(k, c, d, e))
        for k in ("torch", "intent:triton", "intent-v2:triton", "intent:cutile","intent-v2:cutile")
    }
    from ..implementations.cpu_rows import register

    from ..references.aten_rows import registrations
    impls.update(registrations(name))
    impls.update(register(name))
    from ..implementations.mlu_rows import register as mlu_rows
    impls.update(mlu_rows(name))
    return Workload(
        name,
        [
            Case("8192x4096-f32", dict(rows=8192, columns=4096)),
            Case("32x1024-f32", dict(rows=32, columns=1024)),
        ],
        inputs,
        impls,
        oracle=reference,
        atol=5e-5,
        rtol=0,
        family="normalization",
    )
