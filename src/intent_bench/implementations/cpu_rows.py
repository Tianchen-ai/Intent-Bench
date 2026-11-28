from functools import partial
import numpy as np
from ..model import Implementation, PreparedCall


def register(name):
    return {
        impl: Implementation(impl, partial(prepare, name, impl))
        for impl in (
            "ggml",
            "intent:weft",
            "intent:mojo",
            "intent-v2:weft",
            "intent-v2:mojo",
        )
    }


def prepare(name, impl, case, args, env):
    if env.device != "cpu":
        raise ValueError("row CPU implementation requires a CPU environment")
    if impl == "ggml":
        from .native_libraries import ggml

        return ggml(name, case, args, env)
    is_soft = name == "softmax"
    if impl.startswith("intent-v2:"):
        import intentdsl
        from intentdsl.examples import relu, relu_f32, rms_norm, softmax, softmax_f32
        from .native_sequence import NativeSequence
        from .native_sampling import sample_timer

        backend = impl.split(":")[1]
        target = env.options.get("compilerinfra", {})
        math_kwargs = {k: target[k] for k in ("fpmath_mode", "fast_math", "denormal_fp_math", "numerics") if k in target}
        definition = (
            (softmax_f32 if args[0].dtype == np.float32 else softmax)
            if is_soft
            else (relu_f32 if args[0].dtype == np.float32 else relu)
            if name == "relu"
            else rms_norm
        )
        kernel = intentdsl.jit(
            definition,
            backend=backend,
            library=target.get("library"),
            configuration=target.get("configuration"),
            **math_kwargs,
        )
        call = kernel.prepare(*args, artifact_file=target.get("artifact_file"))
        metadata = {
            "candidate": target.get("candidate"),
            "operator_entry": "prepared_native_sequence",
        }
        timer = None

        def compile():
            call.compile()
            metadata["cold_compile"] = call.compilation_report()

        def measure(settings):
            nonlocal timer
            if timer is None:
                timer = NativeSequence((call,))
            return sample_timer(timer, settings, None, metadata=metadata)

        return PreparedCall(
            getattr(call, "execute", call.launch),
            call.result,
            compile=compile,
            device="cpu",
            native_measure=measure,
            timing="cpu_native_monotonic",
            tuning_report=call.tuning_report,
            metadata=metadata,
            close=lambda: timer and timer.close(),
        )
    symbol = (
        (
            "normalization.softmax:"
            + (
                "stable_softmax"
                if args[0].dtype == np.float32
                else "stable_softmax_f16"
            )
        )
        if is_soft
        else (
            "activation.pointwise:"
            + ("relu_forward_f32" if args[0].dtype == np.float32 else "relu_forward")
        )
        if name == "relu"
        else "normalization.rms_norm:weighted_rms_norm"
    )
    ref = "intent.examples.kernels." + symbol
    if impl == "intent:weft":
        from .weft_intent import prepare as weft

        return weft(ref, args, env)
    import torch
    from ..loading import load
    from .intent import make_kernel
    from .prepared import intent_call

    return intent_call(
        make_kernel(load(ref), env, "mojo"),
        tuple(torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in args),
        device="cpu",
    )
