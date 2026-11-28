"""Group 7 GPU-authored workloads, independent mathematical oracles, and mature baselines."""

from __future__ import annotations

from functools import partial
import math
import numpy as np

from ..implementations.context import IntegerTolerance, Tolerance
from ..implementations.prepared import attach_device_timer
from ..model import Case, EnvironmentUnavailable, Implementation, PreparedCall, Workload

V2_EXITS = (
    "intent-v2:triton",
    "intent-v2:cutile",
    "intent-v2:mojo",
    "intent-v2:weft",
    "intent-v2:bangc",
)

BASELINES = {
    "sliding_window_attention": ("cutile", "torch", "aten:cpu"),
    "alibi_attention": ("torch", "aten:cpu"),
    "cross_entropy": ("triton", "torch", "inductor", "aten:cpu"),
    "topk_rows": ("torch", "aten:cpu"),
    "qk_norm_rope": ("inductor", "torch", "aten:cpu"),
    "attention_unscaled_max": ("cutile", "torch", "aten:cpu"),
}

FAMILIES = {
    "sliding_window_attention": "attention",
    "alibi_attention": "attention",
    "cross_entropy": "loss",
    "topk_rows": "selection",
    "qk_norm_rope": "normalization",
    "attention_unscaled_max": "attention",
}


def registrations():
    return {
        name: {
            "name": "group7/" + name,
            "family": FAMILIES[name],
            "implementations": [*BASELINES[name], *V2_EXITS],
        }
        for name in BASELINES
    }


def _has_torch() -> bool:
    from importlib.util import find_spec

    return find_spec("torch") is not None


def _wrap_array(arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    if _has_torch():
        import torch

        return torch.from_numpy(arr)
    return arr


def _to_torch_data(data):
    import torch

    def conv(x):
        if not isinstance(x, np.ndarray):
            return x
        c = np.ascontiguousarray(x)
        if c.dtype.name == "bfloat16":
            return torch.from_numpy(c.view(np.uint16)).view(torch.bfloat16).clone()
        return torch.from_numpy(c)

    return tuple(conv(v) for v in data)


def _bf16_tensor(rng, shape, scale=1.0, bias=0.0):
    arr = rng.standard_normal(shape, dtype=np.float32) * scale
    if _has_torch():
        import torch

        t = torch.from_numpy(arr).to(torch.bfloat16)
        return t + bias if bias else t
    import ml_dtypes

    out = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16))
    return np.ascontiguousarray(out + ml_dtypes.bfloat16(bias)) if bias else out


def _f16_tensor(rng, shape, scale=1.0):
    arr = (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float16)
    return _wrap_array(arr)


def _resolve_device(env, impl):
    if impl in {"aten:cpu", "intent-v2:mojo", "intent-v2:weft"}:
        if env.device not in {"auto", "cpu"}:
            raise EnvironmentUnavailable(f"{impl} requires a CPU environment")
        return "cpu"
    if impl == "intent-v2:bangc":
        device = "mlu:0" if env.device == "auto" else env.device
        if not device.startswith("mlu:"):
            raise EnvironmentUnavailable(f"{impl} requires an MLU environment")
        return device
    if impl in {"triton", "cutile", "inductor", "intent-v2:triton", "intent-v2:cutile"}:
        device = "cuda:0" if env.device == "auto" else env.device
        if not device.startswith("cuda:"):
            raise EnvironmentUnavailable(f"{impl} requires a CUDA environment")
        return device
    if env.device == "auto":
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return env.device


def _transfer(args, device):
    if device == "cpu" and not _has_torch():
        return tuple(
            np.ascontiguousarray(x) if isinstance(x, np.ndarray) else x for x in args
        )
    import torch

    if device.startswith("mlu:"):
        import torch_mlu  # noqa: F401
    return tuple(
        x.to(device).contiguous() if isinstance(x, torch.Tensor) else x for x in args
    )


def _prepare_v2(name, backend, args, env, device):
    import intentdsl
    from intentdsl import examples

    target = env.options.get("compilerinfra", {})
    math_kwargs = {k: target[k] for k in ("fpmath_mode", "fast_math", "denormal_fp_math", "numerics") if k in target}
    kernel = intentdsl.jit(
        getattr(examples, name),
        backend=backend,
        library=target.get("library"),
        configuration=target.get("configuration"),
        **math_kwargs,
    )
    call = kernel.prepare(*args, artifact_file=target.get("artifact_file"))
    metadata = {"author_program": name, "operator_entry": "prepared_execute"}

    def compile_fn():
        call.compile()
        metadata["cold_compile"] = call.compilation_report()

    execute = getattr(call, "execute", call.launch)
    prepared = PreparedCall(
        execute,
        call.result,
        compile=compile_fn,
        device=device,
        tuning_report=call.tuning_report,
        metadata=metadata,
    )
    if device == "cpu":
        from ..implementations.native_sampling import sample_timer
        from ..implementations.native_sequence import NativeSequence

        timer = None

        def measure(settings):
            nonlocal timer
            if timer is None:
                timer = NativeSequence((call,))
            return sample_timer(timer, settings, None, metadata=metadata)

        prepared.native_measure = measure
        prepared.close = lambda: timer and timer.close()
        prepared.timing = "cpu_native_monotonic"
    elif device.startswith("mlu:"):
        import torch
        from ..implementations.mlu_rows import timer
        from .programs import _tree

        stream = torch.mlu.current_stream(device)
        prepared.synchronize = stream.synchronize
        prepared.native_measure = lambda settings: timer(execute, stream, settings)
        prepared.timing = "cnrt_notifier"
        prepared.result = lambda: _tree(call.result(), lambda v: v.cpu())
    return prepared


def _inputs(name, case, seed):
    p = case.parameters
    rng = np.random.default_rng(seed)
    if name == "sliding_window_attention":
        b, hq, hk, s, d, w = (
            p["batch"],
            p["q_heads"],
            p["kv_heads"],
            p["seq"],
            p["head_dim"],
            p["window"],
        )
        return (
            _f16_tensor(rng, (b, hq, s, d), 0.5),
            _f16_tensor(rng, (b, hk, s, d), 0.5),
            _f16_tensor(rng, (b, hk, s, d), 0.5),
            1.0 / math.sqrt(d),
            int(w),
        )
    if name == "alibi_attention":
        b, hq, hk, s, d = (
            p["batch"],
            p["q_heads"],
            p["kv_heads"],
            p["seq"],
            p["head_dim"],
        )
        slopes = (2.0 ** (-8.0 * np.arange(1, hq + 1, dtype=np.float32) / hq)).astype(
            np.float32
        )
        return (
            _bf16_tensor(rng, (b, hq, s, d), 0.5),
            _bf16_tensor(rng, (b, hk, s, d), 0.5),
            _bf16_tensor(rng, (b, hk, s, d), 0.5),
            _wrap_array(slopes),
            1.0 / math.sqrt(d),
        )
    if name == "cross_entropy":
        m, v = p["rows"], p["vocab"]
        logits = _bf16_tensor(rng, (m, v), 1.0)
        targets = _wrap_array(rng.integers(0, v, size=(m,), dtype=np.int32))
        grad_loss = _wrap_array(rng.uniform(0.1, 1.5, size=(m,)).astype(np.float32))
        return (logits, targets, grad_loss)
    if name == "topk_rows":
        m, v = p["rows"], p["cols"]
        base = rng.standard_normal((m, v), dtype=np.float32)
        # Ensure distinct values so tie-breaking never masks genuine selection bugs.
        jitter = (np.arange(v, dtype=np.float32)[None, :] * 1e-6)
        return (_wrap_array((base + jitter).astype(np.float32)),)
    if name == "qk_norm_rope":
        b, s = p["batch"], p["seq"]
        return (
            _bf16_tensor(rng, (b, 32, s, 128), 1.0),
            _bf16_tensor(rng, (b, 8, s, 128), 1.0),
            _bf16_tensor(rng, (128,), 0.5, bias=1.0),
            _bf16_tensor(rng, (128,), 0.5, bias=1.0),
            _bf16_tensor(rng, (s, 64), 0.5),
            _bf16_tensor(rng, (s, 64), 0.5),
            1.0 / 128.0,
            1e-6,
        )
    if name == "attention_unscaled_max":
        b, hq, hk, s, d = (
            p["batch"],
            p["q_heads"],
            p["kv_heads"],
            p["seq"],
            p["head_dim"],
        )
        return (
            _bf16_tensor(rng, (b, hq, s, d), 0.5),
            _bf16_tensor(rng, (b, hk, s, d), 0.5),
            _bf16_tensor(rng, (b, hk, s, d), 0.5),
            1.0 / math.sqrt(d),
        )
    raise KeyError(name)


def _oracle(name, case, data):
    import torch

    data = _to_torch_data(data)

    if name == "sliding_window_attention":
        q, k, v, scale, window = data
        qf, kf, vf = q.float(), k.float(), v.float()
        group = q.shape[1] // k.shape[1]
        kf = kf.repeat_interleave(group, dim=1)
        vf = vf.repeat_interleave(group, dim=1)
        s = q.shape[2]
        q_idx = torch.arange(s)[:, None]
        k_idx = torch.arange(k.shape[2])[None, :]
        mask = (k_idx <= q_idx) & (k_idx > (q_idx - window))
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(torch.float16).float()
        return torch.matmul(probs, vf).to(torch.float16)

    if name == "alibi_attention":
        q, k, v, slopes, scale = data
        qf, kf, vf = q.float(), k.float(), v.float()
        group = q.shape[1] // k.shape[1]
        kf = kf.repeat_interleave(group, dim=1)
        vf = vf.repeat_interleave(group, dim=1)
        s = q.shape[2]
        q_idx = torch.arange(s, dtype=torch.float32)[:, None]
        k_idx = torch.arange(k.shape[2], dtype=torch.float32)[None, :]
        bias = slopes[:, None, None] * (k_idx - q_idx)[None, :, :]
        causal = k_idx <= q_idx
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale + bias[None, :, :, :]
        scores = scores.masked_fill(~causal[None, None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(torch.bfloat16).float()
        return torch.matmul(probs, vf).to(torch.bfloat16)

    if name == "cross_entropy":
        logits, targets, grad_loss = data
        x = logits.double()
        t = targets.long()
        g = grad_loss.double()
        lse = torch.logsumexp(x, dim=-1)
        target_logit = x.gather(1, t[:, None]).squeeze(1)
        loss = (lse - target_logit).float()
        probs = torch.exp(x - lse[:, None])
        probs.scatter_add_(1, t[:, None], torch.full((x.shape[0], 1), -1.0, dtype=torch.float64))
        grad_logits = (probs * g[:, None]).to(torch.bfloat16)
        return loss, grad_logits

    if name == "topk_rows":
        (logits,) = data
        arr = logits.numpy()
        idx = np.argsort(-arr, axis=-1, kind="stable")[:, :8].astype(np.int32)
        vals = np.take_along_axis(arr, idx, axis=-1).astype(np.float32)
        return torch.from_numpy(vals), torch.from_numpy(idx)

    if name == "qk_norm_rope":
        q, k, qw, kw, cos, sin, inv_d, eps = data

        def norm_rope(x, w):
            xf = x.float()
            wf = w.float()
            var = (xf * xf).sum(dim=-1, keepdim=True) * inv_d
            xn = xf * torch.rsqrt(var + eps) * wf
            x0, x1 = xn[..., :64], xn[..., 64:]
            cf = cos.float()[None, None, :, :]
            sf = sin.float()[None, None, :, :]
            y0 = (x0 * cf - x1 * sf).to(torch.bfloat16)
            y1 = (x1 * cf + x0 * sf).to(torch.bfloat16)
            return torch.cat((y0, y1), dim=-1)

        return norm_rope(q, qw), norm_rope(k, kw)

    if name == "attention_unscaled_max":
        q, k, v, scale = data
        qf, kf, vf = q.float(), k.float(), v.float()
        group = q.shape[1] // k.shape[1]
        kf = kf.repeat_interleave(group, dim=1)
        vf = vf.repeat_interleave(group, dim=1)
        s = q.shape[2]
        q_idx = torch.arange(s)[:, None]
        k_idx = torch.arange(k.shape[2])[None, :]
        scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
        scores = scores.masked_fill(~(k_idx <= q_idx)[None, None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1).to(torch.bfloat16).float()
        return torch.matmul(probs, vf).to(torch.bfloat16)

    raise KeyError(name)


def _prepare_baseline(name, impl, case, data, env, device):
    import torch
    import torch.nn.functional as F

    args = _transfer(data, device)
    state: dict[str, object] = {}

    if name == "sliding_window_attention":
        q, k, v, scale, window = args
        if impl == "cutile":
            from ..corpus import PROJECT_ROOT
            from ..implementations.context import Context
            from ..source_support import tilegym_source

            ctx = Context(env.compiler, PROJECT_ROOT, None, "cutile", environment=env)
            module = tilegym_source(
                ctx,
                "corpus/gpu/baselines/cutile/tilegym/attention/sliding_window/swa_attention.py",
                "group7_swa",
            )

            def launch():
                state["out"] = module.tile_swa_attention(
                    q, k, v, window_size=window, scaling=scale, is_causal=True
                )

            return PreparedCall(
                launch,
                lambda: state["out"],
                device=device,
                owners=(args, module),
                metadata={"library": "NVIDIA TileGym tile_swa_attention"},
            )
        s = q.shape[2]
        q_idx = torch.arange(s, device=device)[:, None]
        k_idx = torch.arange(k.shape[2], device=device)[None, :]
        mask = (k_idx <= q_idx) & (k_idx > (q_idx - window))

        def launch():
            state["out"] = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=scale, enable_gqa=True
            )

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=(args, mask),
            metadata={"library": f"PyTorch SDPA ({device})"},
        )

    if name == "alibi_attention":
        q, k, v, slopes, scale = args
        s = q.shape[2]
        q_idx = torch.arange(s, device=device, dtype=torch.float32)[:, None]
        k_idx = torch.arange(k.shape[2], device=device, dtype=torch.float32)[None, :]
        bias = slopes[:, None, None] * (k_idx - q_idx)[None, :, :]
        bias = bias.masked_fill(~(k_idx <= q_idx)[None, :, :], float("-inf")).to(q.dtype)
        bias4d = bias.unsqueeze(0).contiguous()

        def launch():
            state["out"] = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias4d, scale=scale, enable_gqa=True
            )

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=(args, bias4d),
            metadata={"library": f"PyTorch SDPA ALiBi ({device})"},
        )

    if name == "cross_entropy":
        logits, targets, grad_loss = args
        targets_i64 = targets.to(torch.int64)
        if impl == "triton":
            from ..corpus import PROJECT_ROOT
            from ..corpus._common.loading import load_module

            runtime = load_module(
                PROJECT_ROOT
                / "corpus/gpu/baselines/triton/liger-kernel/loss/cross_entropy/cross_entropy_runtime.py",
                "group7_liger_cross_entropy",
            )
            source = runtime.source
            rows, vocab = logits.shape
            block = min(source.MAX_FUSED_SIZE, source.triton.next_power_of_2(vocab))
            loss_buf = torch.empty((rows,), device=device, dtype=torch.float32)

            def launch():
                # Out-of-place contract: preserve caller's `logits` buffer.
                grad_buf = logits.clone()
                source.liger_cross_entropy_kernel[(rows,)](
                    X_ptr=grad_buf,
                    X_stride=grad_buf.stride(0),
                    Y_ptr=targets_i64,
                    Y_stride=targets_i64.stride(0),
                    weight_ptr=None,
                    loss_ptr=loss_buf,
                    z_loss_ptr=None,
                    loss_stride=loss_buf.stride(0),
                    token_accuracy_ptr=None,
                    token_accuracy_stride=0,
                    predicted_tokens_ptr=None,
                    predicted_tokens_stride=0,
                    n_cols=vocab,
                    n_non_ignore=rows,
                    sum_non_ignore_weight=rows,
                    ignore_index=-100,
                    weight_sum=0.0,
                    lse_square_scale=0.0,
                    label_smoothing=0.0,
                    reduction="sum",
                    softcap=None,
                    RETURN_Z_LOSS=False,
                    RETURN_TOKEN_ACCURACY=False,
                    RETURN_PREDICTED_TOKENS=False,
                    BLOCK_SIZE=block,
                    HAS_WEIGHT=False,
                    HAS_SOFTCAPPING=False,
                    HAS_GRADIENTS=True,
                    num_warps=32,
                )
                grad_buf.mul_(grad_loss[:, None].to(grad_buf.dtype))
                state["out"] = (loss_buf, grad_buf)

            return PreparedCall(
                launch,
                lambda: state["out"],
                device=device,
                owners=(args, targets_i64, loss_buf),
                metadata={"library": "Liger fused Triton cross-entropy"},
            )

        def eager_ce(x, t, g):
            req = x.detach().requires_grad_(True)
            loss = F.cross_entropy(req.float(), t, reduction="none")
            (dx,) = torch.autograd.grad(loss, req, grad_outputs=g)
            return loss, dx

        fn = torch.compile(eager_ce) if impl == "inductor" else eager_ce

        def launch():
            state["out"] = fn(logits, targets_i64, grad_loss)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=(args, targets_i64),
            metadata={"library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen'} cross-entropy"},
        )

    if name == "topk_rows":
        (logits,) = args

        def launch():
            vals, idx = torch.topk(logits, 8, dim=-1, largest=True, sorted=True)
            state["out"] = (vals, idx.to(torch.int32))

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=args,
            metadata={"library": f"PyTorch ATen topk ({device})"},
        )

    if name == "qk_norm_rope":
        q, k, qw, kw, cos, sin, inv_d, eps = args

        if inv_d != 1. / q.shape[-1]:
            raise ValueError("native RMSNorm requires the reciprocal feature width")

        def eager_qk(q_in, k_in, qw_in, kw_in, cos_in, sin_in):
            qn = F.rms_norm(q_in.float(), (128,), qw_in.float(), eps)
            kn = F.rms_norm(k_in.float(), (128,), kw_in.float(), eps)
            cf = cos_in.float()[None, None, :, :]
            sf = sin_in.float()[None, None, :, :]
            q0, q1 = qn[..., :64], qn[..., 64:]
            k0, k1 = kn[..., :64], kn[..., 64:]
            qo = torch.cat((q0 * cf - q1 * sf, q1 * cf + q0 * sf), dim=-1).to(
                torch.bfloat16
            )
            ko = torch.cat((k0 * cf - k1 * sf, k1 * cf + k0 * sf), dim=-1).to(
                torch.bfloat16
            )
            return qo, ko

        fn = torch.compile(eager_qk) if impl == "inductor" else eager_qk

        def launch():
            state["out"] = fn(q, k, qw, kw, cos, sin)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=args,
            metadata={"library": f"PyTorch {'Inductor fused' if impl == 'inductor' else 'ATen'} QK-RMSNorm+RoPE"},
        )

    if name == "attention_unscaled_max":
        q, k, v, scale = args
        if impl == "cutile":
            from ..corpus import PROJECT_ROOT
            from ..implementations.context import Context
            from ..source_support import tilegym_source

            ctx = Context(env.compiler, PROJECT_ROOT, None, "cutile", environment=env)
            module = tilegym_source(
                ctx,
                "corpus/gpu/baselines/cutile/tilegym/attention/dense/attention.py",
                "group7_fmha_variant",
            )

            def launch():
                state["out"] = module.tile_fmha(q, k, v, scaling=scale, is_causal=True)

            return PreparedCall(
                launch,
                lambda: state["out"],
                device=device,
                owners=(args, module),
                metadata={"library": "NVIDIA TileGym causal GQA FMHA"},
            )

        def launch():
            state["out"] = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=scale, enable_gqa=True
            )

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=args,
            metadata={"library": f"PyTorch Flash/cuDNN SDPA ({device})"},
        )

    raise KeyError((name, impl))


def build(name):
    cases_map = {
        "sliding_window_attention": [
            Case(
                "2x32x8x4096x128-w1024-f16",
                dict(batch=2, q_heads=32, kv_heads=8, seq=4096, head_dim=128, window=1024),
            )
        ],
        "alibi_attention": [
            Case(
                "2x32x8x2048x128-bf16",
                dict(batch=2, q_heads=32, kv_heads=8, seq=2048, head_dim=128),
            )
        ],
        "cross_entropy": [Case("4096x32768-bf16", dict(rows=4096, vocab=32768))],
        "topk_rows": [Case("1024x4096-k8-f32", dict(rows=1024, cols=4096, k=8))],
        "qk_norm_rope": [Case("4x32x8x2048x128-bf16", dict(batch=4, seq=2048))],
        "attention_unscaled_max": [
            Case(
                "2x32x8x2048x128-bf16",
                dict(batch=2, q_heads=32, kv_heads=8, seq=2048, head_dim=128),
            )
        ],
    }
    tolerances = {
        "sliding_window_attention": Tolerance(0.05, 0.02),
        "alibi_attention": Tolerance(0.05, 0.02),
        "cross_entropy": (Tolerance(0.02, 0.01), Tolerance(0.02, 0.02)),
        "topk_rows": (Tolerance(1e-6, 1e-6), IntegerTolerance(0)),
        "qk_norm_rope": (Tolerance(0.03, 0.02), Tolerance(0.03, 0.02)),
        "attention_unscaled_max": Tolerance(0.05, 0.02),
    }

    def prepare(impl, case, data, env):
        device = _resolve_device(env, impl)
        if impl.startswith("intent-v2:"):
            backend = impl.partition(":")[2]
            return _prepare_v2(name, backend, _transfer(data, device), env, device)
        prepared = _prepare_baseline(name, impl, case, data, env, device)
        return attach_device_timer(prepared)

    impls = {
        impl: Implementation(impl, partial(prepare, impl))
        for impl in (*BASELINES[name], *V2_EXITS)
    }

    from ..references.aten_rows import registrations
    impls.update(registrations("group7/" + name))

    def check(actual, expected):
        import torch
        from ..numeric import compare_outputs
        from .programs import _tree

        def to_tensor(v):
            if isinstance(v, np.ndarray):
                if v.dtype.name == "bfloat16":
                    return torch.from_numpy(v.view(np.uint16)).view(torch.bfloat16)
                return torch.from_numpy(v)
            return v

        act = _tree(actual, lambda v: to_tensor(v).detach().cpu())
        exp = _tree(expected, lambda v: to_tensor(v).detach().cpu())
        compare_outputs(act, exp, tolerances[name])

    return Workload(
        "group7/" + name,
        cases_map[name],
        partial(_inputs, name),
        impls,
        oracle=partial(_oracle, name),
        check=check,
        family=FAMILIES[name],
        default_baseline=BASELINES[name][0],
    )
