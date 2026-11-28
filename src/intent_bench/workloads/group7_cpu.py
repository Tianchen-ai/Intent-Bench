"""Group 7 CPU-authored workloads, independent mathematical oracles, and mature baselines."""

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
    "embedding_bag": ("aten:cpu", "inductor", "torch"),
    "index_add": ("aten:cpu", "inductor", "torch"),
    "triangular_solve": ("aten:cpu", "torch"),
    "log_softmax_kl": ("inductor", "aten:cpu", "torch"),
    "retention_recurrence": ("inductor", "aten:cpu", "torch"),
    "radix_pass": ("aten:cpu", "torch"),
    "layer_norm_welford": ("aten:cpu", "inductor", "ggml", "torch"),
}

FAMILIES = {
    "embedding_bag": "irregular",
    "index_add": "irregular",
    "triangular_solve": "linalg",
    "log_softmax_kl": "loss",
    "retention_recurrence": "recurrence",
    "radix_pass": "routing",
    "layer_norm_welford": "normalization",
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


def _bf16_tensor(rng, shape, scale=1.0):
    arr = rng.standard_normal(shape, dtype=np.float32) * scale
    if _has_torch():
        import torch

        return torch.from_numpy(arr).to(torch.bfloat16)
    import ml_dtypes

    return np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16))


def _resolve_device(env, impl):
    if impl in {"aten:cpu", "ggml", "intent-v2:mojo", "intent-v2:weft"}:
        if env.device not in {"auto", "cpu"}:
            raise EnvironmentUnavailable(f"{impl} requires a CPU environment")
        return "cpu"
    if impl == "intent-v2:bangc":
        device = "mlu:0" if env.device == "auto" else env.device
        if not device.startswith("mlu:"):
            raise EnvironmentUnavailable(f"{impl} requires an MLU environment")
        return device
    if impl in {"intent-v2:triton", "intent-v2:cutile"}:
        device = "cuda:0" if env.device == "auto" else env.device
        if not device.startswith("cuda:"):
            raise EnvironmentUnavailable(f"{impl} requires a CUDA environment")
        return device
    if env.device == "auto":
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if impl == "inductor" and env.device.startswith("mlu:"):
        raise EnvironmentUnavailable("inductor requires a CPU or CUDA environment")
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
    measured = (
        {"artifact_file": target["artifact_file"]}
        if target.get("artifact_file")
        else {}
    )
    call = kernel.prepare(*args, **measured)
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

    if name == "embedding_bag":
        bags, vocab, dim, max_bag = p["bags"], p["vocab"], p["dim"], p["max_bag"]
        table = _wrap_array(rng.standard_normal((vocab, dim), dtype=np.float32))
        lengths = rng.integers(0, max_bag + 1, size=(bags,), dtype=np.int32)
        offsets = np.empty((bags + 1,), dtype=np.int32)
        offsets[0] = 0
        np.cumsum(lengths, dtype=np.int32, out=offsets[1:])
        total = int(offsets[-1])
        indices = _wrap_array(rng.integers(0, vocab, size=(total,), dtype=np.int32))
        return (table, indices, _wrap_array(offsets))

    if name == "index_add":
        rows, updates, dim = p["rows"], p["updates"], p["dim"]
        base = _wrap_array(rng.standard_normal((rows, dim), dtype=np.float32))
        src = _wrap_array(rng.standard_normal((updates, dim), dtype=np.float32))
        # Mix uniform indices with a hot subset to exercise duplicate-index accumulation.
        hot = updates // 4
        idx_np = np.concatenate(
            (
                rng.integers(0, min(rows, 64), size=(hot,), dtype=np.int32),
                rng.integers(0, rows, size=(updates - hot,), dtype=np.int32),
            )
        )
        rng.shuffle(idx_np)
        return (base, _wrap_array(idx_np), src)

    if name == "triangular_solve":
        b, n, k = p["batch"], p["n"], p["rhs"]
        raw = rng.standard_normal((b, n, n), dtype=np.float32) * (0.4 / math.sqrt(n))
        diag = rng.uniform(1.5, 2.5, size=(b, n)).astype(np.float32)
        l_np = np.tril(raw, -1)
        idx = np.arange(n)
        l_np[:, idx, idx] = diag
        rhs = rng.standard_normal((b, n, k), dtype=np.float32)
        return (_wrap_array(l_np), _wrap_array(rhs))

    if name == "log_softmax_kl":
        m, v = p["rows"], p["vocab"]
        logits = _wrap_array(rng.standard_normal((m, v), dtype=np.float32))
        scores = rng.standard_normal((m, v), dtype=np.float32) * 0.8
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        return (logits, _wrap_array(probs.astype(np.float32)))

    if name == "retention_recurrence":
        b, t, h, d = p["batch"], p["seq"], p["heads"], p["head_dim"]
        gamma = (1.0 - 2.0 ** (-5.0 - np.arange(h, dtype=np.float32))).astype(
            np.float32
        )
        return (
            _bf16_tensor(rng, (b, t, h, d), 0.5),
            _bf16_tensor(rng, (b, t, h, d), 0.5),
            _bf16_tensor(rng, (b, t, h, d), 0.5),
            _wrap_array(gamma),
            1.0 / math.sqrt(d),
        )

    if name == "radix_pass":
        rows, cols, divisor = p["rows"], p["cols"], p["divisor"]
        keys = _wrap_array(
            rng.integers(0, 1 << 24, size=(rows, cols), dtype=np.int32)
        )
        # Distinct values per row verify strict stability of the radix pass.
        vals_np = np.broadcast_to(
            np.arange(cols, dtype=np.int32)[None, :], (rows, cols)
        ).copy()
        return (keys, _wrap_array(vals_np), int(divisor))

    if name == "layer_norm_welford":
        r, c, eps = p["rows"], p["cols"], p["eps"]
        return (
            _wrap_array(rng.standard_normal((r, c), dtype=np.float32)),
            _wrap_array(rng.standard_normal((c,), dtype=np.float32) * 0.5 + 1.0),
            _wrap_array(rng.standard_normal((c,), dtype=np.float32) * 0.1),
            float(eps),
        )

    raise KeyError(name)


def _oracle(name, case, data):
    import torch

    data = _to_torch_data(data)

    if name == "embedding_bag":
        table, indices, offsets = data
        tab = table.double().numpy()
        idx = indices.numpy()
        off = offsets.numpy()
        bags = off.shape[0] - 1
        out = np.zeros((bags, tab.shape[1]), dtype=np.float64)
        for b in range(bags):
            lo, hi = int(off[b]), int(off[b + 1])
            if hi > lo:
                out[b] = tab[idx[lo:hi]].mean(axis=0)
        return torch.from_numpy(out.astype(np.float32))

    if name == "index_add":
        base, indices, src = data
        out = base.double().numpy().copy()
        np.add.at(out, indices.numpy(), src.double().numpy())
        return torch.from_numpy(out.astype(np.float32))

    if name == "triangular_solve":
        l, rhs = data
        l64 = l.double().numpy()
        b64 = rhs.double().numpy()
        x64 = np.zeros_like(b64)
        n = l64.shape[1]
        for r in range(n):
            dot = (
                np.einsum("bc,bck->bk", l64[:, r, :r], x64[:, :r, :])
                if r > 0
                else 0.0
            )
            x64[:, r, :] = (b64[:, r, :] - dot) / l64[:, r, r : r + 1]
        return torch.from_numpy(x64.astype(np.float32))

    if name == "log_softmax_kl":
        logits, target = data
        x = logits.double()
        p = target.double()
        lse = torch.logsumexp(x, dim=-1, keepdim=True)
        lp = x - lse
        kl = torch.where(p > 0.0, p * (torch.log(p) - lp), torch.zeros_like(p)).sum(
            dim=-1
        )
        return lp.float(), kl.float()

    if name == "retention_recurrence":
        q, k, v, gamma, scale = data
        qf, kf, vf = q.double(), k.double(), v.double()
        dec = gamma.double()[None, :, None, None]
        b, t, h, d = qf.shape
        state = torch.zeros((b, h, d, vf.shape[-1]), dtype=torch.float64)
        outs = []
        for step in range(t):
            kt = kf[:, step]
            vt = vf[:, step]
            qt = qf[:, step] * scale
            state = state * dec + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            yt = torch.matmul(qt.unsqueeze(-2), state).squeeze(-2)
            outs.append(yt.to(torch.bfloat16))
        return torch.stack(outs, dim=1), state.float()

    if name == "radix_pass":
        keys, values, divisor = data
        k_np = keys.numpy()
        v_np = values.numpy()
        digits = (k_np // divisor) % 256
        order = np.argsort(digits, axis=-1, kind="stable")
        out_k = np.take_along_axis(k_np, order, axis=-1).astype(np.int32)
        out_v = np.take_along_axis(v_np, order, axis=-1).astype(np.int32)
        offsets = np.zeros((k_np.shape[0], 257), dtype=np.int32)
        for r in range(k_np.shape[0]):
            hist = np.bincount(digits[r], minlength=256).astype(np.int32)
            np.cumsum(hist, dtype=np.int32, out=offsets[r, 1:])
        return (
            torch.from_numpy(out_k),
            torch.from_numpy(out_v),
            torch.from_numpy(offsets),
        )

    if name == "layer_norm_welford":
        x, gain, offset, eps = data
        xd = x.double()
        mean = xd.mean(dim=-1, keepdim=True)
        var = ((xd - mean) ** 2).mean(dim=-1, keepdim=True)
        yn = (xd - mean) * torch.rsqrt(var + eps) * gain.double() + offset.double()
        return yn.float()

    raise KeyError(name)


def _prepare_baseline(name, impl, case, data, env, device):
    if impl == "ggml" and name == "layer_norm_welford":
        from ..implementations.native_libraries import ggml

        x, gain, offset, eps = data
        return ggml(
            "layer_norm",
            None,
            (
                x if isinstance(x, np.ndarray) else x.numpy(),
                gain if isinstance(gain, np.ndarray) else gain.numpy(),
                offset if isinstance(offset, np.ndarray) else offset.numpy(),
                1.0 / x.shape[-1],
                float(eps),
            ),
            env,
        )

    import torch
    import torch.nn.functional as F

    args = _transfer(data, device)
    state: dict[str, object] = {}

    if name == "embedding_bag":
        table, indices, offsets = args

        def eager_eb(tab, idx, off):
            return F.embedding_bag(
                idx, tab, off, mode="mean", include_last_offset=True
            )

        fn = torch.compile(eager_eb) if impl == "inductor" else eager_eb

        def launch():
            state["out"] = fn(table, indices, offsets)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=args,
            metadata={
                "library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen'} embedding_bag ({device})"
            },
        )

    if name == "index_add":
        base, indices, src = args
        indices_i64 = indices.to(torch.int64)

        def eager_ia(b_in, idx_in, s_in):
            return torch.index_add(b_in, 0, idx_in, s_in)

        fn = torch.compile(eager_ia) if impl == "inductor" else eager_ia

        def launch():
            state["out"] = fn(base, indices_i64, src)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=(args, indices_i64),
            metadata={
                "library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen'} index_add ({device})"
            },
        )

    if name == "triangular_solve":
        l, rhs = args

        def launch():
            state["out"] = torch.linalg.solve_triangular(l, rhs, upper=False)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=args,
            metadata={"library": f"PyTorch ATen/LAPACK solve_triangular ({device})"},
        )

    if name == "log_softmax_kl":
        logits, target = args

        def eager_lskl(x, p):
            lp = F.log_softmax(x, dim=-1)
            kl = F.kl_div(lp, p, reduction="none").sum(dim=-1)
            return lp, kl

        fn = torch.compile(eager_lskl) if impl == "inductor" else eager_lskl

        def launch():
            state["out"] = fn(logits, target)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=args,
            metadata={
                "library": f"PyTorch {'Inductor fused' if impl == 'inductor' else 'ATen'} log_softmax+kl_div ({device})"
            },
        )

    if name == "retention_recurrence":
        q, k, v, gamma, scale = args
        t = q.shape[1]
        pos = torch.arange(t, device=device, dtype=torch.float32)
        diff = pos[:, None] - pos[None, :]
        causal = diff >= 0

        def eager_retnet(q_in, k_in, v_in, gamma_in):
            # Decay depends on the input gamma, including on warm invocations.
            log_g = torch.log(gamma_in)[:, None, None]
            decay_mat = torch.where(
                causal[None, :, :],
                torch.exp(log_g * diff[None, :, :].clamp_min(0.0)),
                torch.zeros((gamma_in.shape[0], t, t), device=device, dtype=torch.float32),
            )
            tail_decay = torch.exp(log_g[:, 0, 0][:, None] * (float(t - 1) - pos[None, :]))
            qf = q_in.float().permute(0, 2, 1, 3) * scale
            kf = k_in.float().permute(0, 2, 1, 3)
            vf = v_in.float().permute(0, 2, 1, 3)
            scores = torch.matmul(qf, kf.transpose(-1, -2)) * decay_mat[None, :, :, :]
            out = torch.matmul(scores, vf).permute(0, 2, 1, 3).to(torch.bfloat16)
            k_dec = kf * tail_decay[None, :, :, None]
            final = torch.matmul(k_dec.transpose(-1, -2), vf)
            return out, final

        fn = torch.compile(eager_retnet) if impl == "inductor" else eager_retnet

        def launch():
            state["out"] = fn(q, k, v, gamma)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=(args, pos, diff, causal),
            metadata={
                "library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen BLAS'} RetNet retention ({device})"
            },
        )

    if name == "radix_pass":
        keys, values, divisor = args
        rows = keys.shape[0]
        zeros_col = torch.zeros((rows, 1), device=device, dtype=torch.int32)

        def launch():
            digits = torch.remainder(
                torch.div(keys, divisor, rounding_mode="floor"), 256
            )
            order = torch.argsort(digits, dim=-1, stable=True)
            out_k = torch.gather(keys, 1, order)
            out_v = torch.gather(values, 1, order)
            hist = torch.zeros((rows, 256), device=device, dtype=torch.int32)
            ones = torch.ones_like(keys, dtype=torch.int32)
            hist.scatter_add_(1, digits.to(torch.int64), ones)
            off = torch.cat(
                (zeros_col, torch.cumsum(hist, dim=-1, dtype=torch.int32)), dim=-1
            )
            state["out"] = (out_k, out_v, off)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=(args, zeros_col),
            metadata={"library": f"PyTorch ATen/CUB stable radix pass ({device})"},
        )

    if name == "layer_norm_welford":
        x, gain, offset, eps = args
        norm_shape = (x.shape[-1],)

        def eager_ln(x_in, g_in, b_in):
            return F.layer_norm(x_in, norm_shape, g_in, b_in, eps)

        fn = torch.compile(eager_ln) if impl == "inductor" else eager_ln

        def launch():
            state["out"] = fn(x, gain, offset)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=args,
            metadata={
                "library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen'} LayerNorm ({device})"
            },
        )

    raise KeyError((name, impl))


def build(name):
    cases_map = {
        "embedding_bag": [
            Case(
                "4096x256-v32768-f32",
                dict(bags=4096, vocab=32768, dim=256, max_bag=32),
            )
        ],
        "index_add": [
            Case(
                "4096x256-n16384-f32",
                dict(rows=4096, updates=16384, dim=256),
            )
        ],
        "triangular_solve": [
            Case("32x128x64-f32", dict(batch=32, n=128, rhs=64))
        ],
        "log_softmax_kl": [
            Case("2048x4096-f32", dict(rows=2048, vocab=4096))
        ],
        "retention_recurrence": [
            Case(
                "2x512x8x64-bf16",
                dict(batch=2, seq=512, heads=8, head_dim=64),
            )
        ],
        "radix_pass": [
            Case("128x4096-b8-i32", dict(rows=128, cols=4096, divisor=256))
        ],
        "layer_norm_welford": [
            Case("4096x4096-f32", dict(rows=4096, cols=4096, eps=1e-5))
        ],
    }
    tolerances = {
        "embedding_bag": Tolerance(1e-4, 1e-4),
        "index_add": Tolerance(2e-4, 2e-4),
        "triangular_solve": Tolerance(2e-4, 2e-4),
        "log_softmax_kl": (Tolerance(1e-4, 1e-4), Tolerance(2e-4, 2e-4)),
        "retention_recurrence": (Tolerance(0.03, 0.02), Tolerance(1e-3, 1e-3)),
        "radix_pass": (
            IntegerTolerance(0),
            IntegerTolerance(0),
            IntegerTolerance(0),
        ),
        "layer_norm_welford": Tolerance(1e-4, 1e-4),
    }

    def prepare(impl, case, data, env):
        device = _resolve_device(env, impl)
        if impl.startswith("intent-v2:"):
            backend = impl.partition(":")[2]
            return _prepare_v2(name, backend, _transfer(data, device), env, device)
        return attach_device_timer(_prepare_baseline(name, impl, case, data, env, device))

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
    )
