"""Group 7 DSA-authored workloads, independent mathematical oracles, and mature baselines."""

from __future__ import annotations

from functools import partial
import struct
import numpy as np

from ..implementations.context import Tolerance
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
    "q6k_gemv": ("ggml", "inductor", "aten:cpu", "torch"),
    "mxfp4_gemm": ("inductor", "aten:cpu", "torch"),
    "awq_int4_gemm": ("inductor", "aten:cpu", "torch"),
    "depthwise_conv2d": ("inductor", "aten:cpu", "torch"),
    "gemm_nt": ("inductor", "aten:cpu", "torch"),
}

FAMILIES = {
    "q6k_gemv": "quantized",
    "mxfp4_gemm": "quantized",
    "awq_int4_gemm": "quantized",
    "depthwise_conv2d": "convolution",
    "gemm_nt": "linalg",
}

AWQ_SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)
E2M1_TABLE = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float64,
)


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

    if name == "q6k_gemv":
        n, k = p["n"], p["k"]
        g = k // 256
        records = rng.integers(0, 256, size=(n, g, 210), dtype=np.uint8)
        # Signed int8 sub-block scales in [-16, 16] and IEEE f16 super-block scale d in [0.002, 0.01].
        scales_i8 = rng.integers(-16, 17, size=(n, g, 16), dtype=np.int8)
        records[:, :, 192:208] = scales_i8.view(np.uint8)
        d_f16 = rng.uniform(0.002, 0.01, size=(n, g)).astype("<f2")
        records[:, :, 208:210] = d_f16.view(np.uint8).reshape(n, g, 2)
        x = rng.uniform(-1.0, 1.0, size=(g, 256)).astype(np.float32)
        return (_wrap_array(records), _wrap_array(x))

    if name == "mxfp4_gemm":
        m, n, k = p["m"], p["n"], p["k"]
        g = k // 32
        a = rng.integers(0, 256, size=(m, g, 16), dtype=np.uint8)
        b = rng.integers(0, 256, size=(g, 16, n), dtype=np.uint8)
        # Realistic E8M0 shared exponents in [123, 129] (2^-4 .. 2^2).
        sa = rng.integers(123, 130, size=(m, g), dtype=np.uint8)
        sb = rng.integers(123, 130, size=(g, n), dtype=np.uint8)
        return (
            _wrap_array(a),
            _wrap_array(sa),
            _wrap_array(b),
            _wrap_array(sb),
        )

    if name == "awq_int4_gemm":
        m, n, k, group_size = p["m"], p["n"], p["k"], p["group_size"]
        g = k // group_size
        np_cols = n // 8
        x = (rng.standard_normal((m, k), dtype=np.float32) * 0.5).astype(np.float16)
        w_u4 = rng.integers(0, 16, size=(k, np_cols, 8), dtype=np.uint32)
        z_u4 = rng.integers(6, 10, size=(g, np_cols, 8), dtype=np.uint32)
        qw = np.zeros((k, np_cols), dtype=np.uint32)
        qz = np.zeros((g, np_cols), dtype=np.uint32)
        for lane, sh in enumerate(AWQ_SHIFTS):
            qw |= w_u4[:, :, lane] << np.uint32(sh)
            qz |= z_u4[:, :, lane] << np.uint32(sh)
        scales = rng.uniform(0.005, 0.03, size=(g, n)).astype(np.float16)
        return (
            _wrap_array(x),
            _wrap_array(qw.view(np.int32)),
            _wrap_array(scales),
            _wrap_array(qz.view(np.int32)),
        )

    if name == "depthwise_conv2d":
        b, c, h, w = p["batch"], p["channels"], p["height"], p["width"]
        x = rng.standard_normal((b, c, h, w), dtype=np.float32) * 0.5
        weight = rng.standard_normal((c, 3, 3), dtype=np.float32) * 0.3
        bias = rng.standard_normal((c,), dtype=np.float32) * 0.1
        return (
            _wrap_array(x),
            _wrap_array(weight),
            _wrap_array(bias),
        )

    if name == "gemm_nt":
        m, n, k = p["m"], p["n"], p["k"]
        lhs = (rng.standard_normal((m, k), dtype=np.float32) * 0.25).astype(np.float16)
        rhs = (rng.standard_normal((n, k), dtype=np.float32) * 0.25).astype(np.float16)
        return (_wrap_array(lhs), _wrap_array(rhs))

    raise KeyError(name)


def _dequant_q6k_spec(records_np: np.ndarray) -> np.ndarray:
    """Independent GGUF block_q6_K specification decoder (210B -> 256 float64 values)."""
    n, g, _ = records_np.shape
    ql = records_np[:, :, 0:128]
    qh = records_np[:, :, 128:192]
    scales = records_np[:, :, 192:208].view(np.int8).astype(np.float64)
    d = (
        records_np[:, :, 208:210]
        .copy()
        .view("<f2")
        .reshape(n, g)
        .astype(np.float64)
    )
    out = np.empty((n, g, 256), dtype=np.float64)
    for half in range(2):
        ql_h = ql[:, :, 64 * half : 64 * (half + 1)]
        qh_h = qh[:, :, 32 * half : 32 * (half + 1)]
        sc_h = scales[:, :, 8 * half : 8 * (half + 1)]
        q1 = ((ql_h[:, :, 0:32] & 0xF) | (((qh_h >> 0) & 3) << 4)).astype(np.int32) - 32
        q2 = ((ql_h[:, :, 32:64] & 0xF) | (((qh_h >> 2) & 3) << 4)).astype(np.int32) - 32
        q3 = ((ql_h[:, :, 0:32] >> 4) | (((qh_h >> 4) & 3) << 4)).astype(np.int32) - 32
        q4 = ((ql_h[:, :, 32:64] >> 4) | (((qh_h >> 6) & 3) << 4)).astype(np.int32) - 32
        for isub in range(2):
            sl = slice(16 * isub, 16 * (isub + 1))
            base = 128 * half
            out[:, :, base + 0 + sl.start : base + 0 + sl.stop] = (
                d[:, :, None] * sc_h[:, :, isub + 0, None] * q1[:, :, sl]
            )
            out[:, :, base + 32 + sl.start : base + 32 + sl.stop] = (
                d[:, :, None] * sc_h[:, :, isub + 2, None] * q2[:, :, sl]
            )
            out[:, :, base + 64 + sl.start : base + 64 + sl.stop] = (
                d[:, :, None] * sc_h[:, :, isub + 4, None] * q3[:, :, sl]
            )
            out[:, :, base + 96 + sl.start : base + 96 + sl.stop] = (
                d[:, :, None] * sc_h[:, :, isub + 6, None] * q4[:, :, sl]
            )
    return out


def _e8m0_to_f64(exp_u8: np.ndarray) -> np.ndarray:
    e = exp_u8.astype(np.int32)
    out = np.ldexp(1.0, e - 127)
    return np.where(e == 255, np.nan, out)


def _oracle(name, case, data):
    import torch

    data = _to_torch_data(data)

    if name == "q6k_gemv":
        from ..references.quantization import q8_fields

        records, x = data
        w_f64 = _dequant_q6k_spec(records.numpy())
        scale, codes = q8_fields(x.numpy())
        x_q8_f64 = codes.astype(np.float64) * scale[:, None].astype(np.float64)
        res = np.einsum("ngk,gk->n", w_f64, x_q8_f64, dtype=np.float64)
        return torch.from_numpy(res.astype(np.float32))

    if name == "mxfp4_gemm":
        a, sa, b, sb = (t.numpy() for t in data)
        m, g, _ = a.shape
        n = b.shape[2]
        a_codes = np.empty((m, g, 32), dtype=np.uint8)
        a_codes[:, :, 0::2] = a & 0xF
        a_codes[:, :, 1::2] = (a >> 4) & 0xF
        b_codes = np.empty((g, 32, n), dtype=np.uint8)
        b_codes[:, 0::2, :] = b & 0xF
        b_codes[:, 1::2, :] = (b >> 4) & 0xF
        a_f64 = E2M1_TABLE[a_codes] * _e8m0_to_f64(sa)[:, :, None]
        b_f64 = E2M1_TABLE[b_codes] * _e8m0_to_f64(sb)[:, None, :]
        res = np.einsum(
            "mk,kn->mn",
            a_f64.reshape(m, g * 32),
            b_f64.reshape(g * 32, n),
            dtype=np.float64,
        )
        return torch.from_numpy(res.astype(np.float32))

    if name == "awq_int4_gemm":
        x, qweight, scales, qzeros = (t.numpy() for t in data)
        k, np_cols = qweight.shape
        g, n = scales.shape
        group_size = k // g
        qw_u32 = qweight.view(np.uint32)
        qz_u32 = qzeros.view(np.uint32)
        w_u4 = np.empty((k, np_cols, 8), dtype=np.int32)
        z_u4 = np.empty((g, np_cols, 8), dtype=np.int32)
        for lane, sh in enumerate(AWQ_SHIFTS):
            w_u4[:, :, lane] = ((qw_u32 >> np.uint32(sh)) & np.uint32(0xF)).astype(
                np.int32
            )
            z_u4[:, :, lane] = ((qz_u32 >> np.uint32(sh)) & np.uint32(0xF)).astype(
                np.int32
            )
        w_u4 = w_u4.reshape(g, group_size, n)
        z_u4 = z_u4.reshape(g, 1, n)
        w_f16 = ((w_u4 - z_u4).astype(np.float32) * scales[:, None, :].astype(np.float32)).astype(
            np.float16
        )
        res = x.astype(np.float64) @ w_f16.reshape(k, n).astype(np.float64)
        return torch.from_numpy(res.astype(np.float16))

    if name == "depthwise_conv2d":
        x, weight, bias = (t.numpy().astype(np.float64) for t in data)
        b, c, h, w = x.shape
        pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        acc = np.broadcast_to(bias[None, :, None, None], (b, c, h, w)).copy()
        for kh in range(3):
            for kw in range(3):
                acc += pad[:, :, kh : kh + h, kw : kw + w] * weight[None, :, kh, kw, None, None]
        return torch.from_numpy(acc.astype(np.float32))

    if name == "gemm_nt":
        lhs, rhs = (t.numpy().astype(np.float64) for t in data)
        res = lhs @ rhs.T
        return torch.from_numpy(res.astype(np.float16))

    raise KeyError(name)


def _prepare_baseline(name, impl, case, data, env, device):
    if impl == "ggml" and name == "q6k_gemv":
        from ..implementations.native_libraries import ggml

        records, x = data
        g, blk = x.shape
        records_np = records if isinstance(records, np.ndarray) else records.numpy()
        x_np = x if isinstance(x, np.ndarray) else x.numpy()
        packed = ggml(
            "q6_projection_packed",
            case,
            (records_np, x_np.reshape(1, g * blk)),
            env,
        )
        orig_result = packed.result
        if _has_torch():
            import torch

            packed.result = lambda: torch.from_numpy(orig_result()).reshape(-1)
        else:
            packed.result = lambda: orig_result().reshape(-1)
        return packed

    import torch
    import torch.nn.functional as F

    args = _transfer(data, device)
    state: dict[str, object] = {}

    if name == "q6k_gemv":
        records, x = args
        n, g, _ = records.shape

        def eager_q6k(rec_u8, x_in):
            rec_i32 = rec_u8.to(torch.int32)
            ql = rec_i32[:, :, 0:128].view(n, g, 2, 64)
            qh = rec_i32[:, :, 128:192].view(n, g, 2, 32)
            sc = ((rec_i32[:, :, 192:208] ^ 128) - 128).to(torch.float32).view(n, g, 2, 8)
            d_lo = rec_i32[:, :, 208]
            d_hi = rec_i32[:, :, 209]
            d_u16 = (d_lo | (d_hi << 8)).to(torch.int16).view(torch.float16).to(torch.float32)
            q1 = (((ql[:, :, :, 0:32] & 15) | (((qh >> 0) & 3) << 4)) - 32).to(torch.float32)
            q2 = (((ql[:, :, :, 32:64] & 15) | (((qh >> 2) & 3) << 4)) - 32).to(torch.float32)
            q3 = ((((ql[:, :, :, 0:32] >> 4) & 15) | (((qh >> 4) & 3) << 4)) - 32).to(torch.float32)
            q4 = ((((ql[:, :, :, 32:64] >> 4) & 15) | (((qh >> 6) & 3) << 4)) - 32).to(torch.float32)
            q_all = torch.stack((q1, q2, q3, q4), dim=3).reshape(n, g, 2, 8, 16)
            w = (q_all * sc.unsqueeze(-1)).reshape(n, g, 256) * d_u16.unsqueeze(-1)
            abs_x = x_in.abs()
            peak = abs_x.amax(dim=-1, keepdim=True)
            first_idx = torch.where(
                abs_x == peak,
                torch.arange(256, device=x_in.device).unsqueeze(0),
                torch.full_like(abs_x, 256, dtype=torch.long),
            ).amin(dim=-1, keepdim=True)
            extreme = torch.gather(x_in, -1, first_idx)
            inv = torch.where(peak == 0, torch.zeros_like(extreme), -127.0 / extreme)
            ds = torch.where(peak == 0, torch.zeros_like(extreme), 1.0 / inv)
            x_q8 = torch.clamp(torch.round(x_in * inv), -128.0, 127.0) * ds
            return torch.mv(w.reshape(n, g * 256), x_q8.reshape(g * 256))

        fn = torch.compile(eager_q6k) if impl == "inductor" else eager_q6k

        def launch():
            state["out"] = fn(records, x)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=args,
            metadata={
                "library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen'} Q6_K dequant+mv ({device})"
            },
        )

    if name == "mxfp4_gemm":
        a, sa, b, sb = args
        m, g, _ = a.shape
        n = b.shape[2]
        lut = torch.tensor(E2M1_TABLE.astype(np.float32), device=device, dtype=torch.float32)

        def eager_mxfp4(a_u8, sa_u8, b_u8, sb_u8):
            ai = a_u8.to(torch.int32)
            bi = b_u8.to(torch.int32)
            a_c = torch.stack((ai & 15, (ai >> 4) & 15), dim=-1).reshape(m, g, 32)
            b_c = torch.stack((bi & 15, (bi >> 4) & 15), dim=2).reshape(g, 32, n)
            sa_f = torch.ldexp(
                torch.ones_like(sa_u8, dtype=torch.float32),
                sa_u8.to(torch.int32) - 127,
            )
            sb_f = torch.ldexp(
                torch.ones_like(sb_u8, dtype=torch.float32),
                sb_u8.to(torch.int32) - 127,
            )
            af = (lut[a_c] * sa_f.unsqueeze(-1)).reshape(m, g * 32)
            bf = (lut[b_c] * sb_f.unsqueeze(1)).reshape(g * 32, n)
            return torch.matmul(af, bf)

        fn = torch.compile(eager_mxfp4) if impl == "inductor" else eager_mxfp4

        def launch():
            state["out"] = fn(a, sa, b, sb)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=(args, lut),
            metadata={
                "library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen/BLAS'} MXFP4 dequant+matmul ({device})"
            },
        )

    if name == "awq_int4_gemm":
        x, qweight, scales, qzeros = args
        k, np_cols = qweight.shape
        g, n = scales.shape
        group_size = k // g
        shifts = torch.tensor(AWQ_SHIFTS, device=device, dtype=torch.int32)

        def eager_awq(x_h, qw_i32, sc_h, qz_i32):
            w_u4 = (torch.bitwise_right_shift(qw_i32.unsqueeze(-1), shifts) & 15).reshape(
                g, group_size, n
            )
            z_u4 = (torch.bitwise_right_shift(qz_i32.unsqueeze(-1), shifts) & 15).reshape(
                g, 1, n
            )
            w_h = ((w_u4 - z_u4).to(torch.float16) * sc_h.unsqueeze(1)).reshape(k, n)
            return torch.matmul(x_h, w_h)

        fn = torch.compile(eager_awq) if impl == "inductor" else eager_awq

        def launch():
            state["out"] = fn(x, qweight, scales, qzeros)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=(args, shifts),
            metadata={
                "library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen/BLAS'} AWQ int4 dequant+matmul ({device})"
            },
        )

    if name == "depthwise_conv2d":
        x, weight, bias = args
        c = x.shape[1]
        w4d = weight.unsqueeze(1).contiguous()

        def eager_dw(x_in, w_in, b_in):
            return F.conv2d(x_in, w_in, b_in, stride=1, padding=1, groups=c)

        fn = torch.compile(eager_dw) if impl == "inductor" else eager_dw

        def launch():
            state["out"] = fn(x, w4d, bias)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=(args, w4d),
            metadata={
                "library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen/cuDNN/CNNL'} depthwise conv2d ({device})"
            },
        )

    if name == "gemm_nt":
        lhs, rhs = args

        def eager_linear(a_in, b_in):
            return F.linear(a_in, b_in)

        fn = torch.compile(eager_linear) if impl == "inductor" else eager_linear

        def launch():
            state["out"] = fn(lhs, rhs)

        return PreparedCall(
            launch,
            lambda: state["out"],
            device=device,
            owners=args,
            metadata={
                "library": f"PyTorch {'Inductor' if impl == 'inductor' else 'ATen/cuBLASLt/CNNL'} F.linear ({device})"
            },
        )

    raise KeyError((name, impl))


def build(name):
    cases_map = {
        "q6k_gemv": [Case("4096x4096-q6k", dict(n=4096, k=4096))],
        "mxfp4_gemm": [Case("512x4096x4096-mxfp4", dict(m=512, n=4096, k=4096))],
        "awq_int4_gemm": [
            Case(
                "256x4096x4096-g128-awq",
                dict(m=256, n=4096, k=4096, group_size=128),
            )
        ],
        "depthwise_conv2d": [
            Case(
                "8x128x56x56-k3-f32",
                dict(batch=8, channels=128, height=56, width=56),
            )
        ],
        "gemm_nt": [Case("2048x2048x2048-f16", dict(m=2048, n=2048, k=2048))],
    }
    tolerances = {
        "q6k_gemv": Tolerance(2e-4, 2e-4),
        "mxfp4_gemm": Tolerance(1e-3, 1e-3),
        "awq_int4_gemm": Tolerance(2e-2, 2e-2),
        "depthwise_conv2d": Tolerance(1e-4, 1e-4),
        "gemm_nt": Tolerance(2e-2, 2e-2),
    }

    def prepare(impl, case, data, env):
        device = _resolve_device(env, impl)
        if impl.startswith("intent-v2:"):
            backend = impl.partition(":")[2]
            return _prepare_v2(name, backend, _transfer(data, device), env, device)
        return attach_device_timer(
            _prepare_baseline(name, impl, case, data, env, device)
        )

    impls = {
        impl: Implementation(impl, partial(prepare, impl))
        for impl in (*BASELINES[name], *V2_EXITS)
    }

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
