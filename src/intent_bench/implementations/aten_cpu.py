"""Prepared ATen CPU calls through libtorch, timed without Python submission."""
import ctypes as C
from functools import cache
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

from ..model import PreparedCall
from ..output import output_root


OPERATIONS = {name: index for index, name in enumerate((
    "relu", "softmax", "transpose", "embedding", "layer_norm", "gemm", "batched_gemm",
    "swiglu", "cumsum", "adamw", "batch_norm", "cholesky", "group_norm_backward",
    "csr_spmv", "ragged_gemm", "int8_gemm", "attention", "histogram", "jagged_mean",
    "batched_gemm_tn", "batched_gemm_nt", "batched_gemm_tt", "causal_convolution", "nonzero",
    "fp8_gemm", "paged_decode", "mamba_chunk_state", "causal_linear_attention", "aoti", "vllm_moe_alignment",
    "attention_backward", "varlen_causal_conv1d", "varlen_unfold", "layer_norm_backward",
    "qkv_projection", "batched_gemm_subnormal_ftz"))}


@cache
def _supports_march_native(compiler):
    return subprocess.run(
        [compiler, "-march=native", "-E", "-x", "c++", "-", "-o", os.devnull],
        input="", text=True, capture_output=True, timeout=30,
    ).returncode == 0


@cache
def library(root, compiler, onednn=False, dnnl_prefix=None, *, aoti=False, vllm=False):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
    parent = root/"build/native"
    parent.mkdir(parents=True, exist_ok=True)
    variant = "-".join(("aten-cpu", *(tag for tag, enabled in (("onednn", onednn), ("aoti", aoti), ("vllm", vllm)) if enabled)))
    cached = parent/variant
    source = Path(__file__).parents[1]/"native/aten_cpu.cpp"
    sources = [source]
    head = [compiler, "-std=c++20", "-O3", "-shared", "-fPIC",
            "-D_GLIBCXX_USE_CXX11_ABI="+str(int(torch.compiled_with_cxx11_abi())), str(source)]
    head += ["-I"+path for path in include_paths()]
    if aoti:
        head += ["-DINTENT_BENCH_AOTI"]
    if vllm:
        vllm_source = source.with_name("vllm_moe.cpp")
        sources.append(vllm_source)
        head += ["-DINTENT_BENCH_VLLM", "-fopenmp"]
        if _supports_march_native(compiler):
            head += ["-march=native"]
        head += [str(vllm_source)]
        capability = torch.backends.cpu.get_cpu_capability()
        if capability in {"AVX2", "AVX512"}:
            head += ["-DCPU_CAPABILITY_"+capability]
    if onednn:
        sources.append(source.with_name("onednn_matmul.cpp"))
    tail = [option for path in library_paths() for option in ("-L"+path, "-Wl,-rpath,"+path)]
    tail += ["-ltorch_cpu", "-lc10"]
    spec = {"head": head, "tail": tail, "dnnl_prefix": str(dnnl_prefix) if dnnl_prefix else None}
    so_file = cached/"aten.so"
    meta_file = cached/"command.json"
    build = None
    if so_file.is_file() and meta_file.is_file():
        try:
            saved = json.loads(meta_file.read_text())
            so_mtime = so_file.stat().st_mtime_ns
            if saved == spec and all(src.stat().st_mtime_ns <= so_mtime for src in sources):
                build = cached
        except (OSError, ValueError):
            build = None
    if build is None:
        tmp = Path(tempfile.mkdtemp(prefix=variant+"-", dir=parent))
        build_head = list(head)
        build_tail = list(tail)
        if onednn:
            # Compile against the system oneDNN headers separately from libtorch's
            # bundled private headers. The bridge exposes only its own plain C ABI.
            matrix_source = source.with_name("onednn_matmul.cpp")
            matrix_object = tmp/"onednn_matmul.o"
            include = ["-I"+str(Path(dnnl_prefix)/"include")] if dnnl_prefix else []
            result = subprocess.run([compiler, "-std=c++17", "-O3", "-fPIC", *include, "-c", str(matrix_source),
                                     "-o", str(matrix_object)], text=True, capture_output=True, timeout=600)
            (tmp/"onednn-compile.log").write_text(result.stdout+result.stderr)
            if result.returncode:
                from ..corpus._common.measurement import PipelineStageError
                raise PipelineStageError("source_native_compilation",
                    f"oneDNN 3.x development headers are required (environment option dnnl_prefix); "
                    f"see {tmp/'onednn-compile.log'}\n{result.stderr}")
            if dnnl_prefix:
                directory = str(Path(dnnl_prefix)/"lib")
                build_tail = ["-L"+directory, "-Wl,-rpath,"+directory] + build_tail
            build_head += ["-DINTENT_BENCH_ONEDNN", str(matrix_object)]
            build_tail += ["-ldnnl"]
        command = [*build_head, "-o", str(tmp/"aten.so"), *build_tail]
        result = subprocess.run(command, text=True, capture_output=True, timeout=600)
        (tmp/"compile.log").write_text(result.stdout+result.stderr)
        if result.returncode:
            from ..corpus._common.measurement import PipelineStageError
            raise PipelineStageError("source_native_compilation", f"ATen binding build failed: {tmp/'compile.log'}\n{result.stderr}")
        (tmp/"command.json").write_text(json.dumps(spec))
        cached.mkdir(parents=True, exist_ok=True)
        for fname in ("compile.log", "aten.so", "command.json"):
            os.replace(tmp/fname, cached/fname)
        shutil.rmtree(tmp, ignore_errors=True)
        build = cached
    native = C.CDLL(str(build/"aten.so"))
    native.bench_aten_create.argtypes = [C.c_int, C.c_int, C.c_int, C.POINTER(C.c_void_p),
        C.POINTER(C.c_int), C.POINTER(C.c_int64), C.POINTER(C.c_int64), C.POINTER(C.c_int64),
        C.c_int, C.POINTER(C.c_double)]
    native.bench_aten_create.restype = C.c_void_p
    native.bench_aten_run.argtypes = [C.c_void_p]
    native.bench_aten_run.restype = None
    if aoti:
        native.bench_aten_load_package.argtypes = [C.c_void_p, C.c_char_p]
        native.bench_aten_load_package.restype = None
    native.bench_aten_error.argtypes = [C.c_void_p]
    native.bench_aten_error.restype = C.c_char_p
    native.bench_aten_implementation.argtypes = [C.c_void_p]
    native.bench_aten_implementation.restype = C.c_char_p
    native.bench_aten_outputs.argtypes = [C.c_void_p]
    native.bench_aten_outputs.restype = C.c_int
    axes = C.POINTER(C.c_int64)
    native.bench_aten_output.argtypes = [C.c_void_p, C.c_int, axes, C.POINTER(axes),
                                         C.POINTER(axes), C.POINTER(C.c_int)]
    native.bench_aten_output.restype = C.c_void_p
    native.bench_aten_destroy.argtypes = [C.c_void_p]
    native.bench_aten_destroy.restype = None
    return native, build


def prepare(name, arguments, environment, *, package=None):
    import torch
    from .native_sampling import sample_timer
    from .native_sequence import NativeSequence
    if environment.device != "cpu":
        raise ValueError("ATen native CPU comparisons require device=cpu")
    if (name == "aoti") != (package is not None):
        raise ValueError("AOTInductor comparison requires an explicit compiled package")
    tensors = [value for value in arguments if isinstance(value, torch.Tensor)]
    scalars = [value for value in arguments if not isinstance(value, torch.Tensor)]
    dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int8,
              torch.uint8, torch.float8_e4m3fn)
    if any(value.device.type != "cpu" or not value.is_contiguous() for value in tensors):
        raise ValueError("this ATen native binding requires contiguous CPU inputs")
    if name == "adamw" and (scalars[3] != 1-scalars[1] or scalars[4] != 1-scalars[2]):
        raise ValueError("the standard AdamW case requires first-step bias correction")
    native, build = library(output_root(environment.output_root), environment.options.get("cxx", "c++"),
                            name == "mamba_chunk_state", environment.options.get("dnnl_prefix"),
                            aoti=name == "aoti", vllm=name == "vllm_moe_alignment")
    dimensions = [size for tensor in tensors for size in tensor.shape]
    strides = [stride for tensor in tensors for stride in tensor.stride()]
    state = C.c_void_p(native.bench_aten_create(OPERATIONS[name], environment.threads, len(tensors),
        (C.c_void_p*len(tensors))(*(tensor.data_ptr() for tensor in tensors)),
        (C.c_int*len(tensors))(*(dtypes.index(tensor.dtype) for tensor in tensors)),
        (C.c_int64*len(tensors))(*(tensor.ndim for tensor in tensors)),
        (C.c_int64*len(dimensions))(*dimensions), (C.c_int64*len(strides))(*strides),
        len(scalars), (C.c_double*len(scalars))(*scalars)))

    def checked():
        if not state.value:
            raise RuntimeError("ATen native call has been closed")
        error = native.bench_aten_error(state)
        if error:
            raise RuntimeError(error.decode())

    class Call:
        def native_call(self):
            checked()
            return SimpleNamespace(function=native.bench_aten_run, argument_types=(C.c_void_p,),
                                   arguments=(state,), owners=(native, tensors), validate=checked, complete=checked)
        def launch(self):
            checked()
            native.bench_aten_run(state)
            checked()
        def result(self):
            checked()
            outputs = []
            for index in range(native.bench_aten_outputs(state)):
                rank, kind = C.c_int64(), C.c_int()
                shape, strides = C.POINTER(C.c_int64)(), C.POINTER(C.c_int64)()
                pointer = native.bench_aten_output(state, index, C.byref(rank), C.byref(shape),
                                                   C.byref(strides), C.byref(kind))
                if kind.value < 0:
                    raise TypeError("unsupported native ATen result dtype")
                sizes = tuple(shape[i] for i in range(rank.value))
                steps = tuple(strides[i] for i in range(rank.value))
                dtype = dtypes[kind.value]
                if 0 in sizes:
                    outputs.append(torch.empty_strided(sizes, steps, dtype=dtype))
                    continue
                span = 1 + sum((size - 1) * stride for size, stride in zip(sizes, steps))
                width = torch.empty((), dtype=dtype).element_size()
                storage = (C.c_ubyte * (span * width)).from_address(pointer)
                view = torch.frombuffer(storage, dtype=dtype).as_strided(sizes, steps)
                outputs.append(view.clone())
            return outputs[0] if len(outputs) == 1 else tuple(outputs)
        def close(self):
            if state.value:
                native.bench_aten_destroy(state)
                state.value = None

    call = Call()
    try:
        checked()
        if package is not None:
            native.bench_aten_load_package(state, str(package).encode())
            checked()
    except Exception:
        call.close()
        raise
    metadata = {"library": "PyTorch libtorch/ATen CPU", "native_build": str(build),
                "scope_note": "Prepared native ATen operations; output/workspace allocation and Python submission excluded."}
    if name == "ragged_gemm":
        metadata["scope_note"] += " Reading group offsets, slicing views and every group GEMM are inside the native interval."
    if name == "int8_gemm":
        metadata["library"] = "ATen CPU INT8 GEMM with INT32 bias"
        metadata["scope_note"] += " Integer GEMM and bias addition are both timed."
    if name == "attention":
        metadata["library"] = "ATen CPU Flash Attention, causal grouped queries"
        metadata["scope_note"] = ("Public native CPU flash-attention operator on original grouped Q/K/V; "
                                  "packing, output/LSE allocation and auxiliary LSE are included. Python submission is excluded.")
    if name == "paged_decode":
        metadata["library"] = "ATen page gathering and CPU Flash Attention"
        metadata["scope_note"] = ("Original FP16 paged KV and grouped queries; runtime page-table reads, page gathering, "
                                  "layout preparation, CPU flash attention and output writeback are all timed. "
                                  "This is a native library composition, not a dedicated paged-attention kernel. "
                                  "The library's packed KV and auxiliary LSE allocations are included.")
    if name == "mamba_chunk_state":
        metadata["library"] = "ATen chunk scaling and oneDNN BF16 matmul with FP32 output"
        metadata["matmul_implementation"] = native.bench_aten_implementation(state).decode()
        metadata["scope_note"] = ("Complete native library composition: FP32 decay/scale, grouped basis expansion, "
                                  "BF16 weighted-basis rounding, X transpose and BF16 x BF16 -> FP32 oneDNN matmul. "
                                  "All data preparation is timed; primitive construction and reusable allocation are outside.")
    if name == "causal_linear_attention":
        metadata["library"] = "ATen FP32 causal linear attention and final state"
        metadata["scope_note"] = ("Complete native library composition: Q K^T, causal triangular masking, "
                                  "score V and final K^T V. Full score materialization is timed; "
                                  "reusable allocation and Python submission are outside. "
                                  "This is a BLAS composition, not a dedicated linear-attention kernel.")
    if name == "vllm_moe_alignment":
        metadata["library"] = "vLLM/SGLang native CPU MoE block alignment"
        metadata["source"] = "https://github.com/vllm-project/vllm/blob/main/csrc/cpu/sgl-kernels/moe.cpp"
        metadata["scope_note"] = ("Upstream native routing, per-thread counts/prefix, scatter, block IDs and auxiliary offsets; "
                                  "workspace clearing and output padding are timed. Reusable allocation is outside.")
    if name == "histogram":
        metadata["library"] = "ATen CPU bincount"
        metadata["scope_note"] = ("Complete native signed-index conversion, excluded-bin mapping, bincount and INT32 output. "
                                  "The library's count allocation is included; external workspace and Python submission are excluded.")
    if name == "jagged_mean":
        metadata["library"] = "ATen CPU segment_reduce(mean)"
        metadata["scope_note"] += " Segment offsets are read by the native library inside every interval."
    if name == "batched_gemm":
        metadata["library"] = "ATen CPU batched GEMM (FP32-widened strict subnormals)"
        metadata["numerical_contract"] = "strict_ieee_subnormals"
        metadata["scope_note"] = ("BF16 inputs widened to FP32 to preserve subnormal inputs, intermediates and cancellation; "
                                  "widening, FP32 batched GEMM and BF16 output rounding are all timed. "
                                  "ATen's native BF16 AMX flushes subnormals and is compared only under case subnormal_ftz.")
    elif name == "batched_gemm_subnormal_ftz":
        metadata["library"] = "ATen CPU batched GEMM (native BF16 AMX, subnormal FTZ)"
        metadata["numerical_contract"] = "allow_subnormal_ftz"
        metadata["scope_note"] += " Native BF16 batched GEMM (AMX FTZ) timed under explicit allow_subnormal_ftz contract."
    elif name.startswith("batched_gemm"):
        metadata["library"] = "ATen CPU batched GEMM"
        metadata["scope_note"] += " Transposed views and any library packing are inside the native interval; input dtype is unchanged."
    if name == "causal_convolution":
        metadata["library"] = "ATen CPU grouped conv1d and SiLU"
        metadata["scope_note"] = ("FP16 inputs widened to FP32 to preserve source products and accumulation; "
                                  "conversion, grouped convolution, causal slicing, SiLU and FP16 output all timed. "
                                  "The native library's convolution allocation is included.")
    if name == "nonzero":
        metadata["library"] = "ATen CPU nonzero and bincount"
        metadata["scope_note"] = ("Native stable coordinate extraction, row counts, INT32 index conversion and "
                                  "padding to the author's fixed row capacity all timed, including library temporary allocations.")
    if name == "attention_backward":
        metadata["library"] = "ATen CPU flash attention backward"
        metadata["scope_note"] = ("Three gradients from the supplied output and natural-log LSE. "
                                  "KV expansion and FP32 grouped-gradient reduction are timed.")
    if name == "varlen_causal_conv1d":
        metadata["library"] = "ATen CPU grouped conv1d and SiLU"
        metadata["scope_note"] = ("All segments, FP32 convolution, causal slicing, SiLU, BF16 output "
                                  "and the padded final history are inside one native interval.")
    if name == "varlen_unfold":
        metadata["library"] = "ATen CPU pad, unfold and reduction"
        metadata["scope_note"] = ("The retained reference tensor operations, including FP32 products, "
                                  "activation, output conversion, concatenation and stacking of history.")
    if name == "fp8_gemm":
        metadata["library"] = "ATen CPU E4M3/E8M0 decoding and FP32 GEMM"
        metadata["scope_note"] = ("Original E4M3 inputs and E8M0 scale bytes; decoding, scale application and native FP32 GEMM "
                                  "all timed. Reusable output/workspace allocation is outside the interval. "
                                  "This is a library composition, not a fused native FP8 matrix instruction.")
    timers = []
    def close():
        for timer in timers:
            timer.close()
        call.close()
    prepared = PreparedCall(call.launch, call.result, close=close, owners=call,
        device="cpu", timing="cpu_native_monotonic", metadata=metadata)
    def measure(settings):
        if not timers:
            timers.append(NativeSequence((call,)))
        timer = timers[0]
        def interval(repeat=1):
            elapsed = timer.run(repeat)
            # C++ exceptions are retained across the C ABI. Surface them after
            # every interval, including the final sample, outside its timing.
            checked()
            return elapsed
        wrapped = SimpleNamespace(run=interval, supports_repeat=True, _calibrated_repeat=timer._calibrated_repeat)
        try:
            return sample_timer(wrapped, settings, prepared.reset, metadata=metadata)
        finally:
            timer._calibrated_repeat = wrapped._calibrated_repeat
    prepared.native_measure = measure
    return prepared
