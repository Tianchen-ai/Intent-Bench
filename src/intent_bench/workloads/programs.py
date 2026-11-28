"""Compare installed author programs using their own input construction.

The registered programs each submit one complete call. Kernel definitions,
constexprs and multi-kernel composition remain in intent.examples; this module
only binds comparison implementations and observes their declared outputs.
"""
from ..model import Case, EnvironmentUnavailable, Implementation, PreparedCall, Workload
from .author_inputs import Definition as _Definition, inputs as author_inputs, invoke as invoke_author


BASELINES = {
    "relu": {"cutile": "NVIDIA TileGym ReLU", "torch": "ATen ReLU", "ggml": "GGML ReLU with native FP16 input and output conversions", "cnnl": "CNNL ReLU"},
    "softmax": {"triton": "Triton fused-softmax tutorial", "torch": "ATen softmax", "cnnl": "CNNL softmax",
                "ggml": "GGML native softmax with FP16 input and output conversion"},
    "transpose": {"torch": "ATen transpose materialization", "ggml": "GGML materialized transposed view"},
    "embedding": {"triton": "Liger embedding lookup", "torch": "ATen embedding lookup"},
    "layer_norm": {"triton": "Triton layer-norm tutorial", "torch": "ATen LayerNorm",
                   "mojo": "MAX rowwise LayerNorm moments", "ggml": "GGML norm with FP32 gain and bias"},
    "gemm": {"triton": "Triton matrix-multiplication tutorial", "torch": "ATen GEMM", "cnnl": "CNNL matmul"},
    "batched_gemm": {"cutile": "NVIDIA TileGym batched GEMM", "torch": "ATen batched GEMM"},
    "histogram": {"cub": "NVIDIA CUB DeviceHistogram, including initialization and merge"},
    "swiglu": {"triton": "Liger fused SwiGLU", "torch": "ATen SiLU and multiply composition"},
    "rope": {"torch": "ATen split-half RoPE including both InOut updates",
             "triton": "Liger RoPE, including layout conversion and InOut writeback",
             "mojo": "MAX native split-half RoPE for Q/K, including phase interleaving"},
    "jagged_mean": {"triton": "TritonBench jagged mean", "torch": "ATen segment_reduce"},
    "cumsum": {"torch": "ATen cumsum"},
    "batch_norm": {"torch": "ATen native_batch_norm, including running-state updates"},
    "group_norm_backward": {"torch": "ATen native_group_norm_backward"},
    "adamw": {"torch": "ATen fused AdamW"},
    "cholesky": {"torch": "ATen/cuSOLVER Cholesky, InOut"},
    "dropout": {"cutile": "NVIDIA TileGym xorshift dropout",
                "inductor:cpu": "Native AOTInductor fused xorshift dropout with the same RNG and FP16 rounding"},
    "attention": {"cutile": "NVIDIA TileGym causal grouped-query FMHA"},
    "paged_decode": {"triton": "vLLM split-K paged GQA decode, including merge"},
    "mamba_chunk_state": {"triton": "state-spaces/mamba chunk state, FP32 states"},
    "gated_delta": {"cutile": "NVIDIA TileGym recurrent gated delta, including final state",
                    "ggml": "GGML native recurrent gated delta, including BF16 conversions, grouped heads and final state"},
    "csr_spmv": {"torch": "ATen/cuSPARSE CSR matrix-vector product"},
    "ragged_gemm": {"cutile": "NVIDIA TileGym grouped GEMM, unequal groups and prebound output slices",
                    "triton": "Triton grouped TMA GEMM, including required weight transpose"},
    "int8_gemm": {"torch": "ATen/cuBLAS INT8 GEMM with INT32 bias addition"},
    "causal_convolution": {"triton": "Meta Applied AI fused causal depthwise convolution and SiLU"},
    "moe_alignment": {"cutile": "NVIDIA TileGym MoE alignment, including workspace initialization and all stages",
                      "vllm:cpu": "vLLM/SGLang native CPU MoE alignment, including workspace clearing and padding"},
    "nonzero": {"triton": "Triton kernels stable masked compaction, including mask packing and row counts"},
    "causal_linear_attention": {"triton": "TritonBench chunk linear attention, FP32 output and final state"},
    "fp8_gemm": {"cutile": "NVIDIA cuTile block-scaled E4M3 GEMM with E8M0 scales",
                 "torch": "ATen scale decoding and cuBLAS FP32 GEMM; all conversions included"},
    "q4_projection": {"ggml": "GGML Q4_K projection using original packed weights; activation quantization inside call"},
}
from ..implementations.aten_cpu import OPERATIONS as CPU_PROGRAMS
for _name in BASELINES.keys() & CPU_PROGRAMS.keys():
    BASELINES[_name]["aten:cpu"] = "Native libtorch/ATen CPU " + _name
BASELINES["attention"]["aten:cpu"] = "ATen CPU Flash Attention, causal grouped queries and auxiliary LSE"
BASELINES["paged_decode"]["aten:cpu"] = "ATen page gathering and CPU Flash Attention; complete native library composition"
BASELINES["mamba_chunk_state"]["aten:cpu"] = "ATen chunk scaling and oneDNN BF16 matmul with FP32 output; all data preparation included"
BASELINES["causal_linear_attention"]["aten:cpu"] = "ATen FP32 causal score contractions and final state; complete native BLAS composition"
BASELINES["int8_gemm"]["aten:cpu"] = "ATen CPU INT8 GEMM, including INT32 bias addition"
BASELINES["histogram"]["aten:cpu"] = "ATen CPU bincount, including signed conversion, excluded bins and INT32 output"
BASELINES["jagged_mean"]["aten:cpu"] = "ATen CPU segment_reduce(mean), including runtime segment offsets"
BASELINES["causal_convolution"]["aten:cpu"] = "ATen CPU grouped conv1d and SiLU, including FP32 widening and output conversion"
BASELINES["nonzero"]["aten:cpu"] = "ATen CPU nonzero and bincount, including stable row packing and padding"
BASELINES["fp8_gemm"]["aten:cpu"] = "ATen CPU E4M3/E8M0 decoding and FP32 GEMM; all conversions included"


def backends():
    from importlib.util import find_spec
    import json
    spec = find_spec("intent")
    if spec is None:
        return ()
    # Use the package loader so editable installations and wheels expose the
    # same product declaration, without executing intent's package initializer.
    root = spec.loader.get_resource_reader("intent").files()
    return tuple(json.loads(root.joinpath("targets").joinpath("components.json").read_text(encoding="utf-8")))


def _tree(value, leaf):
    if isinstance(value, tuple):
        return tuple(_tree(item, leaf) for item in value)
    if isinstance(value, list):
        return [_tree(item, leaf) for item in value]
    if isinstance(value, dict):
        return {key: _tree(item, leaf) for key, item in value.items()}
    return leaf(value)


def _capture_results(value, getter, bindings):
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _capture_results(item, lambda index=index: getter()[index], bindings)
    elif isinstance(value, dict):
        for key, item in value.items():
            _capture_results(item, lambda key=key: getter()[key], bindings)
    else:
        bindings[id(value)] = getter


def _mutable_arguments(program):
    if not isinstance(program, _Definition):
        return tuple(getattr(program, "mutable_inputs", ()))
    if hasattr(program.definition, "mutable_inputs"):
        return tuple(program.definition.mutable_inputs)
    from intent.language.annotations import ConstexprSpec, ViewKind, ViewSpec
    parameters = [p.annotation for p in program.definition.signature.parameters.values()
                  if not isinstance(p.annotation, ConstexprSpec)
                  and not (isinstance(p.annotation, ViewSpec) and p.annotation.kind is ViewKind.OUT)]
    return tuple(i for i, p in enumerate(parameters)
                 if isinstance(p, ViewSpec) and p.kind is ViewKind.INOUT)


def _device(env, backend=None):
    device = ("cpu" if backend in {"mojo", "weft", "ggml", "aten:cpu", "inductor:cpu", "vllm:cpu"} else
              "mlu:0" if backend in {"bangc", "cnnl"} else "cuda:0") if env.device == "auto" else env.device
    if backend in {"triton", "cutile", "cub"} and not device.startswith("cuda:"):
        raise EnvironmentUnavailable(f"{backend} requires a CUDA execution environment")
    if backend in {"mojo", "weft", "ggml", "aten:cpu"} and device != "cpu":
        raise EnvironmentUnavailable(f"{backend} requires a CPU execution environment")
    if backend in {"bangc", "cnnl"} and not device.startswith("mlu:"):
        raise EnvironmentUnavailable(f"{backend} requires an MLU execution environment")
    return device


def _intent(name, backend, values, env):
    from ..loading import intent_environment
    intent_environment(env)
    from intent.examples.programs.common import ExampleContext
    from ..implementations.intent import backend_options, execution, compile_options

    class PrepareContext(ExampleContext):
        invocation = None

        def call(self, program, *arguments):
            if self.invocation is not None:
                raise NotImplementedError("register complete multi-call author orchestration before benchmarking it")
            self.buffers.begin()
            self.invocation = program.prepare(*arguments)
            return self.invocation.result()

    device, workers = execution(env, backend)
    context = PrepareContext(backend, device, compiler=env.compiler, stage="native",
                             workers=workers,
                             backend_options=backend_options(env, backend), options=compile_options(env),
                             tuning_config=env.options.get('tuning_config'))
    try:
        output = invoke_author(name, values, context)
        if context.invocation is None:
            raise ValueError("author program did not submit its complete call")
        call = context.invocation
        try:
            from intentdsl.programs.inputs import Array
        except ImportError:
            from intent.examples.programs.inputs import Array
        # Author returns can include original InOut arrays alongside Out values.
        # Resolve those arrays to the same resident buffers already bound above.
        output = _tree(output, lambda value: context.buffers.input(value, element=value.dtype)
                       if isinstance(value, Array) else value)
        restore = tuple(context.buffers.resetters.values())
        metadata = {"author_program": name, "input_profile": "installed author program",
                    "compiler_artifacts": [str(a.cache_directory) for a in context.artifacts]}
        reset = lambda: [function() for function in restore]
        def selection():
            from ..implementations.prepared import native_selection
            selections = [native_selection(call, k) for k in context.kernels]
            return selections[0] if len(selections) == 1 else {"members": selections}
        if backend == "bangc":
            from ..implementations.bangc import prepare_sequence
            prepared = prepare_sequence(call, output, environment=env, reset=reset,
                                        close=context.close, owners=context)
            prepared.metadata.update(metadata)
            prepared.tuning_report = selection
            return prepared
        native_measure = None
        if device == "cpu":
            from ..implementations.prepared import intent_sequence
            def result():
                from ..implementations.outputs import native_result
                return native_result(output)
            native = intent_sequence(call, result, reset=reset, metadata=metadata)
            def close():
                try:
                    native.close()
                finally:
                    context.close()
            from dataclasses import replace
            return replace(native, close=close, owners=context, tuning_report=selection,
                compiler_observations=lambda: [a.observation for k in context.kernels for a in k.artifacts
                                                if a.observation is not None])
        return PreparedCall(call.launch, lambda: output, compile=call.compile,
            reset=reset, close=context.close, native_measure=native_measure, tuning_report=selection,
            timing="cpu_native_monotonic" if native_measure else None,
            device=context.device, owners=context,
            metadata=metadata,
            compiler_observations=lambda: [a.observation for k in context.kernels for a in k.artifacts
                                            if a.observation is not None])
    except Exception:
        context.close()
        raise


def _q4_reference(values, env, implementation):
    from ..workloads.weft import q4_projection_reference

    class Context:
        invocation = None
        def compile(self, *args, **kwargs):
            return None
        def call(self, program, weights, activation):
            if implementation == "aten:source":
                import torch
                from ..references.aten_rows import q4_projection
                from ..implementations.prepared import attach_device_timer
                from ..corpus._common.measurement import functional_launch
                device = _device(env, implementation)
                if device.startswith("mlu"):
                    import torch_mlu
                args = tuple(torch.as_tensor(v.data, device=device) for v in (weights, activation))
                call = functional_launch(lambda: q4_projection(*args))
                call.device = device
                call.metadata["library"] = "ATen Q4_K dot including Q8_K activation quantization"
                self.invocation = attach_device_timer(call)
            elif implementation == "ggml":
                from dataclasses import replace
                from ..implementations.native_libraries import ggml
                call = ggml("q4_projection_packed", None, (weights.data, activation.data), env)
                get_output = call.result
                self.invocation = replace(call, result=lambda: get_output().reshape(-1))
            elif implementation == "oracle":
                self.invocation = PreparedCall(lambda: None,
                    lambda: q4_projection_reference(weights.data, activation.data), device="cpu")
            else:
                raise NotImplementedError(("q4_projection", implementation))
            return None

    _device(env, implementation)
    context = Context()
    invoke_author("q4_projection", values, context)
    context.invocation.metadata.update(author_program="q4_projection", input_profile="installed author program")
    return context.invocation


def _reference(name, values, env, implementation="torch", *, allow_subnormal_ftz=False):
    if implementation == "ggml" and name in ("relu", "transpose", "layer_norm", "softmax"):
        from ..implementations.native_libraries import ggml
        args = tuple(arg.to_numpy() if hasattr(arg, "to_numpy") else arg for arg in values["arguments"])
        return ggml(name, None, args, env)
    if name == "q4_projection":
        return _q4_reference(values, env, implementation)
    import torch
    try:
        from intentdsl.programs.inputs import Array
    except ImportError:
        from intent.examples.programs.inputs import Array
    from ..corpus._common.measurement import functional_launch

    device = _device(env, implementation)
    if implementation in {"cnnl", "oracle"} and device.startswith(("mlu:", "npu:")):
        device = "cpu"
    dtypes = {"f16": torch.float16, "f32": torch.float32, "bf16": torch.bfloat16,
              "i8": torch.int8, "u8": torch.uint8, "i32": torch.int32, "i64": torch.int64,
              "f8e4m3fn": torch.float8_e4m3fn}
    buffers, results = {}, {}

    def tensor(value):
        if not isinstance(value, Array):
            return value
        if id(value) in buffers:
            return buffers[id(value)]
        # Preserve BF16 storage bits; NumPy's BF16 is not a torch.from_numpy dtype.
        data = torch.frombuffer(value.data, dtype=dtypes[value.dtype]).reshape(value.shape)
        buffers[id(value)] = data.to(device)
        return buffers[id(value)]

    class ReferenceContext:
        invocation = None

        def compile(self, definition, *, constexprs=None):
            # Author construction still describes its composition; the independent
            # library call does not compile or interpret Intent kernel definitions.
            return _Definition(definition)

        def bind(self, call, arguments, mutable):
            self.invocation = call
            initial = []
            for index in mutable:
                value = arguments[index]
                if value.device.type == "cpu":
                    import numpy as np
                    contents = value.detach().view(torch.uint8).numpy()
                    saved = contents.copy()
                    initial.append(lambda contents=contents, saved=saved: np.copyto(contents, saved))
                else:
                    saved = value.clone()
                    initial.append(lambda value=value, saved=saved: value.copy_(saved))
            if initial:
                call.reset = lambda: [restore() for restore in initial]
            # Some library APIs allocate their output lazily. One unmeasured
            # invocation establishes the author's return-tree projections.
            try:
                call.launch()
            except Exception:
                if call.close:
                    call.close()
                raise
            output = call.result()
            _capture_results(output, call.result, results)
            return output

        def call(self, program, *arguments):
            if self.invocation is not None:
                raise NotImplementedError("register the full author sequence before benchmarking it")
            arguments = tuple(tensor(value) for value in arguments)
            mutable = _mutable_arguments(program)
            if implementation == "aten:cpu":
                from ..implementations.aten_cpu import prepare
                op_name = ("batched_gemm_subnormal_ftz"
                           if name == "batched_gemm" and allow_subnormal_ftz else name)
                return self.bind(prepare(op_name, arguments, env), arguments, mutable)
            if implementation == "inductor:cpu":
                from ..implementations.aoti_cpu import prepare
                return self.bind(prepare(name, arguments, env), arguments, mutable)
            if implementation == "vllm:cpu":
                from ..implementations.aten_cpu import prepare
                try:
                    from intentdsl.programs import EXPERTS, BLOCK_SIZE, PADDED_ROUTES
                except ImportError:
                    from intent.examples.kernels.routing.moe_align import EXPERTS, BLOCK_SIZE, PADDED_ROUTES
                return self.bind(prepare("vllm_moe_alignment",
                    (*arguments, EXPERTS, BLOCK_SIZE, PADDED_ROUTES), env), arguments, mutable)
            if implementation == "ggml":
                from ..implementations.native_libraries import ggml
                if name != "gated_delta":
                    raise NotImplementedError((name, implementation))
                *values, scale = arguments
                if scale != values[0].shape[-1]**-0.5:
                    raise ValueError("GGML gated delta uses inverse-square-root query scaling")
                if any(value.dtype != torch.bfloat16 or not value.is_contiguous() for value in values):
                    raise ValueError("GGML gated delta binding requires contiguous BF16 inputs")
                bits = tuple(value.view(torch.uint16).numpy() for value in values)
                return self.bind(ggml(name, None, bits, env), arguments, mutable)
            if implementation == "cnnl":
                from ..implementations.native_libraries import cnnl
                operation = "matmul" if name == "gemm" else name
                call = cnnl(operation, None, tuple(value.numpy() for value in arguments), env)
                return self.bind(call, arguments, mutable)
            if name == "histogram" and implementation == "aten:source":
                from ..references.aten_rows import histogram
                from ..implementations.prepared import attach_device_timer
                call = functional_launch(lambda: histogram(arguments[0], 256))
                call.device = device
                call.metadata["library"] = "ATen integer histogram with excluded bin"
                return self.bind(attach_device_timer(call), arguments, mutable)
            if name == "histogram":
                from ..implementations.cub import histogram
                return self.bind(histogram(arguments[0], env), arguments, mutable)
            if implementation == "mojo":
                if name == "rope":
                    from ..implementations.max_rope import prepare
                    return self.bind(prepare(arguments, env), arguments, mutable)
                from types import SimpleNamespace
                from ..corpus.cpu.baselines.mojo.modular.normalization.layer_norm.layer_norm_runtime import prepare
                from ..implementations.native_sampling import sample_timer
                native = prepare(SimpleNamespace(workers=env.threads, executable=env.options.get("mojo")),
                                 *arguments)
                metadata = {"library": "MAX rowwise LayerNorm moments",
                            "reference_source": "corpus/cpu/baselines/mojo/modular/normalization/layer_norm/layer_norm.mojo"}
                call = PreparedCall(native.launch, native.result, device=device,
                    timing="cpu_native_monotonic", owners=native, metadata=metadata,
                    native_measure=lambda settings: sample_timer(native, settings, None, metadata=metadata))
                return self.bind(call, arguments, mutable)
            from ..references.author_programs import prepare as prepare_additional
            if name in {"csr_spmv", "ragged_gemm", "int8_gemm", "causal_convolution", "moe_alignment", "nonzero"}:
                from ..references.author_irregular import prepare
                return self.bind(prepare(name, implementation, arguments, env), arguments, mutable)
            if name in {"attention", "paged_decode", "mamba_chunk_state", "gated_delta", "causal_linear_attention", "fp8_gemm"}:
                from ..references.author_streaming import prepare
                return self.bind(prepare(name, implementation, arguments, env), arguments, mutable)
            if name in {"swiglu", "rope", "jagged_mean", "cumsum", "batch_norm",
                        "group_norm_backward", "adamw", "cholesky", "dropout"}:
                return self.bind(prepare_additional(name, implementation, arguments, env), arguments, mutable)
            x = arguments[0]
            source = "PyTorch ATen "+device.split(":")[0].upper()
            if implementation == "triton":
                from ..corpus import PROJECT_ROOT
                from ..corpus._common.loading import load_module
                paths = {
                    "softmax": "triton/normalization/softmax/02-fused-softmax_runtime.py",
                    "layer_norm": "triton/normalization/layer_norm/05-layer-norm_runtime.py",
                    "gemm": "triton/gemm/dense/03-matrix-multiplication_runtime.py",
                    "embedding": "liger-kernel/embedding/lookup/embedding_runtime.py",
                }
                path = PROJECT_ROOT/"corpus/gpu/baselines/triton"/paths[name]
                runtime = load_module(path, "author_reference_"+name)
                if name == "softmax":
                    native = runtime.load_softmax()
                    function = lambda: native(x)
                elif name == "gemm":
                    native = runtime.load_matmul()
                    function = lambda: native(*arguments)
                elif name == "layer_norm":
                    native = runtime.load_layer_norm()
                    function = lambda: native(x, (x.shape[-1],), arguments[1], arguments[2], arguments[4])
                else:
                    function = lambda: runtime.LigerEmbeddingFunction.apply(*arguments)
                source = str(path)
            elif implementation == "cutile":
                from ..corpus import PROJECT_ROOT
                from ..implementations.context import Context
                from ..source_support import tilegym_source
                context = Context(env.compiler, PROJECT_ROOT, None, "cutile", environment=env)
                path = ("activation/relu/relu.py" if name == "relu" else "gemm/batched/bmm.py")
                module = tilegym_source(context, "corpus/gpu/baselines/cutile/tilegym/"+path, "author_"+name)
                function = (lambda: module.relu(x)) if name == "relu" else (
                    lambda: module.bmm(*arguments, transpose_a=False, transpose_b=False,
                                       static_persistent=True, compiler_timeout=60))
                source = "NVIDIA TileGym/"+path
            elif implementation == "oracle" and name in {"layer_norm", "softmax"}:
                import numpy as np
                data = x.cpu().numpy().astype(np.float32)
                if name == "layer_norm":
                    gain, bias, inv, eps = arguments[1:]
                    gain, bias = gain.cpu().numpy(), bias.cpu().numpy()
                    def function():
                        center = data.sum(-1, keepdims=True) * np.float32(inv)
                        variance = (data * data).sum(-1, keepdims=True) * np.float32(inv) - center * center
                        rstd = np.reciprocal(np.sqrt(variance + np.float32(eps)))
                        return torch.from_numpy((data - center) * rstd * gain + bias).to(device)
                else:
                    def function():
                        weights = np.exp(data - data.max(-1, keepdims=True))
                        return torch.from_numpy(weights / weights.sum(-1, keepdims=True)).to(x)
                source = "Independent NumPy FP32 author formula"
            elif implementation == "aten:source" and name == "layer_norm":
                from ..references.aten_rows import raw_moment_norm
                function = lambda: raw_moment_norm(*arguments)
                source = "ATen FP32 raw moments and rsqrt"
            elif implementation == "aten:source" and name == "softmax":
                from ..references.aten_rows import softmax
                function = lambda: softmax(x)
                source = "ATen FP32 exp, sum and div"
            elif name == "relu":
                function = lambda: torch.relu(x)
            elif name == "softmax":
                function = lambda: torch.softmax(x, dim=-1)
            elif name == "transpose":
                function = lambda: x.t().contiguous()
            elif name == "embedding":
                function = lambda: torch.nn.functional.embedding(arguments[1], x)
            elif name == "layer_norm":
                function = lambda: torch.nn.functional.layer_norm(
                    x, (x.shape[-1],), arguments[1], arguments[2], arguments[4])
            elif name == "gemm":
                function = lambda: torch.mm(x, arguments[1])
            elif name == "batched_gemm":
                if device == "cpu" and not allow_subnormal_ftz:
                    function = lambda: torch.bmm(x.float(), arguments[1].float()).to(torch.bfloat16)
                else:
                    function = lambda: torch.bmm(x, arguments[1])
            else:
                raise NotImplementedError(name)
            call = functional_launch(function)
            if device == "cpu" and implementation == "aten:source":
                from ..implementations.prepared import attach_device_timer
                call = attach_device_timer(call)
            call.device = device
            call.owners = arguments
            if device.startswith("mlu:"):
                from ..implementations.mlu_rows import timer
                import torch_mlu
                stream = torch.mlu.current_stream(device)
                if (name == "batched_gemm" and implementation == "torch" and
                        "370" in torch.mlu.get_device_name(x.device.index)):
                    call.metadata["numerical_contract"] = "allow_subnormal_ftz"
                    call.metadata["scope_note"] = (
                        "Native MLU370 BF16 bmm flushes subnormal inputs and products. "
                        "This matches the author-unspecified BF16 dot and explicit FTZ. "
                        "Explicit IEEE or FP32-product is a separate contract.")
                call.native_measure = lambda settings: timer(call.launch, stream, settings)
                call.synchronize = stream.synchronize
                call.timing = "cnrt_notifier"
            call.metadata["library"] = source
            return self.bind(call, arguments, mutable)

    context = ReferenceContext()
    output = invoke_author(name, values, context)
    if context.invocation is None:
        raise ValueError("author program did not submit its complete call")
    if device.startswith("mlu:") and implementation in {"torch", "aten:source"} and not context.invocation.native_measure:
        from ..implementations.mlu_rows import timer
        stream = torch.mlu.current_stream(device)
        call = context.invocation
        call.synchronize = stream.synchronize
        call.native_measure = lambda settings: timer(call.launch, stream, settings, call.reset)
        call.timing = "cnrt_notifier"
    if implementation == "aten:source" and device == "cpu":
        from ..implementations.prepared import attach_device_timer
        attach_device_timer(context.invocation)
    context.invocation.metadata.update(author_program=name, input_profile="installed author program")
    from dataclasses import replace
    def observe(value):
        if isinstance(value, Array):
            return buffers[id(value)]
        return results[id(value)]() if id(value) in results else value
    return replace(context.invocation, result=lambda: _tree(output, observe))


def build(name):
    if name not in BASELINES:
        raise KeyError(name)
    from ..implementations.context import Tolerance
    tolerances = {"softmax": Tolerance(.01, .01), "layer_norm": Tolerance(5e-5, 1e-5),
                  "gemm": Tolerance(.01, .01), "batched_gemm": Tolerance(.05, .02),
                  "swiglu": Tolerance(.02, .01), "rope": (Tolerance(.002, .002),)*2,
                  "jagged_mean": Tolerance(2e-5, 1e-5), "cumsum": Tolerance(.002, 1e-5),
                  "group_norm_backward": Tolerance(.05, .02), "adamw": Tolerance(1e-5, 1e-5),
                  "cholesky": Tolerance(.0002),
                  "attention": Tolerance(.05, .02), "paged_decode": Tolerance(.03),
                  "mamba_chunk_state": Tolerance(.02, .02),
                  "gated_delta": (Tolerance(.1, .05),)*2,
                  "csr_spmv": Tolerance(2e-5), "ragged_gemm": Tolerance(.02, .01),
                  "causal_convolution": Tolerance(.003),
                  "causal_linear_attention": (Tolerance(.0002, 1e-5),)*2,
                  "fp8_gemm": Tolerance(1.0, .02),
                  "q4_projection": Tolerance(1e-4, .002),
                  "batch_norm": (Tolerance(.02, .01), *((Tolerance(.0001, .0001),)*4))}
    tolerance = tolerances.get(name, Tolerance(0))
    implementations = {baseline: Implementation(baseline,
        lambda case, values, env, baseline=baseline: _reference(
            name, values, env, baseline,
            allow_subnormal_ftz=bool(case.parameters.get("allow_subnormal_ftz"))),
        description)
        for baseline, description in BASELINES[name].items()}
    if name in {"layer_norm", "softmax", "adamw", "histogram", "batch_norm",
                "causal_convolution", "nonzero", "mamba_chunk_state", "ragged_gemm", "q4_projection",
                "dropout", "jagged_mean", "moe_alignment", "csr_spmv", "paged_decode", "causal_linear_attention",
                "attention", "gated_delta"}:
        implementations["aten:source"] = Implementation("aten:source", lambda case, values, env: _reference(name, values, env, "aten:source"), "ATen composition at the author rounding points")
    for backend in backends():
        implementations["intent:"+backend] = Implementation("intent:"+backend,
            lambda case, values, env, backend=backend: _intent(name, backend, values, env))
    from ..implementations.v2_program import registrations
    implementations.update(registrations(name))
    # Histogram's independent CPU oracle also checks excluded bins in the author
    # input; its output is integer, so no tolerance is applied to counts.
    def histogram_oracle(case, values):
        import numpy as np
        import torch
        class Context:
            def compile(self, *args, **kwargs): return None
            def call(self, program, samples):
                indices = samples.data.astype(np.int32)
                counts = np.bincount(indices[(indices >= 0) & (indices < 256)], minlength=256).astype(np.int32)
                return torch.from_numpy(counts)
        return invoke_author(name, values, Context())
    oracle = histogram_oracle if name == "histogram" else Implementation(
        "oracle", lambda case, values, env: _reference(
            name, values, env, "oracle",
            allow_subnormal_ftz=bool(case.parameters.get("allow_subnormal_ftz"))))
    def check(actual, expected):
        import numpy as np
        if name == "q4_projection":
            if hasattr(actual, "detach"):
                actual = actual.detach().cpu().numpy()
            np.testing.assert_allclose(actual, expected, atol=tolerance.atol, rtol=tolerance.rtol)
            return
        import torch
        from ..numeric import compare_outputs
        def tensor(value):
            if not isinstance(value, np.ndarray):
                return value
            if value.dtype.name == "bfloat16":
                return torch.from_numpy(value.view(np.uint16)).view(torch.bfloat16)
            return torch.from_numpy(value)
        actual, expected = _tree(actual, tensor), _tree(expected, tensor)
        if name == "nonzero":
            actual = actual["indices"], actual["counts"]
            expected = expected["indices"], expected["counts"]
        if name == "moe_alignment":
            from ..sides.cutile_moe_alignment_source import _canonical_alignment
            if actual["block_size"] != expected["block_size"]:
                raise AssertionError("MoE alignment block size differs")
            def canonical(output):
                return _canonical_alignment(output["route_ids"], output["expert_blocks"],
                                            output["total_padded"])
            actual, expected = canonical(actual), canonical(expected)
        if name == "histogram":
            expected = expected.to(actual.device)
        if any(isinstance(t, torch.Tensor) and t.device.type == "mlu"
                             for t in (actual if isinstance(actual, tuple) else (actual,))):
            from ..oracle import _align_actual
            actual = _align_actual(actual, expected)
        compare_outputs(actual, expected, tolerance)
    cases = ([Case("author"), Case("subnormal_ftz", {"allow_subnormal_ftz": True})]
             if name == "batched_gemm" else [Case("author")])
    return Workload("program/"+name, cases, lambda case, seed: author_inputs(name, seed),
        implementations, oracle=oracle, check=check,
        family="program", parameter_keys=())
