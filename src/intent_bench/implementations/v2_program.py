"""Bind the installed V2 kernel to the existing author's exact input buffers."""
from functools import partial

from ..model import Implementation, PreparedCall


CATALOG = {"rotary_embedding_equivalent_index","qkv_projection","selective_state_scan"}
CONTRACTIONS = {"gemm","batched_gemm","ragged_gemm","int8_gemm","fp8_gemm","q4_projection","mamba_chunk_state"}
RECURRENCES = {"cholesky", "causal_convolution", "gated_delta", "causal_linear_attention"}
MIXED = {"attention", "paged_decode"}


def registrations(name):
    if name not in CONTRACTIONS and name not in CATALOG and name not in RECURRENCES and name not in MIXED and name not in {"relu", "transpose", "swiglu", "dropout", "adamw", "rope", "softmax", "layer_norm", "batch_norm","group_norm_backward","cumsum","embedding","csr_spmv","jagged_mean","histogram","nonzero","moe_alignment"}:
        return {}
    return {
        "intent-v2:" + backend: Implementation(
            "intent-v2:" + backend, partial(prepare, name, backend)
        )
        for backend in ("triton", "cutile", "mojo", "weft", "bangc")
    }


def prepare(name, backend, case, values, env):
    from ..workloads.programs import _device

    device = _device(env, backend)
    if name in CATALOG:
        args = values.arguments
        if device != "cpu":
            if device.startswith("mlu:"):
                import torch_mlu
            args = tuple(v.to(device) for v in args)
        call = bind(name, backend, args, env, device)
        if device.startswith("cuda:"):
            result = call.result
            call.result = lambda: result().cpu()
        return call
    args=author_arguments(values,device)
    if name=="moe_alignment":
        try:
            from intentdsl.programs import EXPERT_BLOCKS, BLOCK_SIZE
        except ImportError:
            from intent.examples.kernels.routing.moe_align import EXPERT_BLOCKS, BLOCK_SIZE
        # This public entry accepts the expert-block destination. Seed its
        # inactive tail before measurement, just like a caller reusing a buffer.
        if device=="cpu":
            import numpy as np
            blocks=np.full(EXPERT_BLOCKS,-1,dtype=np.int32)
        else:
            import torch
            blocks=torch.full((EXPERT_BLOCKS,),-1,device=device,dtype=torch.int32)
        call=bind("moe_alignment_into",backend,(*args,blocks),env,device)
        result=call.result
        def output():
            routes,experts,total=result()
            return {"route_ids":routes,"expert_blocks":experts,"total_padded":total,"block_size":BLOCK_SIZE}
        call.result=output
        call.metadata["caller_output"]="expert_blocks"
        return call
    symbol = ("batched_gemm_subnormal_ftz"
              if name == "batched_gemm" and case.parameters.get("allow_subnormal_ftz")
              else name)
    call=bind(symbol, backend, args, env, device)
    if name=="nonzero":
        result=call.result
        call.result=lambda:dict(zip(("indices","counts"),result(),strict=True))
    return call


def author_arguments(values, device):
    try:
        from intentdsl.programs.inputs import Array
    except ImportError:
        from intent.examples.programs.inputs import Array
    if device == "cpu":
        args = [x.to_numpy() if isinstance(x, Array) else x for x in values["arguments"]]
    else:
        import torch
        if device.startswith("mlu:"):
            import torch_mlu
        dtypes = {"f16": torch.float16, "bf16": torch.bfloat16,
                  "f32": torch.float32, "i32": torch.int32,"i64":torch.int64,
                  "i8":torch.int8,"u8":torch.uint8}
        if any(isinstance(x,Array) and x.dtype=="f8e4m3fn" for x in values["arguments"]):
            dtypes["f8e4m3fn"]=torch.float8_e4m3fn
        def transfer(x):
            if not isinstance(x, Array):
                return x
            if device.startswith("mlu") and x.dtype=="f8e4m3fn":
                # CNRT accepts the storage; this avoids requesting a CNNL FP8
                # conversion while preserving the logical dtype for the JIT.
                raw=torch.frombuffer(x.data,dtype=torch.uint8).reshape(x.shape).to(device)
                return raw.view(dtypes[x.dtype])
            return torch.frombuffer(x.data,dtype=dtypes[x.dtype]).reshape(x.shape).to(device)
        args = [transfer(x) for x in values["arguments"]]
    return args


def with_reciprocal_permission(kernel, division_count):
    """Change only the registered division sites in a benchmark author variant."""
    import ast
    from types import SimpleNamespace

    class ReciprocalPermission(ast.NodeTransformer):
        def visit_BinOp(self, op):
            self.generic_visit(op)
            if isinstance(op.op, ast.Div):
                return ast.copy_location(ast.Call(
                    func=ast.Attribute(value=ast.Name(id="I", ctx=ast.Load()), attr="fdiv", ctx=ast.Load()),
                    args=[op.left, op.right],
                    keywords=[ast.keyword(arg="approximate", value=ast.Constant(True))]), op)
            return op

    module = ast.parse(kernel.source)
    if sum(isinstance(op, ast.Div) for op in ast.walk(module)) != division_count:
        raise ValueError("recheck the registered reference quotient sites")
    return SimpleNamespace(source=ast.unparse(ast.fix_missing_locations(ReciprocalPermission().visit(module))).encode())


def bind(name, backend, args, env, device):
    import intentdsl
    from intentdsl import examples
    options = env.options.get("compilerinfra", {})
    math_kwargs = {k: options[k] for k in ("fpmath_mode", "fast_math", "denormal_fp_math", "numerics") if k in options}
    definition = getattr(examples, name)
    if options.get("reference_numerics") is not None:
        contracts = {("adamw", "bangc", "torch_mlu_fused_adamw"): 3,
                     ("softmax", "mojo", "aten_cpu_softmax"): 1,
                     ("softmax", "weft", "aten_cpu_softmax"): 1}
        count = contracts.get((name, backend, options["reference_numerics"]))
        if count is None or math_kwargs or (options.get("configuration") or {}).get("numerical"):
            raise ValueError("reference arithmetic requires its registered workload, backend and default environment")
        definition = with_reciprocal_permission(definition, count)
    kernel = intentdsl.jit(definition, backend=backend,
                           library=options.get("library"), configuration=options.get("configuration"),
                           **math_kwargs)
    mutable = {"adamw": (1,2,3), "rope":(0,1), "batch_norm":(3,4),"cholesky":(0,)}.get(name, ())
    original = [(args[i], args[i].clone() if hasattr(args[i],"clone") else args[i].copy()) for i in mutable]
    def reset():
        for dest, value in original:
            if device=="cpu":
                dest[...] = value
            else:
                dest.copy_(value)
    measured = {"artifact_file": options["artifact_file"]} if options.get("artifact_file") else {}
    call = kernel.prepare(*args, **measured)
    if "row_id" in options:
        call = call.bind_row(options["row_id"])
    metadata = {"author_program": name}
    def compile():
        call.compile()
        metadata["cold_compile"] = call.compilation_report()

    # Bench already owns the terminal timing event or native host interval.
    execute = getattr(call, "execute", call.launch)
    prepared = PreparedCall(execute, call.result, device=device, compile=compile,
                            tuning_report=call.tuning_report, reset=reset if original else None, metadata=metadata)
    if device == "cpu":
        from .native_sequence import NativeSequence
        from .native_sampling import sample_timer

        timer = None
        def measure(settings):
            nonlocal timer
            if timer is None:
                timer = NativeSequence((call,))
            return sample_timer(timer, settings, reset if original else None, metadata=metadata)
        prepared.native_measure = measure
        prepared.close = lambda: timer and timer.close()
        prepared.timing = "cpu_native_monotonic"
    elif device.startswith("mlu:"):
        from .mlu_rows import timer
        import torch
        stream = torch.mlu.current_stream(device)
        prepared.synchronize = stream.synchronize
        prepared.native_measure = lambda settings: timer(execute, stream, settings, reset)
        prepared.timing = "cnrt_notifier"
        from ..workloads.programs import _tree
        prepared.result = lambda: _tree(call.result(), lambda v:v.cpu())
    return prepared
