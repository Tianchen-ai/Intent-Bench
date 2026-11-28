"""Independent sparse, grouped and routing implementations for author inputs."""
def prepare(name, implementation, arguments, environment):
    import torch
    from ..corpus import PROJECT_ROOT
    from ..corpus._common.loading import load_module
    from ..corpus._common.measurement import functional_launch
    from ..corpus.cpu.baselines.pytorch import cpu_runtime

    device = str(arguments[0].device)
    owners = arguments
    note = ""
    if implementation == "aten:source" and name == "csr_spmv":
        offsets, columns, values, vector = arguments
        def run():
            rows = torch.arange(offsets.numel() - 1, device=device)
            indices = rows.repeat_interleave(offsets.diff().long(), output_size=values.numel())
            return values.new_zeros(rows.numel()).index_add_(0, indices, values * vector[columns.long()])
        library = "ATen CSR gather and segmented sum"
    elif implementation == "aten:source" and name == "moe_alignment":
        try:
            from intentdsl.programs import BLOCK_SIZE, EXPERTS
        except ImportError:
            from intent.examples.kernels.routing.moe_align import BLOCK_SIZE, EXPERTS
        run = lambda: cpu_runtime.moe_align_block_size(arguments[0], BLOCK_SIZE, EXPERTS)
        library = "ATen stable route ordering and block padding"
    elif implementation == "aten:source" and name == "ragged_gemm":
        x, offsets, weights = arguments
        def run():
            parts = [x[offsets[i]:offsets[i+1]].float() @ w.float() for i, w in enumerate(weights)]
            return torch.cat(parts).to(x.dtype)
        library = "ATen grouped FP32 matmul including dynamic group bounds"
    elif implementation == "aten:source" and name in {"causal_convolution", "nonzero"}:
        fn = cpu_runtime.causal_conv1d if name == "causal_convolution" else cpu_runtime.compact_nonzero
        run = lambda: fn(*arguments)
        library = "ATen grouped convolution and SiLU" if name == "causal_convolution" else "ATen stable nonzero compaction"
    elif implementation == "oracle":
        if name == "csr_spmv":
            offsets, columns, values, vector = arguments
            run = lambda: torch.segment_reduce(values * vector[columns.long()], "sum", offsets=offsets)
        elif name == "ragged_gemm":
            x, offsets, weight = arguments
            boundaries = offsets.cpu().tolist()
            run = lambda: torch.cat([(x[begin:end].float() @ weight[group].float()).to(x.dtype)
                for group, (begin, end) in enumerate(zip(boundaries, boundaries[1:]))])
        elif name == "int8_gemm":
            a, b, bias = arguments
            run = lambda: (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32) + bias.cpu()).to(device)
        elif name == "causal_convolution":
            run = lambda: cpu_runtime.causal_conv1d(*arguments)
        elif name == "moe_alignment":
            try:
                from intentdsl.programs import BLOCK_SIZE, EXPERTS
            except ImportError:
                from intent.examples.kernels.routing.moe_align import BLOCK_SIZE, EXPERTS
            run = lambda: tuple(value.to(device) for value in
                cpu_runtime.moe_align_block_size(arguments[0].cpu(), BLOCK_SIZE, EXPERTS))
        elif name == "nonzero":
            run = lambda: cpu_runtime.compact_nonzero(arguments[0])
        else:
            raise NotImplementedError(name)
        library = "Independent numerical oracle"
    elif name == "nonzero" and implementation == "triton":
        x, = arguments
        rows, columns = x.shape
        if columns % 32 or columns & (columns-1):
            raise NotImplementedError("Triton compaction requires power-of-two rows of at least 32 elements")
        source = load_module(PROJECT_ROOT / "corpus/gpu/baselines/triton/triton-kernels/compaction/masked_compaction.py",
                             "author_masked_compaction")
        indices = torch.arange(columns, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
        shifts = torch.arange(32, dtype=torch.int64, device=device)
        values, compacted = torch.empty_like(x), torch.empty_like(indices)
        def run():
            active = x != 0
            bitmask = torch.sum(active.reshape(rows, -1, 32).to(torch.int64) << shifts, dim=-1).to(torch.int32)
            counts = active.sum(dim=1, dtype=torch.int32)
            source._masked_compaction[(rows,)](x, indices, bitmask, *bitmask.stride(),
                                               values, compacted, -1, K=columns)
            return compacted, counts
        owners = arguments, indices, shifts, values, compacted
        library = "Triton kernels stable masked compaction"
        note = "Mask packing and counts included; library also writes its auxiliary compacted values."
    elif name == "csr_spmv" and implementation == "torch":
        offsets, columns, values, vector = arguments
        matrix = torch.sparse_csr_tensor(offsets, columns, values,
            size=(offsets.numel()-1, vector.numel()), device=device)
        output = torch.empty((matrix.shape[0],), dtype=values.dtype, device=device)
        run = lambda: torch.mv(matrix, vector, out=output)
        owners = arguments, matrix, output
        library = "ATen/cuSPARSE CSR matrix-vector product"
    elif name == "int8_gemm" and implementation == "torch":
        if device.startswith("mlu") and not torch._C._dispatch_has_kernel_for_dispatch_key("aten::_int_mm.out","PrivateUse1"):
            from ..model import EnvironmentUnavailable
            raise EnvironmentUnavailable("torch_mlu _int_mm falls back to CPU on the registered MLU370")
        a, b, bias = arguments
        output = torch.empty((a.shape[0], b.shape[1]), dtype=torch.int32, device=device)
        def run():
            torch._int_mm(a, b, out=output)
            output.add_(bias)
            return output
        owners = arguments, output
        library = "ATen/cuBLAS INT8 GEMM plus INT32 bias"
        note = "Bias addition is included in the complete operator interval."
    elif name == "ragged_gemm" and implementation in {"cutile", "triton"}:
        x, offsets, weights = arguments
        boundaries = offsets.cpu().tolist()
        groups, hidden, columns = weights.shape
        output = torch.empty((x.shape[0], columns), dtype=x.dtype, device=device)
        a = [x[begin:end] for begin, end in zip(boundaries, boundaries[1:])]
        b = list(weights.unbind(0))
        c = [output[begin:end] for begin, end in zip(boundaries, boundaries[1:])]
        owners = arguments, output, a, b, c
        note = "Input/output views bound before timing; all group computation is timed."
        if implementation == "cutile":
            from ..implementations.context import Context
            from ..source_support import tilegym_source
            context = Context(environment.compiler, PROJECT_ROOT, None, "cutile", environment=environment)
            source = tilegym_source(context, "corpus/gpu/baselines/cutile/tilegym/gemm/grouped/group_gemm.py",
                                   "author_grouped_gemm")
            def run():
                source._cutile_autotune_group_gemm(torch.cuda.current_stream(x.device), a, b, c, False, x.device)
                return output
            library = "NVIDIA TileGym grouped GEMM, original unequal groups"
        else:
            import triton
            transposed = torch.empty((groups, columns, hidden), dtype=x.dtype, device=device)
            b = list(transposed.unbind(0))
            pointers = [torch.tensor([value.data_ptr() for value in values], dtype=torch.int64, device=device)
                        for values in (a, b, c)]
            sizes = torch.tensor([size for item in a for size in (item.shape[0], columns, hidden)],
                                 dtype=torch.int32, device=device)
            strides = torch.tensor([hidden, hidden, columns]*groups, dtype=torch.int32, device=device)
            runtime = load_module(PROJECT_ROOT / "corpus/gpu/baselines/triton/triton/gemm/grouped/08-grouped-gemm_runtime.py",
                                  "author_grouped_tma")
            kernel = runtime.load_grouped_gemm(tma=True).__globals__["grouped_matmul_tma_kernel"]
            sms = torch.cuda.get_device_properties(x.device).multi_processor_count
            triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device=device, dtype=torch.int8))
            def run():
                transposed.copy_(weights.transpose(1, 2))
                kernel[(sms,)](*pointers, sizes, strides, groups, FP8=False, NUM_SM=sms)
                return output
            owners = owners, transposed, b, pointers, sizes, strides
            library = "Triton grouped-GEMM TMA tutorial, descriptor boundary checks"
            note += " Required weight transpose is included."
    elif name == "causal_convolution" and implementation == "triton":
        source = load_module(PROJECT_ROOT / "corpus/gpu/baselines/triton/meta-applied-ai/convolution/causal_conv1d/causal_1d_conv.py",
                             "author_causal_conv1d")
        x, weight, bias = arguments
        def run():
            channel_last = x.float().transpose(1, 2).contiguous().transpose(1, 2)
            return source.causal_conv1d_fwd(channel_last, weight.float(), bias=bias.float(),
                                          activation="silu").to(x.dtype).contiguous()
        library = "Meta Applied AI causal_conv1d_fwd with SiLU"
        note = ("Source requires channel-last storage and FP32 operands to preserve the author's FP32 products; "
                "widening, layout conversion and final FP16 materialization are included.")
    elif name == "moe_alignment" and implementation == "cutile":
        try:
            from intentdsl.programs import BLOCK_SIZE, EXPERTS
        except ImportError:
            from intent.examples.kernels.routing.moe_align import BLOCK_SIZE, EXPERTS
        from ..implementations.context import Context
        from ..source_support import runtime_module
        context = Context(environment.compiler, PROJECT_ROOT, None, "cutile", environment=environment)
        runtime = runtime_module(context, "corpus/gpu/baselines/cutile/tilegym/moe/fused/moe_runtime.py",
                                 "author_moe_runtime")
        run = lambda: runtime.align.moe_align_block_size(arguments[0], BLOCK_SIZE, EXPERTS)[:3]
        library = "NVIDIA TileGym MoE count/prefix/scatter/mark"
        note = "All kernels and workspace fills included; route order within an expert is unordered."
    else:
        raise NotImplementedError((name, implementation))
    call = functional_launch(run)
    call.device = device
    call.owners = owners
    call.metadata.update(library=library, note=note)
    return call
