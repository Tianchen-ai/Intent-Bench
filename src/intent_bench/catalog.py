"""Discovery uses metadata only, never imports an implementation's SDK."""
import json
from pathlib import Path

ADVANCED_OPERATIONS=('scaled_fp8_gemm','fused_linear_cross_entropy','flash_attention_bias','flash_varlen_attention',
    'chunk_retention','flash_attention_inline','flash_attention_select','flash_attention_full_causal_stream',
    'mamba3_siso_step','mamba3_siso_forward','paged_gqa_f16_reduction')
# Corpus names retain each author's program and inputs. Names grouped below
# have the same argument order and observable outputs at the ATen binding.
ATEN_NATIVE_OPERATIONS = {
    workload: operation
    for operation, workloads in {
        'relu': ('relu',),
        'softmax': ('fused_softmax', 'chunked_softmax', 'streamed_online_softmax',
                    'streamed_online_softmax_f16', 'stable_softmax_online',
                    'streamed_online_softmax_inline'),
        'transpose': ('flaggems_transpose_copy', 'matrix_transpose_scalar_domains',
                      'matrix_transpose_product_domains'),
        'embedding': ('embedding_lookup', 'embedding_lookup_f32', 'index_select'),
        'cumsum': ('flaggems_cumsum',),
        'gemm': ('dense_gemm', 'tilegym_dense_gemm', 'gemm_loop_interchange'),
        'batched_gemm': ('batched_gemm',),
        'qkv_projection': ('qkv_projection',),
        'batched_gemm_tn': ('batched_gemm_tn',),
        'batched_gemm_nt': ('batched_gemm_nt',),
        'batched_gemm_tt': ('batched_gemm_tt',),
        'causal_linear_attention': ('causal_linear_attention_f32',),
    }.items()
    for workload in workloads
}


def catalog():
    from .implementations.v2_program import CATALOG, registrations
    from .side_workload import mappings
    sides = mappings()
    path=Path(__file__).with_name("catalog.json")
    rows=json.loads(path.read_text()) if path.exists() else []
    for row in rows:
        provider=row['provider']
        if provider=='mojo':
            source='mojo' if '/baselines/mojo/' in (row.get('source_runtime') or '') else 'torch'
            row['implementations']=[source,'intent:mojo']
            if row['kernel'] in ADVANCED_OPERATIONS and 'mojo' not in row['implementations']:row['implementations'].append('mojo')
        elif provider=='weft':
            row['implementations']=['weft','intent:weft']
            if row["kernel"]=="q4_k_projection":
                row["implementations"].extend("intent-v2:"+b for b in ("triton","cutile","mojo","weft","bangc"))
            else:
                row["implementations"].append("intent-v2:weft")
            row['source_runtime']='implementations/triton_weft_source.py'
        elif provider=='bangc':
            source='cnnl' if row['kernel'] in {'relu','fused_softmax','dense_gemm'} else 'reference:mojo' if row['kernel'] in {'causal_attention_f32','causal_linear_attention_f32'} else 'reference:cuda'
            row['implementations']=[source,'intent:bangc']
        if 'source_implementation' in row:
            targets=('triton','cutile') if provider in ('triton','cutile') else (provider,)
            row['implementations']=[row['source_implementation'],*('intent:'+target for target in targets)]
        if provider=='mojo' and row['kernel'] in ATEN_NATIVE_OPERATIONS:
            row['implementations'].append('aten:cpu')
        if provider == 'mojo' and row['kernel'] in CATALOG:
            row['implementations'].extend(registrations(row['kernel']))
        row['implementations'].extend(sides.get(row['name'], {}).get('alternatives', {}))
        if sides.get(row["name"],{}).get("v2_symbol"):
            row["implementations"].extend("intent-v2:"+b for b in ("triton","cutile","mojo","weft","bangc"))
    return rows


def get_workload(name):
    from .corpus_adapter import build
    return build(name)
