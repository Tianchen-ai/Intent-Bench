"""Typed input/result transport, including views sharing one storage allocation."""
import json
from pathlib import Path
import sys
from dataclasses import fields

import numpy as np


def record_types():
    from .sides.norm_inputs import Arguments
    from .structured import InputBundle, KernelSymbol
    from .implementations.context import Tolerance, IntegerTolerance, SimilarityTolerance
    return {kind.__name__: kind for kind in
            (InputBundle, KernelSymbol, Tolerance, IntegerTolerance, SimilarityTolerance, Arguments)}


def numpy_dtype(name):
    if name in {"bfloat16", "float8_e4m3fn", "float8_e5m2"}:
        import ml_dtypes
        return np.dtype(getattr(ml_dtypes, name))
    return np.dtype(name)


def numpy_storage(value):
    owner = value
    while isinstance(owner.base, np.ndarray):
        owner = owner.base
    if not (owner.flags.c_contiguous or owner.flags.f_contiguous):
        raise ValueError("transport requires an array with a contiguous storage owner")
    if owner.base is not None:
        owner = np.frombuffer(owner.base, dtype=np.uint8)
    raw = owner.ravel(order="K").view(np.uint8)
    return raw, int(value.ctypes.data - raw.ctypes.data)


def write_tree(value, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    allocations, objects = {}, {}

    def storage(key, raw):
        if key not in allocations:
            name = f"storage-{len(allocations)}.npy"
            np.save(directory / name, raw, allow_pickle=False)
            allocations[key] = name
        return allocations[key]

    def encode(item):
        if isinstance(item, np.generic):
            item = item.item()
        if item is None or isinstance(item, (str, bool, int)):
            return {"kind": "scalar", "value": item}
        if isinstance(item, float):
            return {"kind": "float", "value": item.hex()}
        if id(item) in objects:
            return {"kind": "reference", "id": objects[id(item)]}
        identity = len(objects)
        objects[id(item)] = identity
        result = {"id": identity}
        if isinstance(item, np.ndarray):
            raw, offset = numpy_storage(item)
            result.update(kind="numpy", storage=storage(("numpy", raw.ctypes.data, raw.nbytes), raw),
                          dtype=item.dtype.name if item.dtype.name in
                          {"bfloat16", "float8_e4m3fn", "float8_e5m2"} else item.dtype.str,
                          shape=item.shape, strides=item.strides, offset=offset)
        elif (torch := sys.modules.get("torch")) is not None and isinstance(item, torch.Tensor):
            if item.layout != torch.strided or item.is_conj() or item.is_neg():
                raise NotImplementedError("oracle transport requires ordinary strided tensor storage")
            backing = item.untyped_storage()
            raw = torch.empty(0, dtype=torch.uint8, device=item.device).set_(
                backing, 0, (backing.nbytes(),), (1,)).cpu().numpy()
            result.update(kind="torch", storage=storage(
                ("torch", str(item.device), backing.data_ptr(), backing.nbytes()), raw),
                dtype=str(item.dtype).removeprefix("torch."), shape=tuple(item.shape),
                strides=tuple(s * item.element_size() for s in item.stride()),
                offset=item.storage_offset() * item.element_size())
        elif (
            ((module := sys.modules.get("intentdsl.programs.inputs")) is not None and isinstance(item, module.Array))
            or ((module := sys.modules.get("intent.examples.programs.inputs")) is not None and isinstance(item, module.Array))
        ):
            if item.data is None:
                raise ValueError("oracle transport requires actual array contents")
            result.update(kind="intent_array", dtype=item.dtype, data=encode(item.data))
        elif isinstance(item, (list, tuple)):
            result.update(kind="tuple" if isinstance(item, tuple) else "list", items=[encode(v) for v in item])
        elif isinstance(item, dict):
            result.update(kind="dict", items=[[encode(k), encode(v)] for k, v in item.items()])
        elif type(item) in record_types().values():
            result.update(kind="record", name=type(item).__name__,
                          fields={field.name: encode(getattr(item, field.name)) for field in fields(item)})
        else:
            raise TypeError(f"oracle transport does not support {type(item).__name__}")
        return result

    (directory / "tree.json").write_text(json.dumps(encode(value)) + "\n")


def read_tree(directory, *, tensor_device="cpu", host_arrays=False):
    directory = Path(directory)
    allocations, objects, tensors = {}, {}, {}

    def allocation(name):
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("transport storage must be a local filename")
        if name not in allocations:
            raw = np.load(directory / name, allow_pickle=False)
            if raw.ndim != 1 or raw.dtype != np.uint8:
                raise ValueError("transport storage must be raw bytes")
            allocations[name] = raw
        return allocations[name]

    def decode(item):
        kind = item["kind"]
        if kind == "reference":
            return objects[item["id"]]
        if kind == "scalar":
            return item["value"]
        if kind == "float":
            return float.fromhex(item["value"])
        if kind in {"numpy", "torch"}:
            raw = allocation(item["storage"])
            if kind == "numpy" or host_arrays:
                result = np.ndarray(tuple(item["shape"]), dtype=numpy_dtype(item["dtype"]),
                                    buffer=raw, offset=item["offset"], strides=tuple(item["strides"]))
            else:
                import torch
                key = item["storage"]
                dtype = getattr(torch, item["dtype"])
                width = torch.empty((), dtype=dtype).element_size()
                if item["offset"] % width or any(s % width for s in item["strides"]):
                    raise ValueError("unaligned tensor storage geometry")
                if key not in tensors:
                    base = torch.from_numpy(raw).to(tensor_device)
                    tensors[key] = base
                result = tensors[key].view(dtype).as_strided(tuple(item["shape"]),
                    tuple(s // width for s in item["strides"]), item["offset"] // width)
        elif kind == "intent_array":
            try:
                from intentdsl.programs.inputs import Array
            except ImportError:
                from intent.examples.programs.inputs import Array
            data = decode(item["data"])
            result = Array(data, item["dtype"], data.shape)
        elif kind in {"list", "tuple"}:
            result = [decode(v) for v in item["items"]]
            if kind == "tuple":
                result = tuple(result)
        elif kind == "dict":
            result = {decode(k): decode(v) for k, v in item["items"]}
        elif kind == "record":
            result = record_types()[item["name"]](**{name: decode(value) for name, value in item["fields"].items()})
        else:
            raise ValueError(f"unknown transport value kind: {kind}")
        objects[item["id"]] = result
        return result

    return decode(json.loads((directory / "tree.json").read_text()))
