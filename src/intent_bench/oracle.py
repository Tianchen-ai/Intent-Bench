"""An explicit correctness environment; never a cross-device performance baseline."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import traceback

from .model import DeviceBusy, Result


def validate_request(request, directory, rows):
    from .engine import assign_speedups
    from .execution import Workspace
    from .output import output_root

    rows = [Result(**row) if isinstance(row, dict) else row for row in rows]
    pending = [row for row in rows if row.metadata.get("oracle_capture")]
    if not pending:
        return rows
    environment = request.oracle_environment
    workspace = Workspace(environment, directory=output_root(environment.output_root)/"work")
    try:
        workspace.copy([directory/"artifacts/oracle"])
        workload = request.workload
        filename, separator, symbol = workload.partition(":")
        if Path(filename).is_file():
            workspace.copy([Path(filename).resolve()])
            workload = "./" + Path(filename).name + (separator + symbol if separator else "")
        local = replace(environment, transport="local", host=None, python=None,
                        output_root=Path(environment.remote_output_root)
                        if workspace.transport == "ssh" else output_root(environment.output_root))
        oracle_request = replace(request, workload=workload, environment=local,
                                 oracle_environment=None, output=None)
        (workspace.local/"request.json").write_text(json.dumps(oracle_request.record()))
        (workspace.local/"participants.json").write_text(json.dumps([
            {"case": row.case, "implementation": row.implementation,
             "capture": row.metadata["oracle_capture"]} for row in pending]))
        location = workspace.stage()
        code = workspace.run([workspace.python, "-m", "intent_bench.oracle_worker", location],
                             timeout=request.timeout, log=directory/"oracle.log",
                             environment={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        workspace.fetch("validation", directory/"artifacts/oracle-validation")
        if code:
            raise RuntimeError(f"oracle worker exited with code {code}; see {directory/'oracle.log'}")
        validations = json.loads((directory/"artifacts/oracle-validation/results.json").read_text())
        indexed = {(item["case"], item["implementation"]): item for item in validations}
        for row in pending:
            verdict = indexed[row.case, row.implementation]
            row.correctness, row.status = verdict["correctness"], verdict["status"]
            row.error, row.traceback = verdict.get("error"), verdict.get("traceback")
            row.metadata["oracle_environment"] = {"name": environment.name,
                "host": environment.host, "device": verdict["device"], "actual_inputs_transferred": True}
            row.stage = "complete" if row.status == "passed" else "correctness"
    except Exception as error:
        for row in pending:
            row.status = row.correctness = "reference_failed"
            row.stage = "correctness"
            row.error = f"{type(error).__name__}: {error}"
            row.traceback = traceback.format_exc()
    finally:
        workspace.close()
    assign_speedups(rows, request.baseline)
    return rows


def _align_actual(actual, expected):
    """Move result storage to the oracle device, without changing its dtype."""
    import sys
    import numpy as np
    if isinstance(expected, dict):
        return {key: _align_actual(value, expected[key]) for key, value in actual.items()}
    if isinstance(expected, (tuple, list)):
        items = [_align_actual(a, e) for a, e in zip(actual, expected, strict=True)]
        return tuple(items) if isinstance(actual, tuple) else items
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(expected, torch.Tensor):
        if isinstance(actual, np.ndarray):
            from .storage import numpy_storage
            raw, offset = numpy_storage(actual)
            aliases = {"bool": "bool", "float16": "float16", "float32": "float32", "float64": "float64",
                       "bfloat16": "bfloat16", "float8_e4m3fn": "float8_e4m3fn", "float8_e5m2": "float8_e5m2",
                       "int8": "int8", "uint8": "uint8", "int16": "int16", "uint16": "uint16",
                       "int32": "int32", "uint32": "uint32", "int64": "int64", "uint64": "uint64"}
            dtype = getattr(torch, aliases[actual.dtype.name])
            base = torch.from_numpy(raw).view(dtype)
            actual = base.as_strided(actual.shape, tuple(s // actual.itemsize for s in actual.strides),
                                     offset // actual.itemsize)
        return actual.to(expected.device)
    return actual


def validate_captured(request, root, participants):
    from .loading import load_workload
    from .model import Case, Implementation
    from .measurement import device_window, reset, synchronize
    from .storage import read_tree
    from .trees import clone, assert_close
    import sys
    environment = request.environment
    if environment.cpus is not None:
        cpus = set(environment.cpus)
        if cpus and not cpus.issubset(os.sched_getaffinity(0)):
            try:
                from weft_core.cpu import bind_cpus
                bind_cpus(sorted(cpus))
            except Exception:
                pass
        if not cpus or not cpus.issubset(os.sched_getaffinity(0)):
            raise ValueError("oracle CPUs must be a nonempty subset of the worker's allowed CPUs")
        os.sched_setaffinity(0, cpus)
    workload = load_workload(request.workload)
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(environment.threads)
    if (torch := sys.modules.get("torch")) is not None:
        torch.set_num_threads(environment.threads)
    rows, references = [], {}
    for participant in participants:
        captured = participant["capture"]
        folder = root/"oracle"/captured["case"]
        call = None
        row = {"case": participant["case"], "implementation": participant["implementation"],
               "device": environment.device, "status": "reference_failed", "correctness": "reference_failed"}
        try:
            if captured["case"] not in references:
                case = Case(**json.loads((folder/"case.json").read_text()))
                values = read_tree(folder/"inputs")
                if workload.oracle is None:
                    raise ValueError("selected workload has no independent oracle")
                if isinstance(workload.oracle, Implementation):
                    call = workload.oracle.prepare(case, values, environment)
                    if call.compile:
                        call.compile()
                    with device_window(call.device):
                        reset(call)
                        call.launch()
                        synchronize(call)
                        expected = clone(call.result())
                else:
                    expected = workload.oracle(case, values)
                references[captured["case"]] = expected
            actual = read_tree(folder/captured["actual"])
            expected = references[captured["case"]]
            actual = _align_actual(actual, expected)
            try:
                if workload.check:
                    workload.check(actual, expected)
                else:
                    assert_close(actual, expected, atol=workload.atol, rtol=workload.rtol)
                row.update(status="passed", correctness="passed")
            except AssertionError as error:
                row.update(status="numerical_failed", correctness="failed", error=str(error))
        except DeviceBusy as error:
            row.update(status=error.status, correctness="unchecked", error=str(error))
        except Exception as error:
            row.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        finally:
            if call is not None and call.close:
                call.close()
        rows.append(row)
    return rows
