"""Resolve and execute one workload without CLI-dependent semantics."""
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import traceback
from .model import Result
from .request import RunRequest
from .output import run_output
from .results import save

@dataclass
class RunOutcome:
    results: list
    directory: Path

    @property
    def exit_code(self):
        return int(any((r.status if isinstance(r, Result) else r["status"]) not in
                       {"passed", "unchecked", "generated", "compiled", "executed"} for r in self.results))

def run(request: RunRequest):
    from .execution import dispatch_request
    if request.measurement.fresh_compile_cache:
        from .cold import isolated
        return isolated(request, dispatch_request)
    return dispatch_request(request)

def execute_local(request):
    from .loading import load_workload
    from .engine import benchmark
    with run_output(request.environment, request.workload, request.output) as directory:
        try:
            if request.environment.cpus is not None:
                import os
                selected = set(request.environment.cpus)
                if selected and not selected.issubset(os.sched_getaffinity(0)):
                    try:
                        from weft_core.cpu import bind_cpus
                        bind_cpus(sorted(selected))
                    except Exception:
                        pass
                if not selected or not selected.issubset(os.sched_getaffinity(0)):
                    raise ValueError("requested CPUs must be a nonempty subset of this worker's allowed CPU set")
                os.sched_setaffinity(0, selected)
            if request.kind == "program":
                from .mainline import check_program
                rows = [check_program(request)]
                save(rows, directory, command=request.record())
                return RunOutcome(rows, directory)
            if request.kind != "workload":
                raise ValueError(f"unknown request kind: {request.kind}")
            workload = load_workload(request.workload)
            if request.cases:
                unknown = set(request.cases) - {c.name for c in workload.cases}
                if unknown:
                    raise ValueError(f"unknown cases: {sorted(unknown)}")
                workload = replace(workload, cases=[c for c in workload.cases if c.name in request.cases])
            if request.parameters:
                if workload.parameter_keys is not None and set(request.parameters)-set(workload.parameter_keys):
                    raise ValueError("requested parameters are not exposed by this workload")
                from .model import Case
                keys = list(request.parameters)
                choices = [v if isinstance(v, list) else [v] for v in request.parameters.values()]
                cases = {}
                for original in workload.cases:
                    for values in product(*choices):
                        parameters = {**original.parameters, **dict(zip(keys, values))}
                        name = ",".join(f"{k}={v}" for k, v in parameters.items())
                        cases[name] = Case(name, parameters)
                workload = replace(workload, cases=list(cases.values()))
            # Keep completed participants when a later native compiler or worker
            # reaches its deadline. This is the same current result, not a log.
            partial = {(case.name, name): Result(workload.name, workload.family,
                        case.name, dict(case.parameters), name, stage="waiting",
                        device=request.environment.device, scope=request.measurement.scope,
                        baseline=request.baseline)
                       for case in workload.cases for name in request.implementations}
            def observe(row):
                partial[row.case, row.implementation] = row
                save(list(partial.values()), directory, command=request.record())
            save(list(partial.values()), directory, command=request.record())
            from .cold import compiler_events
            with compiler_events(request) as profile:
                rows = benchmark(workload, request.implementations, environment=request.environment,
                             measurement=request.measurement, baseline=request.baseline, seed=request.seed,
                             observer=observe,
                             oracle_output=directory / "artifacts" / "oracle" if request.oracle_environment is not None else None)
            if profile is not None:
                for row in rows:
                    row.metadata.setdefault("cold_compile",profile)
        except Exception as error:
            status = "environment_missing" if isinstance(error, ImportError) else getattr(error, "status", "preparation_failed")
            rows = [Result(request.workload, "requested", case, {}, name, status=status,
                           stage="load", device=request.environment.device, error=str(error),
                           traceback=traceback.format_exc())
                    for case in request.cases or ("load",) for name in request.implementations]
        save(rows, directory, command=request.record())
        return RunOutcome(rows, directory)
