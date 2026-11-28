"""Suites select registered requests; they do not own an execution path."""
from .registry import workloads
from .output import destination
from .request import RunRequest
from .results import save
from .service import run

SUITES = {"gpu":{"triton","cutile"},"cpu":{"mojo"},"rvv":{"weft"},"mlu":{"bangc"},
          "all":{"triton","cutile","mojo","weft","bangc"}}

def run_suite(args,environment,measurement,implementations,*,oracle_environment=None):
    if args.jobs != 1:
        raise ValueError("suite requests use serial device ownership; --jobs must be 1")
    providers = SUITES[args.suite]
    expected = {"intent:" + p for p in providers} | {"intent-v2:" + p for p in providers}
    selected = [entry.metadata for entry in workloads().values()
                if any(impl in expected for impl in entry.metadata["implementations"])]
    root = destination(environment,"suites/"+args.suite,args.output)
    rows = []
    for index,item in enumerate(selected):
        participants = implementations or tuple(name for name in item["implementations"]
                                                if name in expected)
        outcome = run(RunRequest(item["name"],participants,environment,measurement,args.baseline,
            tuple(args.case or ()),{},args.seed,root/"cases"/item["name"],timeout=args.timeout,
            oracle_environment=oracle_environment))
        rows.extend(outcome.results)
        save(rows,root,command={"suite":args.suite})
        print(f"{index+1}/{len(selected)} {item['name']}",flush=True)
    return int(any((r.status if hasattr(r,"status") else r["status"]) not in {"passed","unchecked"} for r in rows))
