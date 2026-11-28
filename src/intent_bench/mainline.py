"""Product checks consume installed author programs through the public local JIT."""
import time
import traceback
from .loading import intent_environment
from .model import Result
from .implementations.intent import backend_options, execution, compile_options

def check_program(request):
    env = request.environment
    intent_environment(env)
    try:
        from intent.examples import PROGRAMS, load_program
        from intent.examples.programs.common import ExampleContext
        from intent.examples.programs import inputs
    except ImportError:
        from intentdsl.programs import PROGRAMS, load_program, inputs
        ExampleContext = None
    if len(request.implementations) != 1 or not request.implementations[0].startswith("intent:"):
        raise ValueError("a program check requires one explicit Intent backend")
    backend = request.implementations[0].partition(":")[2]
    name = request.workload.rsplit("/",1)[-1]
    if name not in PROGRAMS:
        raise ValueError(f"unknown author program: {name}")
    device, workers = execution(env, backend)
    row = Result(request.workload,"mainline","author",{},request.implementations[0],
                 device=device,stage=request.stage)
    start = time.perf_counter()
    context = ExampleContext(backend,device,compiler=env.compiler,prepared=True,stage=request.stage,
        workers=workers,
        backend_options=backend_options(env, backend), options=compile_options(env),
        tuning_config=env.options.get('tuning_config'))
    try:
        inputs.seed([request.seed,list(PROGRAMS).index(name)])
        load_program(name)(context)
        row.status = {"source":"generated","native":"compiled","run":"executed"}[request.stage]
        row.metadata["note"] = "Author entry check; execution alone does not establish numerical correctness."
    except Exception as error:
        row.stage = getattr(error,"stage",request.stage)
        row.status = ("execution_failed" if row.stage in {"run", "provider_invocation", "provider_tuning"}
                      else "compilation_failed")
        row.error = str(error)
        row.traceback = traceback.format_exc()
    finally:
        row.stages_seconds[request.stage] = time.perf_counter()-start
        row.metadata["artifacts"] = [str(a.cache_directory) for a in context.artifacts]
        try:
            context.close()
        except Exception as error:
            cleanup_traceback = traceback.format_exc()
            if row.error:
                row.metadata["cleanup_error"] = str(error)
                row.metadata["cleanup_traceback"] = cleanup_traceback
            else:
                row.status, row.stage = "cleanup_failed", "cleanup"
                row.error, row.traceback = str(error), cleanup_traceback
    return row
