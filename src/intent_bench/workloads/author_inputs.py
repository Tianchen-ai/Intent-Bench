"""Capture author arguments once; every implementation consumes those bytes."""
from dataclasses import dataclass


@dataclass
class Definition:
    definition: object


def _seed(name, seed):
    try:
        from intentdsl.programs import PROGRAMS, inputs
    except ImportError:
        from intent.examples import PROGRAMS
        from intent.examples.programs import inputs
    inputs.seed([seed, list(PROGRAMS).index(name)])


def _load_program(name):
    try:
        from intentdsl.programs import load_program
    except ImportError:
        from intent.examples import load_program
    return load_program(name)


def inputs(name, seed):
    class Captured(Exception):
        def __init__(self, arguments):
            self.arguments = arguments

    class Context:
        def compile(self, definition, **_):
            return Definition(definition)

        def call(self, program, *arguments):
            # Stop before execution and before the author projects its outputs.
            raise Captured(arguments)

    _seed(name, seed)
    try:
        _load_program(name)(Context())
    except Captured as captured:
        return {"seed": seed, "arguments": captured.arguments}
    raise ValueError("author program did not declare a complete call")


def invoke(name, values, context):
    """Replay author construction, binding captured inputs at its public call."""
    try:
        from intentdsl.programs.inputs import Array
    except ImportError:
        from intent.examples.programs.inputs import Array
    replacements = {}

    class Context:
        called = False

        def compile(self, *args, **kwargs):
            return context.compile(*args, **kwargs)

        def call(self, program, *declared):
            if self.called:
                raise NotImplementedError("register the complete author sequence before benchmarking it")
            self.called = True
            arguments = values["arguments"]
            if len(arguments) != len(declared):
                raise ValueError("captured arguments disagree with the current author program")
            for previous, actual in zip(declared, arguments, strict=True):
                if isinstance(previous, Array):
                    if not isinstance(actual, Array) or previous.dtype != actual.dtype or previous.shape != actual.shape:
                        raise ValueError("captured author argument shape or dtype changed")
                    replacements[id(previous)] = actual
            return context.call(program, *arguments)

    def returned(value):
        if isinstance(value, tuple):
            return tuple(returned(item) for item in value)
        if isinstance(value, list):
            return [returned(item) for item in value]
        if isinstance(value, dict):
            return {key: returned(item) for key, item in value.items()}
        return replacements.get(id(value), value)

    _seed(name, values["seed"])
    replay = Context()
    result = _load_program(name)(replay)
    if not replay.called:
        raise ValueError("author program did not declare a complete call")
    return returned(result)
