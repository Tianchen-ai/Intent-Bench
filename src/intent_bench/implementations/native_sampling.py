"""Sample one or more public product native invocations in a single interval."""

_MIN_INTERVAL_MS = 0.05
_MAX_REPEAT = 65536


def _run_interval(run, repeat, supports_repeat):
    if supports_repeat:
        per_call = run(repeat)
        return per_call, per_call * repeat
    if repeat == 1:
        per_call = run()
        return per_call, per_call
    import time

    begin = time.perf_counter()
    for _ in range(repeat):
        run()
    total = (time.perf_counter() - begin) * 1000.0
    return total / repeat, total


def _save_repeat(owner, repeat):
    if owner is None:
        return
    try:
        owner._calibrated_repeat = repeat
    except AttributeError:
        pass


def _sample(run, settings, restore=None, *, metadata=None, idempotent=None, supports_repeat=False, owner=None):
    import numpy as np

    if idempotent is None:
        idempotent = restore is None
    if restore is None:
        restore = lambda: None
    if metadata is not None:
        metadata["cache_policy"] = "flush_cpu_64m" if settings.cache == "flush" else "warm"
    cache = np.empty(64 * 1024 * 1024, dtype=np.uint8) if settings.cache == "flush" else None
    repeat = getattr(owner, "_calibrated_repeat", None) if (idempotent and owner is not None) else (None if idempotent else 1)

    for _ in range(settings.warmup):
        restore()
        _run_interval(run, repeat or 1, supports_repeat)

    values = []
    for _ in range(settings.samples):
        if repeat is None:
            probe_repeat = 1
            while True:
                restore()
                if cache is not None:
                    cache.fill(0)
                per_call, total = _run_interval(run, probe_repeat, supports_repeat)
                if total >= _MIN_INTERVAL_MS or probe_repeat >= _MAX_REPEAT:
                    repeat = probe_repeat
                    _save_repeat(owner, repeat)
                    break
                if total <= 0:
                    probe_repeat = min(_MAX_REPEAT, probe_repeat * 2)
                else:
                    probe_repeat = min(
                        _MAX_REPEAT,
                        max(probe_repeat * 2, int(_MIN_INTERVAL_MS / per_call) + 1),
                    )
        else:
            restore()
            if cache is not None:
                cache.fill(0)
            per_call, _ = _run_interval(run, repeat, supports_repeat)
        values.append(per_call)

    if metadata is not None and repeat and repeat > 1:
        metadata["timer_repeat"] = repeat
    return values


def sample_native_call(run, settings, restore=None, *, metadata=None):
    return _sample(run, settings, restore, metadata=metadata)


def sample_timer(timer, settings, restore=None, *, metadata=None, idempotent=None):
    return _sample(
        timer.run,
        settings,
        restore,
        metadata=metadata,
        idempotent=idempotent,
        supports_repeat=bool(getattr(timer, "supports_repeat", False)),
        owner=timer,
    )
