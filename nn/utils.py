import inspect


def call_fn(fn, *args, **keywords):
    sig = inspect.signature(fn)
    kwargs = {}
    for keyword, value in keywords.items():
        if keyword in sig.parameters:
            kwargs[keyword] = value
    return fn(*args, **kwargs)
