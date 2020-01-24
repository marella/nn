import inspect
import sys

from tensorflow.keras import optimizers

from .core import to_tf


def wrap(Optimizer):

    class Wrapper(Optimizer):

        def apply_gradients(self, grads_and_vars, *args, **kwargs):
            gv = map(to_tf, grads_and_vars)
            return super(Wrapper, self).apply_gradients(gv, *args, **kwargs)

    return Wrapper


__all__ = []
module = sys.modules[__name__]
for name in dir(optimizers):
    attr = getattr(optimizers, name)
    if not inspect.isclass(attr) or not issubclass(attr, optimizers.Optimizer):
        continue
    attr = wrap(attr)
    setattr(module, name, attr)
    __all__.append(name)
