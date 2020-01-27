import inspect
import sys

from tensorflow.keras import layers, models

from .core import to_tf, tensor
from .utils import call_fn


def wrap(Layer):

    class Wrapper(Layer):

        def call(self, x, *args, **kwargs):
            x = to_tf(x)
            x = call_fn(super(Wrapper, self).call, x, *args, **kwargs)
            x = tensor(x)
            return x

    return Wrapper


Sequential = wrap(models.Sequential)

__all__ = ['Sequential']
module = sys.modules[__name__]
for name in dir(layers):
    attr = getattr(layers, name)
    if not inspect.isclass(attr) or not issubclass(attr, layers.Layer):
        continue
    attr = wrap(attr)
    setattr(module, name, attr)
    __all__.append(name)
