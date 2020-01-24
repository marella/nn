import tensorflow as tf

from . import ops


class Tensor():

    def __init__(self, value, *args, **kwargs):
        if isinstance(value, Tensor):
            value = value.tf()
        elif not isinstance(value, (tf.Tensor, tf.Variable)):
            value = tf.convert_to_tensor(value, *args, **kwargs)
        self._value = value

    def __getattr__(self, name):
        value = self.tf()
        if hasattr(value, name):
            return getattr(value, name)
        raise AttributeError(f"Attribute {name} doesn't exist.")

    def tf(self):
        return self._value

    def __repr__(self):
        return 'nn.Tensor' + repr(self.tf())


def tensor(x, *args, **kwargs):
    if isinstance(x, Tensor):
        return x
    return Tensor(x, *args, **kwargs)


def tensors(x):
    if isinstance(x, list):
        return list(map(tensor, x))
    elif isinstance(x, tuple):
        return tuple(map(tensor, x))
    else:
        return tensor(x)


def variable(x, *args, **kwargs):
    if isinstance(x, Tensor):
        value = x.tf()
        if isinstance(value, tf.Variable):
            return x
        x = value
    if not isinstance(x, tf.Variable):
        x = tf.Variable(x, *args, **kwargs)
    return Tensor(x)


def to_tf(x):
    if isinstance(x, list):
        return list(map(to_tf, x))
    elif isinstance(x, tuple):
        return tuple(map(to_tf, x))
    elif isinstance(x, Tensor):
        return x.tf()
    else:
        return x


def op(fn, binary=False):

    def wrapper(x, *args, **kwargs):
        x = to_tf(x)
        x = fn(x, *args, **kwargs)
        return tensor(x)

    if binary:
        wrapper = handle_binary(wrapper)
    return wrapper


def handle_binary(fn):

    def wrapper(x, y, *args, **kwargs):
        y = to_tf(y)
        return fn(x, y, *args, **kwargs)

    return wrapper


def add_method(name):
    if hasattr(Tensor, name):
        return
    attr = getattr(ops, name)
    if not callable(attr):
        raise TypeError(f'Operation {name} is not callable.')

    def method(self, *args, **kwargs):
        x = self.tf()
        z = attr(x, *args, **kwargs)
        return Tensor(z)

    if name in ops.binary_ops:
        method = handle_binary(method)
    setattr(Tensor, name, method)


def extend_method(name):
    if hasattr(Tensor, name):
        return

    def method(self, *args, **kwargs):
        x = self.tf()
        attr = getattr(x, name)
        z = attr(*args, **kwargs)
        return Tensor(z)

    if name in ops.binary_ops:
        method = handle_binary(method)
    setattr(Tensor, name, method)


for name in ops.ops:
    add_method(name)

for name in ops.magic_ops:
    extend_method(name)


def conversion_func(*args, **kwargs):
    raise TypeError()  # prevent tensorflow from trying to auto convert


tf.register_tensor_conversion_function(Tensor, conversion_func)


class GradientTape(tf.GradientTape):

    def gradient(self, target, sources, *args, **kwargs):
        target = to_tf(target)
        sources = to_tf(sources)
        x = super(GradientTape, self).gradient(target, sources, *args,
                                               **kwargs)
        return tensors(x)


function = tf.function
Module = tf.Module
