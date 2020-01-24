A neural network library built on top of TensorFlow for quickly building deep learning models.

[![Build Status](https://travis-ci.org/marella/nn.svg?branch=master)](https://travis-ci.org/marella/nn)

## Usage

`nn.Tensor` is the core data structure which is a wrapper for `tf.Tensor` and provides additional functionality. It can be created using the `nn.tensor()` function:

```py
import nn

a = nn.tensor([1, 2, 3])
assert isinstance(a, nn.Tensor)
assert a.shape == (3, )
```

It supports method chaining:

```py
c = a.square().sum()
assert c.numpy() == 14
```

and can be used with `tf.Tensor` objects:

```py
import tensorflow as tf

b = tf.constant(2)
c = (a - b).square().sum()
assert c.numpy() == 2
```

It can also be used with high level APIs such as `tf.keras`:

```py
model = nn.Sequential([
  nn.Dense(128, activation='relu'),
  nn.Dropout(0.2),
  nn.Dense(10)
])

y = model(x)
assert isinstance(y, nn.Tensor)
```

and to perform automatic differentiation and optimization:

```py
optimizer = nn.Adam()
with nn.GradientTape() as tape:
    outputs = model(inputs)
    loss = (targets - outputs).square().mean()
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

To use it with ops that expect `tf.Tensor` objects as inputs, wrap the ops using `nn.op()`:

```py
mean = nn.op(tf.reduce_mean)
c = mean(a)
assert isinstance(c, nn.Tensor)

maximum = nn.op(tf.maximum, binary=True)
c = maximum(a, b)
assert isinstance(c, nn.Tensor)
```

or convert it to a `tf.Tensor` object using the `tf()` method or `nn.tf()` function:

```py
b = a.tf()
assert isinstance(b, tf.Tensor)

b = nn.tf(a)
assert isinstance(b, tf.Tensor)
```

See more examples [here][examples].

## Installation

Requirements:

-   TensorFlow >= 2.0
-   Python >= 3.6

Install from PyPI (recommended):

```sh
pip install nn
```

Alternatively, install from source:

```sh
git clone https://github.com/marella/nn.git
cd nn
pip install -e .
```

[TensorFlow] should be installed separately.

## Testing

To run tests, install dependencies:

```sh
pip install -e .[tests]
```

and run:

```sh
pytest tests
```

[tensorflow]: https://www.tensorflow.org/install
[examples]: https://github.com/marella/train/tree/master/examples
