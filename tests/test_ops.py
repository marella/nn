import pytest
import nn
import tensorflow as tf


@pytest.mark.parametrize('symbol', ['+', '-', '*', '/', '@'])
@pytest.mark.parametrize('size', [1, 3])
def test_magic_ops(symbol, size, U):

    def op(x, y):
        return eval(f'x {symbol} y')

    shape = (size, size)
    a, b = U.random(shape), U.random(shape)
    p, q = tf.convert_to_tensor(a), tf.convert_to_tensor(b)
    x, y = nn.tensor(a), nn.tensor(b)
    actual = op(p, q)
    assert isinstance(actual, tf.Tensor)
    actual = actual.numpy()
    args = [(x, j) for j in [b, q, y]]
    args += [(i, y) for i in [a, p]]
    for i, j in args:
        result = op(i, j)
        assert isinstance(result, nn.Tensor)
        assert U.equal(result, actual)
