import pytest
import nn
import tensorflow as tf
import numpy as np

sizes = [None, 1, 3, (1, 1), (3, 3)]


class TestTensor():

    @pytest.mark.parametrize('size', sizes)
    def test_tensor(self, size, U):
        a = U.random(size)
        p = tf.convert_to_tensor(a)

        x = nn.tensor(a)
        assert U.equal(x, p)
        assert nn.tf(x) is x.tf()

        x = nn.tensor(p)
        assert U.equal(x, p)
        assert x.tf() is p
        self.assert_properties(x, p)

        y = nn.tensor(x)
        assert y is x

        y = nn.Tensor(x)
        assert y is not x
        assert U.equal(y, x)
        assert y.tf() is x.tf()
        self.assert_properties(y, x)

    @pytest.mark.parametrize('size', sizes)
    @pytest.mark.parametrize('length', [1, 3])
    def test_tensors(self, size, length, U):
        L = [tf.convert_to_tensor(U.random(size)) for _ in range(length)]
        for P in [L, tuple(L)]:
            X = nn.tensors(P)
            if isinstance(P, tuple):
                assert isinstance(X, tuple)
            else:
                assert isinstance(X, list)
            assert len(X) == len(P)
            for x, p in zip(X, P):
                assert U.equal(x, p)
                assert x.tf() is p

            Q = nn.tf(X)
            if isinstance(P, tuple):
                assert isinstance(Q, tuple)
            else:
                assert isinstance(Q, list)
            assert len(Q) == len(P)
            for q, p in zip(Q, P):
                assert q is p

    @pytest.mark.parametrize('size', sizes)
    def test_variable_with_variable(self, size, U):
        a = U.random(size)
        p = tf.Variable(a)

        x = nn.variable(a)
        assert U.equal(x, p)
        assert nn.tf(x) is x.tf()

        x = nn.variable(p)
        assert U.equal(x, p)
        assert x.tf() is p
        self.assert_properties(x, p)

        y = nn.variable(x)
        assert y is x

        y = nn.Tensor(x)
        assert y is not x
        assert U.equal(y, x)
        assert y.tf() is x.tf()
        self.assert_properties(y, x)

    @pytest.mark.parametrize('size', sizes)
    def test_variable_with_tensor(self, size, U):
        a = U.random(size)
        p = tf.convert_to_tensor(a)

        x = nn.variable(a)
        assert U.equal(x, p)
        assert nn.tf(x) is x.tf()

        x = nn.variable(p)
        assert U.equal(x, p)
        assert x.tf() is not p

        y = nn.variable(x)
        assert y is x

        y = nn.Tensor(x)
        assert y is not x
        assert U.equal(y, x)
        assert y.tf() is x.tf()
        self.assert_properties(y, x)

    def assert_properties(self, a, b):
        for name in [
                '_id', 'device', 'dtype', 'graph', 'name', 'op', 'shape',
                'value_index'
        ]:
            if not hasattr(b, name):
                assert not hasattr(a, name)
                continue
            assert getattr(a, name) == getattr(b, name)


@pytest.mark.parametrize('size', sizes)
def test_op(size, U, mocker):
    a, b = U.random(size), U.random(size)
    p, q = tf.convert_to_tensor(a), tf.convert_to_tensor(b)
    x, y = nn.tensor(p), nn.tensor(q)

    mock = mocker.Mock(return_value=p)
    op = nn.op(mock)
    z = op(x, y)
    mock.assert_called_once_with(p, y)
    assert z.tf() is p

    mock = mocker.Mock(return_value=q)
    op = nn.op(mock, binary=True)
    z = op(x, y)
    mock.assert_called_once_with(p, q)
    assert z.tf() is q


class TestGradient():

    @pytest.mark.parametrize('size', sizes)
    def test_gradient(self, size, U):
        a, b = U.random(size), U.random(size)
        p, q = tf.Variable(a), tf.Variable(b)
        x, y = nn.variable(a), nn.variable(b)

        def tf_grad(sources):
            with tf.GradientTape() as tape:
                r = p * q
                r = tf.stop_gradient(r)
                r = r * p * q
                r = tf.reduce_sum(r)
                assert r.shape == ()
            grads = tape.gradient(r, sources)
            self.assert_isinstance(grads, tf.Tensor, sources)
            return grads

        def nn_grad(sources):
            with nn.GradientTape() as tape:
                z = x * y
                z = z.detach()
                z = z * x * y
                z = z.sum()
                assert z.shape == ()
            grads = tape.gradient(z, sources)
            self.assert_isinstance(grads, nn.Tensor, sources)
            return grads

        def assert_grad(s1, s2):
            grads = nn_grad(s1)
            actual = tf_grad(s2)
            assert U.equal(grads, actual)

        assert_grad([x, y], [p, q])
        assert_grad((x, y), (p, q))
        assert_grad(x, p)
        assert_grad(y, q)

    def assert_isinstance(self, v, t, s):
        if isinstance(s, list):
            assert isinstance(v, list)
        elif isinstance(s, tuple):
            assert isinstance(v, tuple)
        else:
            assert isinstance(v, t)
        if isinstance(v, (list, tuple)):
            assert len(v) == len(s)
            for i in v:
                assert isinstance(i, t)
