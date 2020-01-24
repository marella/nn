import pytest
import tensorflow as tf
import numpy as np


@pytest.fixture(scope='session', autouse=True)
def disable_tensorflow_warnings():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@pytest.fixture(scope='session')
def U():
    return Utils()


class Utils():

    def __init__(self):
        self.epsilon = 1e-7

    def random(self, size, nozeros=True):
        a = np.random.random(size)
        if nozeros:
            a += self.epsilon
        return a

    def equal(self, a, b):
        if isinstance(a, list) and not isinstance(b, list):
            return False
        if isinstance(a, tuple) and not isinstance(b, tuple):
            return False
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return False
            for i, j in zip(a, b):
                if not self.equal(i, j):
                    return False
            return True
        if hasattr(a, 'numpy'):
            a = a.numpy()
        if hasattr(b, 'numpy'):
            b = b.numpy()
        return np.array_equal(a, b)
