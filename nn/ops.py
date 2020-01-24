import tensorflow as tf
from tensorflow.keras.backend import *

detach = tf.stop_gradient
gather = tf.gather

ops = [
    'abs', 'all', 'any', 'argmax', 'argmin', 'batch_dot', 'batch_flatten',
    'batch_normalization', 'bias_add', 'binary_crossentropy', 'cast',
    'categorical_crossentropy', 'clip', 'conv1d', 'conv2d', 'conv2d_transpose',
    'conv3d', 'cos', 'count_params', 'ctc_batch_cost', 'ctc_decode',
    'ctc_label_dense_to_sparse', 'cumprod', 'cumsum', 'dot', 'dropout', 'elu',
    'equal', 'exp', 'expand_dims', 'flatten', 'get_value', 'gradients',
    'greater', 'greater_equal', 'hard_sigmoid', 'in_top_k', 'int_shape',
    'is_sparse', 'l2_normalize', 'learning_phase_scope', 'less', 'less_equal',
    'local_conv1d', 'local_conv2d', 'log', 'max', 'maximum', 'mean', 'min',
    'minimum', 'moving_average_update', 'name_scope', 'ndim',
    'normalize_batch_in_training', 'not_equal', 'one_hot', 'ones_like',
    'permute_dimensions', 'pool2d', 'pool3d', 'pow', 'prod', 'relu', 'repeat',
    'repeat_elements', 'reshape', 'resize_images', 'resize_volumes', 'reverse',
    'round', 'separable_conv2d', 'set_value', 'sigmoid', 'sign', 'sin',
    'softmax', 'softplus', 'softsign', 'sparse_categorical_crossentropy',
    'spatial_2d_padding', 'spatial_3d_padding', 'sqrt', 'square', 'squeeze',
    'stack', 'std', 'stop_gradient', 'sum', 'switch', 'tanh',
    'temporal_padding', 'tile', 'to_dense', 'transpose', 'update',
    'update_add', 'update_sub', 'var', 'zeros_like'
] + ['detach', 'gather']

magic_ops = [
    '__abs__', '__add__', '__and__', '__bool__', '__div__', '__eq__',
    '__floordiv__', '__ge__', '__getitem__', '__gt__', '__invert__',
    '__iter__', '__le__', '__len__', '__lt__', '__matmul__', '__mod__',
    '__mul__', '__ne__', '__neg__', '__nonzero__', '__or__', '__pow__',
    '__radd__', '__rand__', '__rdiv__', '__rfloordiv__', '__rmatmul__',
    '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rsub__', '__rtruediv__',
    '__rxor__', '__sub__', '__truediv__', '__xor__'
]

binary_ops = {
    'gather', '__add__', '__radd__', '__sub__', '__rsub__', '__mul__',
    '__rmul__', '__div__', '__rdiv__', '__truediv__', '__rtruediv__',
    '__floordiv__', '__rfloordiv__', '__mod__', '__rmod__', '__lt__', '__le__',
    '__gt__', '__ge__', '__ne__', '__eq__', '__and__', '__rand__', '__or__',
    '__ror__', '__xor__', '__rxor__', '__getitem__', '__pow__', '__rpow__',
    '__matmul__', '__rmatmul__'
}
