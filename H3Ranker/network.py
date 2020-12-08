# -*- coding: utf-8 -*-
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy, KLDivergence
from keras.layers import Activation, Add, Conv2D, SpatialDropout2D, Permute, ReLU, Input, BatchNormalization, Layer, \
    Conv1D, SpatialDropout1D, Multiply, Reshape, GlobalAveragePooling2D
from keras import backend as K
import numpy as np
from numba import jit
import os

current_directory = os.path.dirname(os.path.realpath(__file__))

# REMEMBER TO pip install . EACH TIME YOU UPDATE

angle_steps = 12.0
bins = np.arange(-180., 181, angle_steps)
mean_angle_bins = (bins[1:] + bins[:-1]) / 2
classes = len(bins)

dist_bins = np.linspace(3, 17, classes - 2)
dist_bins = np.append(dist_bins, 20)
mean_dist_bins = (dist_bins[1:] + dist_bins[:-1]) / 2


@jit
def encode(x, classes):
    """ One hot encodes a scalar x into a vector of length classes.
    This is the function used for Sequence encoding.
    """
    one_hot = np.zeros(classes)
    one_hot[x] = 1
    return one_hot


@jit
def one_hot(num_list, classes=21):
    """ One hot encodes a 1D vector x.
    This is the function used for Sequence encoding.
    """
    end_shape = (len(num_list), classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        finish[i] = encode(num_list[i], classes)
    return finish


class ExpandDimensions(Layer):
    """ Keras layer that transforms a 1D tensor into a 2D tensor by pairwise addition.
    """

    def __init__(self, **kwargs):
        super(ExpandDimensions, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        a = K.expand_dims(x, axis=-2)
        b = K.permute_dimensions(a, (0, 2, 1, 3))
        return a + b


class SqueezeAndExcite2D(Layer):
    """ Keras layer for a Squeeze and Excitation block.

    Basically finds which channels are contributing and blocks out the ones that aren't
    """

    def __init__(self, channels, squeezed_channels=None, **kwargs):
        super(SqueezeAndExcite2D, self).__init__(**kwargs)

        self.channels = channels
        if squeezed_channels is None:
            self.squeezed_channels = self.channels // 4
        else:
            self.squeezed_channels = squeezed_channels

        self.pool = GlobalAveragePooling2D()
        self.reshape = Reshape((1, 1, self.channels))
        self.squeeze = Conv2D(self.squeezed_channels, activation="relu", kernel_size=1, trainable=True)
        self.excite = Conv2D(self.channels, activation="sigmoid", kernel_size=1, trainable=True)
        self.end = Multiply()

    def call(self, x, **kwargs):
        y = self.pool(x)
        y = self.reshape(y)
        y = self.squeeze(y)
        y = self.excite(y)
        return self.end([x, y])


class ResidualBlock1D(Layer):
    """ 1D Residual Block
    """
    def __init__(self, channels=64, kernel_size=17, dropout=0.5, **kwargs):
        super(ResidualBlock1D, self).__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.conv_1 = Conv1D(self.channels, activation="relu", kernel_size=self.kernel_size, strides=1, padding="same",
                             trainable=True)
        self.drop_1 = SpatialDropout1D(self.dropout)
        self.conv_2 = Conv1D(self.channels, kernel_size=self.kernel_size, strides=1, padding="same", trainable=True)
        self.batch_norm = BatchNormalization(scale=True, trainable=True)
        self.end = Add()

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.drop_1(x)
        x = self.conv_2(x)
        x = self.batch_norm(x)
        return self.end([inputs, x])


class ResidualBlock2D(Layer):
    """ 2D Residual block with the possibility of Squeeze and Excitation at then end

    """
    def __init__(self, channels=64, kernel_size=5, dropout=0.5, dilation=1, excite=True, squeeze_channels=None,
                 **kwargs):
        super(ResidualBlock2D, self).__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilation = dilation
        self.excite = excite
        self.squeeze_channels = squeeze_channels

        self.conv_1 = Conv2D(self.channels, activation="relu", kernel_size=self.kernel_size, strides=1, padding="same",
                             dilation_rate=self.dilation, trainable=True)
        self.drop_1 = SpatialDropout2D(self.dropout)
        self.conv_2 = Conv2D(self.channels, kernel_size=self.kernel_size, strides=1, padding="same",
                             dilation_rate=self.dilation, trainable=True)
        self.batch_norm = BatchNormalization(scale=True, trainable=True)
        self.end = Add()

        if self.excite:
            self.excitation = SqueezeAndExcite2D(self.channels, self.squeeze_channels)

    def call(self, inputs, **kwargs):
        x = self.conv_1(inputs)
        x = self.drop_1(x)
        x = self.conv_2(x)
        x = self.batch_norm(x)
        if self.excite:
            x = self.excitation(x)
        return self.end([inputs, x])


def deep2d_model(lr=1e-3, blocks=15, blocks_1d=5):
    """ Main model function (hopefully self-explanatory).
    """
    inp = Input(shape=(None, 21))
    mix1 = Conv1D(64, kernel_size=17, strides=1, padding="same", name="1Dconv_1", trainable=True)(inp)
    residual_1d = SpatialDropout1D(0.5)(mix1)

    for i in range(blocks_1d):
        residual_1d = ResidualBlock1D(channels=64, kernel_size=17, dropout=0.5)(residual_1d)

    activate_1d = ReLU()(residual_1d)
    residual_2d = ExpandDimensions()(activate_1d)
    for i in range(blocks):
        residual_2d = ResidualBlock2D(dilation=2 ** (i % 5))(residual_2d)

    activate = ReLU()(residual_2d)
    drop = SpatialDropout2D(0.5)(activate)

    # This should be number of classes, defined by the number of bins
    dist_1 = Conv2D(classes, kernel_size=3, strides=1, padding="same", trainable=True)(drop)
    dist_2 = Permute(dims=(2, 1, 3))(dist_1)
    dist_symmetry = Add()([dist_1, dist_2])
    dist_end = Activation(activation='softmax', name="distance_out")(dist_symmetry)

    omega_1 = Conv2D(classes, kernel_size=3, strides=1, padding="same", trainable=True)(drop)
    omega_2 = Permute(dims=(2, 1, 3))(omega_1)
    omega_symmetry = Add()([omega_1, omega_2])
    omega_end = Activation(activation='softmax', name="omega_out")(omega_symmetry)

    theta_1 = Conv2D(classes, kernel_size=3, strides=1, padding="same", trainable=True)(drop)
    theta_end = Activation(activation='softmax', name="theta_out")(theta_1)

    phi_1 = Conv2D(classes, kernel_size=3, strides=1, padding="same", trainable=True)(drop)
    phi_end = Activation(activation='softmax', name="phi_out")(phi_1)

    model = Model(inp, outputs=[dist_end, omega_end, theta_end, phi_end])
    model.compile(optimizer=Adam(lr), loss=KLDivergence(), loss_weights=[1 / 6, 1 / 6, 2 / 6, 2 / 6])
    return model
