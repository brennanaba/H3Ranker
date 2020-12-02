# -*- coding: utf-8 -*-
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import kl_divergence, mse, CategoricalCrossentropy
from keras.layers import Activation, Add, Conv2D, SpatialDropout2D, Permute, ReLU, Input, BatchNormalization, Layer, Conv1D, SpatialDropout1D
from keras import backend as K
import numpy as np
from numba import jit
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
latest = os.path.join(current_directory,"models/kullback_centered_gaussian_10blocks_50dropout_12binseparation.h5")

# REMEMBER TO pip install . EACH TIME YOU UPDATE

angle_steps = 12.0
bins = np.arange(-180.,181,angle_steps)
mean_angle_bins = (bins[1:] + bins[:-1]) / 2
classes = len(bins)

dist_bins = np.linspace(3,17,classes - 2)
dist_bins = np.append(dist_bins,20)
mean_dist_bins = (dist_bins[1:] + dist_bins[:-1]) / 2

def JensenShannonDivergence(y_true, y_pred):
    y_mean = (y_true + y_pred)/2
    return (kl_divergence(y_true,y_mean) + kl_divergence(y_pred, y_mean))/2


@jit
def encode(x, classes):
    one_hot = np.zeros(classes)
    one_hot[x] = 1
    return one_hot

@jit
def one_hot(num_list, classes = 21):
    end_shape = (len(num_list), classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        finish[i] = encode(num_list[i], classes)
    return finish

class Expand_Dimensions(Layer):
    def __init__(self, **kwargs):
        super(Expand_Dimensions, self).__init__(**kwargs)

    def call(self, x):
        a = K.expand_dims(x, axis=-2)
        b = K.permute_dimensions(a, (0,2,1,3))
        return a + b

def deep2d_model(lr = 1e-2, blocks = 25, blocks_1d = 5):
    inp = Input(shape=(None, 21))
    mix1 = Conv1D(64, kernel_size= 17, strides = 1, padding= "same", name = "1Dconv_1", trainable = True)(inp)
    block_start_1d = SpatialDropout1D(0.5)(mix1)
    
    for i in range(blocks_1d):
        block_conv1_1d = Conv1D(64, kernel_size= 17, strides = 1, padding= "same", trainable = True)(block_start_1d)
        block_act_1d = ReLU()(block_conv1_1d)
        block_drop_1d = SpatialDropout1D(0.5)(block_act_1d)
        block_conv2_1d = Conv1D(64, kernel_size= 17, strides = 1, padding= "same", trainable = True)(block_drop_1d)
        block_norm_1d = BatchNormalization(scale = True, trainable = True)(block_conv2_1d)
        block_start_1d = Add()([block_start_1d,block_norm_1d])

    
    activate_1d = ReLU()(block_start_1d)
    block_start = Expand_Dimensions()(activate_1d)
    for i in range(blocks):
        block_conv1 = Conv2D(64, kernel_size= 5, strides = 1, padding= "same", trainable = True, dilation_rate = 2**(i%6))(block_start)
        block_act = ReLU()(block_conv1)
        block_drop = SpatialDropout2D(0.5)(block_act)
        block_conv2 = Conv2D(64, kernel_size= 5, strides = 1, padding= "same", trainable = True, dilation_rate = 2**(i%6))(block_drop)
        block_norm = BatchNormalization(scale = True, trainable = True)(block_conv2)
        block_start = Add()([block_start,block_norm])
        
    block_end = block_start
    activate = ReLU()(block_end)
    drop = SpatialDropout2D(0.5)(activate)
    
    
    # This should be number of classes, defined by the number of bins
    dist_1 = Conv2D(classes, kernel_size= 3, strides = 1, padding= "same", trainable = True)(drop)
    dist_2 = Permute(dims=(2,1,3))(dist_1)
    dist_symmetry = Add()([dist_1,dist_2])
    dist_end = Activation(activation='softmax', name= "distance_out")(dist_symmetry)
    
    omega_1 = Conv2D(classes, kernel_size= 3, strides = 1, padding= "same", trainable = True)(drop)
    omega_2 = Permute(dims=(2,1,3))(omega_1)
    omega_symmetry = Add()([omega_1,omega_2])
    omega_end = Activation(activation='softmax', name= "omega_out")(omega_symmetry)
    
    theta_1 = Conv2D(classes, kernel_size= 3, strides = 1, padding= "same", trainable = True)(drop)
    theta_end = Activation(activation='softmax', name = "theta_out")(theta_1)
    
    phi_1 = Conv2D(classes, kernel_size= 3, strides = 1, padding= "same", trainable = True)(drop)
    phi_end = Activation(activation='softmax', name = "phi_out")(phi_1)
    
    model = Model(inp, outputs = [dist_end,omega_end,theta_end,phi_end])
    model.compile(optimizer = Adam(lr), loss = CategoricalCrossentropy(), loss_weights = [1/6,1/6,2/6,2/6])
    return model
