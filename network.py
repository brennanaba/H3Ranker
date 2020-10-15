# -*- coding: utf-8 -*-
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import KLDivergence
from keras.layers import Activation, Add, Conv2D, SpatialDropout2D, Permute, ReLU, Input, BatchNormalization
import numpy as np
from numba import jit


angle_steps = 15.
bins = np.arange(-180.,181,angle_steps)
mean_angle_bins = (bins[1:] + bins[:-1]) / 2
classes = len(bins)

dist_bins = np.linspace(3,14,classes - 2)
dist_bins = np.append(dist_bins,18)
mean_dist_bins = (dist_bins[1:] + dist_bins[:-1]) / 2


@jit
def encode(x, classes):
    one_hot = np.zeros(classes)
    one_hot[x] = 1
    return one_hot

@jit
def one_hot(num_list, classes = 20):
    end_shape = (len(num_list), len(num_list), classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        for j in range(end_shape[1]):
            finish[i,j] = encode(num_list[i], classes) + encode(num_list[j], classes) 
    return finish

def deep2d_model():
    inp = Input(shape=(None, None, 20))
    mix1 = Conv2D(64, kernel_size= 5, strides = 1, padding= "same", name = "2Dconv_1", trainable = True)(inp)
    mix2 = SpatialDropout2D(0.5)(mix1)
    
    block_start = mix2
    for i in range(15):
        block_conv1 = Conv2D(64, kernel_size= 5, strides = 1, padding= "same", trainable = True)(block_start)
        block_act = ReLU()(block_conv1)
        block_drop = SpatialDropout2D(0.5)(block_act)
        block_conv2 = Conv2D(64, kernel_size= 5, strides = 1, padding= "same", trainable = True)(block_drop)
        block_norm = BatchNormalization(scale = True)(block_conv2)
        block_start = Add()([block_start,block_norm])
        
    block_end = block_start
    activate = ReLU()(block_end)
    drop = SpatialDropout2D(0.5)(activate)
    
    
    # This should be number of classes
    dist_1 = Conv2D(classes, kernel_size= 3, strides = 1, padding= "same", trainable = True)(drop)
    dist_2 = Permute(dims=(2,1,3))(dist_1)
    dist_symmetry = Add()([dist_1,dist_2])
    dist_end = Activation(activation='softmax')(dist_symmetry)
    
    omega_1 = Conv2D(classes, kernel_size= 3, strides = 1, padding= "same", trainable = True)(drop)
    omega_2 = Permute(dims=(2,1,3))(omega_1)
    omega_symmetry = Add()([omega_1,omega_2])
    omega_end = Activation(activation='softmax')(omega_symmetry)
    
    theta_1 = Conv2D(classes, kernel_size= 3, strides = 1, padding= "same", trainable = True)(drop)
    theta_end = Activation(activation='softmax')(theta_1)
    
    phi_1 = Conv2D(classes, kernel_size= 3, strides = 1, padding= "same", trainable = True)(drop)
    phi_end = Activation(activation='softmax')(phi_1)
    
    model = Model(inp, outputs = [dist_end,omega_end,theta_end,phi_end])
    model.compile(optimizer = Adam(2e-3), loss = KLDivergence())
    return model