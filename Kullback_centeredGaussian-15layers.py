import numpy as np
import pandas as pd
from numba import jit

from keras.models import Model
from keras.optimizers import Adam
from keras.losses import KLDivergence
from keras.layers import Activation, Add, Conv2D, SpatialDropout2D, Permute, ReLU, Input, BatchNormalization


val_table = pd.read_csv("validation_data.csv")
data = pd.read_csv("train_data.csv")

angle_steps = 15.
bins = np.arange(-180.,181,angle_steps)
mean_angle_bins = (bins[1:] + bins[:-1]) / 2
classes = len(bins)

dist_bins = np.linspace(3,14,classes - 2)
dist_bins = np.append(dist_bins,18)
mean_dist_bins = (dist_bins[1:] + dist_bins[:-1]) / 2


@jit
def gauss_encode_distance(measured, std = (dist_bins[3] - dist_bins[2])):
    answer = np.zeros(classes)
    if measured < 0:
        answer[0] = 1
        return answer
    elif measured > dist_bins[-1]:
        answer[-1] = 1
        return answer
    else:
        answer[1:-1] = (1/std*np.sqrt(2*np.pi))*np.exp((-((mean_dist_bins - measured)/std)**2)/2)
        return answer/np.sum(answer)
    
@jit
def gauss_encode_angles(measured, std = (bins[3] - bins[2])):
    answer = np.zeros(classes)
    if measured < -180:
        answer[0] = 1
        return answer
    else:
        answer[1:] = (1/std*np.sqrt(2*np.pi))*np.exp((-(np.abs(measured%360 - mean_angle_bins%360)/std)**2)/2)
        return answer/np.sum(answer)

dict_ = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19'}

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

@jit
def encode_distances(matrix):
    end_shape = (matrix.shape[0], matrix.shape[1], classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        for j in range(end_shape[1]):
            finish[i,j] = gauss_encode_distance(matrix[i,j])
    return finish

@jit
def encode_angles(matrix):
    end_shape = (matrix.shape[0], matrix.shape[1], classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        for j in range(end_shape[1]):
            finish[i,j] = gauss_encode_angles(matrix[i,j])
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


def batch_it(data, batch = 1, batchmin = 0):
    train_data = []
    train_labels = []
    data["seqlen"] = [len(str(x)) for x in data.Sequence.values]
    lens = np.unique(data.seqlen.values)
    for l in lens:
        df = data[data.seqlen == l]
        for j in range(len(df)//batch + 1):
            structs = df.ID.values[j*batch:(j+1)*batch]
            batch_tfirst = []
            batch_tsecond = []
            batch_tsecond1 = []
            batch_tsecond2 = []
            batch_tlabels = []
            for i in range(len(structs)):
                pair = np.load("data/"+structs[i]+".npy")
                pair[pair == -1] = -float("Inf")
                pair[np.isnan(pair)] = -float("Inf")
                first = pair[0] 
                first = encode_distances(first)
                second = pair[1]
                second = encode_angles(second)
                second1 = pair[2]
                second1 = encode_angles(second1)
                second2 = pair[3]
                second2 = encode_angles(second2)
                seq = str(df[df.ID == structs[i]].Sequence.iloc[0])
                first_in = one_hot(np.array([int(dict_[x]) for x in seq]))
                if first_in.shape[0:2] ==  first.shape[0:2]:
                    batch_tlabels.append(first_in)
                    batch_tfirst.append(first)
                    batch_tsecond.append(second)
                    batch_tsecond1.append(second1)
                    batch_tsecond2.append(second2)
            if len(batch_tfirst) > batchmin:
                train_labels.append(np.array(batch_tlabels))
                train_data.append([np.array(batch_tfirst), np.array(batch_tsecond),np.array(batch_tsecond1),np.array(batch_tsecond2)])
    return train_data, train_labels



model = deep2d_model()

train_data, train_labels = batch_it(data,4,1)
val_data, val_labels = batch_it(val_table)


indices = np.arange(len(train_data))
np.random.seed(24)
np.random.shuffle(indices)


val_loss = []
train_loss = []
best_loss = float("Inf")
best_model = deep2d_model()


print("Training loss    | Validation Loss")
for j in range(150):
    val_loss_one = []
    train_loss_one = []
    for i in indices:
        train_loss_one += model.fit(train_labels[i], train_data[i], verbose = 0).history["loss"]
    tloss = np.mean(train_loss_one)
    print(str(j) + " " + str(tloss), end=" ")
    train_loss.append(tloss)
    for i in range(len(val_data)):
        val_loss_one.append(model.evaluate(val_labels[i], val_data[i], verbose = 0)[0])
    loss = np.mean(val_loss_one)
    print(loss)
    val_loss.append(loss)
    if loss < best_loss:
        best_loss = loss
        best_model.set_weights(model.get_weights())
        best_model.save_weights("models/kullback_centered_gaussian_15layers_50dropout_checkpoint.h5")

model.save_weights("models/kullback_centered_gaussian_15layers_50drop.h5")


