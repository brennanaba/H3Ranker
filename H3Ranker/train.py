import numpy as np
import pandas as pd
from numba import jit
from H3Ranker.network import dist_bins, mean_dist_bins, bins, mean_angle_bins, deep2d_model, one_hot
from keras.utils import Sequence
from keras.callbacks import EarlyStopping,  ModelCheckpoint
import os
import sys

current_directory = os.path.dirname(os.path.realpath(__file__))
val_table = pd.read_csv(os.path.join(current_directory, "validation_data.csv"))
data = pd.read_csv(os.path.join(current_directory, "train_data.csv"))

dict_ = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19', '-': '20'}
classes = len(bins)

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

@jit
def encode_distances(matrix, std = (dist_bins[3] - dist_bins[2])):
    """ Goes throught each distance measurement and encodes it into different bins as a gaussian.
    """
    end_shape = (matrix.shape[0], matrix.shape[1], classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        for j in range(end_shape[1]):
            finish[i,j] = gauss_encode_distance(matrix[i,j], std = std)
    return finish

@jit
def encode_angles(matrix, std = (bins[3] - bins[2])):
    """ Goes throught each angle measurement and encodes it into different bins as a gaussian.
    """
    end_shape = (matrix.shape[0], matrix.shape[1], classes)
    finish = np.zeros(end_shape)
    for i in range(end_shape[0]):
        for j in range(end_shape[1]):
            finish[i,j] = gauss_encode_angles(matrix[i,j], std = std)
    return finish

class DataLoader(Sequence):
    'Loads data for Keras'
    def __init__(self, data,shuffle = True, batch_size = 8):
        'Initialization'
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.data)//self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        X, y = self.__data_generation(self.data[index*self.batch_size:self.batch_size*(index+1)].reset_index(drop=True))

        return X, y

    def on_epoch_end(self):
        if self.shuffle == True:
            self.data = self.data.sample(frac = 1).reset_index(drop=True)

    def __data_generation(self, data):
        'Generates data containing batch_size samples'
        batch_tfirst = []
        batch_tsecond = []
        batch_tsecond1 = []
        batch_tsecond2 = []
        batch_tlabels = []
        # Generate data
        for i in range(len(data)):
            pair = np.load(os.path.join(current_directory, "../../data/"+data.ID[i]+".npy"))
            pair[pair == -1] = -float("Inf")
            pair[np.isnan(pair)] = -float("Inf")
            first = encode_distances(pair[0])
            second = encode_angles(pair[1])
            second1 = encode_angles(pair[2])
            second2 = encode_angles(pair[3])
            seq = data.Sequence[i]
            first_in = one_hot(np.array([int(dict_[x]) for x in seq]))
            batch_tlabels.append(first_in)
            batch_tfirst.append(first)
            batch_tsecond.append(second)
            batch_tsecond1.append(second1)
            batch_tsecond2.append(second2)
            
        return  np.array(batch_tlabels), [np.array(batch_tfirst), np.array(batch_tsecond),np.array(batch_tsecond1),np.array(batch_tsecond2)]


if __name__ == "__main__":
    latest = os.path.join(current_directory, "models", str(sys.argv[1]))
    print("Saving model to: " + latest)
    model = deep2d_model(lr = 0.001)
    
    
    training_generator = DataLoader(data, batch_size=4)
    validation_generator = DataLoader(val_table, batch_size=4)
    
    es =  EarlyStopping(patience= 20, restore_best_weights= True)
    check = ModelCheckpoint(filepath=latest, save_best_only= True, save_weights_only= True)
    model.fit(training_generator,validation_data=validation_generator, epochs= 500, callbacks=[es,check], verbose = 2)
            
