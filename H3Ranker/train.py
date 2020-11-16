import numpy as np
import pandas as pd
from numba import jit
from H3Ranker.network import dist_bins, mean_dist_bins, bins, mean_angle_bins, deep2d_model, one_hot
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

def batch_it(data, batch = 1, batchmin = 0):
    """ Batches training data into groups of `batch` with the same sequence lenght. 
    If there are less than `batchmin` with the same sequence length, they are discarded.
    """
    train_data = []
    train_labels = []
    data["seqlen"] = [len(str(x)) for x in data.Sequence.values]
    lens = np.unique(data.seqlen.values)
    dist_diff = (dist_bins[3] - dist_bins[2])
    angle_diff = (bins[3] - bins[2])
    for l in lens:
        df = data[data.seqlen == l]
        for j in range(len(df)//batch + 1):
            structs = df.ID.values[j*batch:(j+1)*batch]
            resol = df.Resolution.values[j*batch:(j+1)*batch]
            batch_tfirst = []
            batch_tsecond = []
            batch_tsecond1 = []
            batch_tsecond2 = []
            batch_tlabels = []
            for i in range(len(structs)):
                pair = np.load(os.path.join(current_directory, "../../data/"+structs[i]+".npy"))
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


if __name__ == "__main__":
    latest = os.path.join(current_directory, "models", str(sys.argv[1]))
    print("Saving model to: " + latest)
    model = deep2d_model(lr = 1e-4)
    
    
    val_data, val_labels = batch_it(val_table)
    train_data, train_labels = batch_it(data.sample(frac = 0.5),4,1)
    indices = np.arange(len(train_data))
    
    
    
    val_loss = []
    train_loss = float("Inf")
    best_loss = float("Inf")
    best_model = deep2d_model()
    
    o = 0
    print("Training loss    | Validation Loss")
    for j in range(500):
        np.random.shuffle(indices)
        val_loss_one = []
        train_loss_one = []
        for i in indices:
            train_loss_one += model.fit(train_labels[i], train_data[i], verbose = 0).history["loss"]
        train_loss = np.mean(train_loss_one)
        print(str(j) + " " + str(train_loss), end=" ")
        for i in range(len(val_data)):
            val_loss_one.append(model.evaluate(val_labels[i], val_data[i], verbose = 0)[0])
        loss = np.mean(val_loss_one)
        print(loss)
        val_loss.append(loss)
        if loss < best_loss:
            o = 0
            best_loss = loss
            best_model.set_weights(model.get_weights())
            best_model.save_weights(latest)
        else:
            o = o + 1
            if o > 20:
                break
            
