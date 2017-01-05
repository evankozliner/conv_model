from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

NUM_TEST_SAMPLES = 5000
NUM_TRAIN_SAMPLES = 5000
EXT = '.pickle'

def get_next_batch(batch_num, X, Y, batch_size):
    # return values from the shuffled array for both columns from batch_num * batch_size
    start_index = batch_num * batch_size
    end_index = batch_num * batch_size + batch_size
    return X[start_index:end_index], Y[start_index:end_index]

def reshuffle_data(X,Y):
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(Y)
    return X,Y

def one_hot_encode(y):
    y = y.astype(int)
    encoded = np.zeros((y.shape[0], 2))
    encoded[np.arange(y.shape[0]), y] = 1
    return encoded
    
def shuffle_images_and_labels(images,labels):
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    images = images.reshape(images.shape[0], images.shape[1] * images.shape[2] * images.shape[3])
    labels = labels.reshape(labels.shape[0])
    return images, labels

def pluck_data(dataset):
    data_ben = np.load(dataset + '/' + 'Benign' + EXT)
    data_mal = np.load(dataset + '/' + 'Malignant' + EXT)
    Y_mal = np.ones(data_mal.shape[0]).reshape(data_mal.shape[0],1)
    Y_ben = np.zeros(data_ben.shape[0]).reshape(data_ben.shape[0],1)
    images = np.vstack([data_ben, data_mal])
    labels = np.vstack([Y_ben, Y_mal])
    return shuffle_images_and_labels(images,labels)

def plot_confusion_matrix(cm, labels,title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def print_confusion_scores(conf):
    tp = conf[1,1]
    tn = conf[0,0]
    fp = conf[0,1]
    fn = conf[1,0]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall_sens = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    fp_rate = fp / (fp + tp)
    fn_rate = fn / (fn + tn)
    print "Accuracy : " + str(accuracy)
    print "Recall / Sensitvity : " + str(recall_sens)
    print "Precision : " + str(precision)
    print "Specificity : " +str(specificity)
    print "False Positive Rate : " + str(fp_rate)
    print "False Negative Rate : " + str(fn_rate)

