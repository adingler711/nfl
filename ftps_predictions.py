from sklearn import preprocessing # scale the X varaibles to the same scale
from keras.preprocessing.sequence import pad_sequences # Pad your sequences so they are the same length
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils.training_utils import multi_gpu_model
from pyimagesearch.minigooglenet import MiniGoogLeNet
from keras.utils import np_utils, to_categorical
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def print_results(model, history, testx, testy):

    pred_classes = model.predict_classes(testx, verbose=0)
    cm_normalized = pd.crosstab(testy, pred_classes,
                                rownames=['True'],
                                colnames=['Predicted']).apply(
        lambda r: 100.0 * r / r.sum())

    print("Normalized Confusion Matrix after optimal threshold:")
    print cm_normalized

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def convert_onehot_y(y_var, binary_threshold):

    y_var_indicator = np.array(y_var)
    y_var_indicator[y_var_indicator <= binary_threshold] = 0
    y_var_indicator[y_var_indicator > binary_threshold] = 1
    one_hot_labels = to_categorical(y_var_indicator,
                                    num_classes=len(np.unique(y_var_indicator)))

    return one_hot_labels

# Convert labels to categorical one-hot encoding