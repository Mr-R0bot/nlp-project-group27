import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def data():
    '''
    Data providing function:

    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''

    
    seq_length = 100
    unknown_token = 'U'
    filename_train = "alice-in-wonderland.txt"
    train_text = open(filename_train).read()
    train_text = train_text.lower()
    train_text = [c if c not in ['\xe2','\x80','\x99','\x98','\x9d','\x9c','0','3','*','[',']','_'] \
    else unknown_token for c in train_text]
    

    

    filename_test = "chapter12.txt"
    test_text = open(filename_test).read()
    test_text = test_text.lower()
    test_text = [c if c not in ['\xe2','\x80','\x99','\x98','\x9d','\x9c','0','3','*','[',']','_'] \
    else unknown_token for c in test_text]
    

    # create mapping of unique chars to integers, and a reverse mapping
    chars = sorted(list(set(train_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))

    # summarize the loaded data
    n_chars = len(train_text)
    n_vocab = len(chars)
    print "Total Characters: ", n_chars
    print "Total Vocab: ", n_vocab

    n_chars_test = len(test_text)

    dataX = []
    dataY = []
    dataX2 = []
    dataY2 = []
    for i in range(0, n_chars - seq_length, 1):
    	seq_in = train_text[i:i + seq_length]
    	seq_out = train_text[i + seq_length]
    	dataX.append([char_to_int[char] for char in seq_in])
    	dataY.append(char_to_int[seq_out])

    for i in range(0, n_chars_test - seq_length, 1):
    	seq_in2 = test_text[i:i + seq_length]
    	seq_out2 = test_text[i + seq_length]
    	dataX2.append([char_to_int[char] for char in seq_in2])
    	dataY2.append(char_to_int[seq_out2])

    n_patterns = len(dataX)
    print "Total Patterns: ", n_patterns
    n_patterns_test = len(dataX2)

    X_train = numpy.reshape(dataX, (n_patterns, seq_length, 1))
    X_train = X_train / float(n_vocab)
    Y_train = np_utils.to_categorical(dataY)

    X_test= numpy.reshape(dataX2, (n_patterns_test, seq_length, 1))
    X_test = X_test / float(n_vocab)
    Y_test = np_utils.to_categorical(dataY2)

    return X_train, Y_train, X_test, Y_test


#Function to create model
def create_model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout({{uniform(0, 0.4)}})) 
    model.add(LSTM(256))
    model.add(Dropout({{uniform(0, 0.4)}}))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    checkpointer = ModelCheckpoint(filepath='2x256_weights.h5', monitor = 'loss', verbose = 1, save_best_only = True)

    model.fit(X_train, Y_train, nb_epoch={{choice([10, 25, 50])}}, batch_size=64, callbacks=[checkpointer])
    model.save('model_2x256.h5')

    loss = model.evaluate(X_test, Y_test, verbose=0)
    
    print('Cross entropy loss on Test data:', loss)
    return {'loss': loss, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=1,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    best_model.save('final_model_2x256.h5')