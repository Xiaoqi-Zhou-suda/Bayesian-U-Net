from keras import Model
from keras.regularizers import l2
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import numpy as np
import math
from scipy.special import logsumexp
from keras.callbacks import LearningRateScheduler
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint


def read_exploration_file(label_path, cpt_path):
    DATA = []
    label = pd.read_excel(label_path, header=None)
    for Cur_path, Cur_dirs, Cur_files in os.walk(cpt_path):
        for cur_dir in Cur_dirs:
            path = os.path.join(Cur_path, cur_dir)
            # print(path)
            for cur_path, cur_dirs, cur_files in os.walk(path):
                for cur_file in cur_files[:]:
                    path = os.path.join(cur_path, cur_file)
                    # print(path)
                    data = pd.read_excel(path, header=None)
                    data = np.array(data)
                    Data = data[:, 0:2]
                    DATA.append(Data)
    index = []
    for i in range(len(DATA)):
        if DATA[i].shape[0] >= 400:
            index.append(i)
    print("the total number of original data has:{}".format(len(index)))
    training_data = np.zeros((len(index), 400, 3))
    depth0 = np.arange(0, 40, 0.1)
    depth = np.arange(0, 40, 0.1)
    for i in range(training_data.shape[0] - 1):
        depth = np.row_stack((depth0, depth))
    # training_data[:,:,0]=depth[:,:]
    for i in range(len(index)):
        k = index[i]
        training_data[i, 0:400, 1:3] = DATA[k][0:400, :]
    training_data[310, 0:400, 1:3] = DATA[310][0:400, :]
    for i in range(training_data.shape[0]):
        for j in range(training_data.shape[1]):
            training_data[i, j, 0] = training_data[i, j, 2] * 1. / (10. * (training_data[i, j, 1]+0.01))
    training_label = np.zeros((len(index), 400, 1))
    for i in range(len(index)):
        k = index[i]
        l1 = np.max([10, label.iloc[k, 1]])
        # l1 = label.iloc[k, 1]
        l2 = label.iloc[k, 2]
        l3 = label.iloc[k, 3]
        # l4 = label.iloc[k, 4]
        l4 = np.min([390,label.iloc[k, 4]])
        # training_label[i, 0:l1] = 0
        # training_label[i, l1:l2] = 1
        # training_label[i, l2:l3] = 2
        # training_label[i, l3:l4] = 3
        # training_label[i, l4:400] = 4

        training_label[i, (l1 - 5) : (l1 + 5)] = 1
        training_label[i, (l2 - 5) : (l2 + 5)] = 1
        training_label[i, (l3 - 5) : (l3 + 5)] = 1
        training_label[i, (l4 - 5) : (l4 + 5)] = 1
    return training_data, training_label


def get_dataset(training_data, training_label, division_rate=0.8):
    X = []
    y = []
    input_size=224
    Length = training_data.shape[0]
    for i in range(0,Length - input_size + 1,4):
        for j in range(0,400 - input_size + 1,3):
            X.append(training_data[i:i + input_size, j:j+input_size, :])
            y.append(training_label[i:i + input_size, j:j+input_size, :])
    X = np.array(X)
    y = np.array(y)
    print('the shape of training data:{}'.format(X.shape))
    print('the shape of training label:{}'.format(y.shape))
    index = np.arange(0, X.shape[0], 1)
    index_train, index_test = train_test_split(index, test_size=0.1, random_state=42)
    X_train_original = X[index_train]
    y_train_original = y[index_train]
    X_test = X[index_test]
    y_test = y[index_test]
    num_training_examples = int(division_rate * X_train_original.shape[0])
    X_validation = X_train_original[num_training_examples:]
    y_validation = y_train_original[num_training_examples:]
    X_train = X_train_original[0:num_training_examples]
    y_train = y_train_original[0:num_training_examples]

    return X_train, y_train, X_train_original, y_train_original, X_validation, y_validation, X_test, y_test,index_train,index_test

def generate_arrats_from_memory(X_train_normalized, y_train,batch_size):
    x=X_train_normalized
    y=y_train
    ylen=y.shape[0]
    loop=ylen//batch_size
    while True:
        i=np.random.randint(0,loop)
        X=x[i * batch_size : (i+1) * batch_size]
        Y=y[i * batch_size : (i+1) * batch_size]
        yield X, Y


# 定义学习率衰减函数
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 20:
        lr *= 0.1
    elif epoch > 10:
        lr *= 0.5
    return lr


def unet(pretrained_weights=None, input_size=(224, 224, 3), dropout=0.5, tau=1.0):
    """
        Constructor for the class implementing a Bayesian neural network
        trained with the probabilistic back propagation method.

        @param X_train      Matrix with the features for the training data.
        @param y_train      Vector with the target variables for the
                            training data.
        @param n_hidden     Vector with the number of neurons for each
                            hidden layer.
        @param n_epochs     Numer of epochs for which to train the
                            network. The recommended value 40 should be
                            enough.
        @param normalize    Whether to normalize the input features. This
                            is recommended unles the input vector is for
                            example formed by binary features (a
                            fingerprint). In that case we do not recommend
                            to normalize the features.
        @param tau          Tau value used for regularization
        @param dropout      Dropout rate for all the dropout layers in the
                            network.
    """
    N = 400
    lengthscale = 1e-2
    reg0 = lengthscale ** 2 * (1 - 0.2*dropout) / (2. * N * tau)
    reg1 = lengthscale ** 2 * (1 - 0.4*dropout) / (2. * N * tau)
    reg2 = lengthscale ** 2 * (1 - 0.6*dropout) / (2. * N * tau)
    reg3 = lengthscale ** 2 * (1 - 0.8*dropout) / (2. * N * tau)
    reg4 = lengthscale ** 2 * (1 - dropout) / (2. * N * tau)
    inputs = Input(input_size)
    drop0=Dropout(0.2*dropout)(inputs,training=True)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg0))(drop0)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg0))(conv1)
    drop1 = Dropout(0.4*dropout)(conv1, training=True)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg1))(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg1))(conv2)
    drop2 = Dropout(0.6*dropout)(conv2, training = True)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg2))(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg2))(conv3)
    drop3 = Dropout(0.8*dropout)(conv3, training=True)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg3))(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg3))(conv4)
    drop4 = Dropout(dropout)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg4))(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(reg4))(conv5)
    drop5 = Dropout(dropout)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    drop6 = Dropout(0.8 * dropout)(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'
                   )(drop6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal'
                   )(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    drop7 = Dropout(0.8 * dropout)(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'
                   )(drop7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal'
                   )(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    drop8 = Dropout(0.8 * dropout)(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    drop9 = Dropout(0.8 * dropout)(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # model.summary()

    # if (pretrained_weights):
    #     model.load_weights(pretrained_weights)

    return model

def test_generator(X_test):
    x = X_test
    num = X_test.shape[0]
    for i in range(num):
        X = x[i]
        X = np.reshape(X, (1,) + X.shape)
        yield X


def predict(model, mean_X_train, std_X_train, X_test, y_test, dropout,tau):
    """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

    X_test = np.array(X_test,dtype='float32')
    y_test = np.array(y_test,dtype='float32')
    print(y_test.shape)
    np.savetxt('./real_output_1m.txt',y_test[:,:,:,0].flatten(),'%.3f')
    # We normalize the test set

    X_test_normalized = (X_test - np.full(X_test.shape, mean_X_train)) / \
                        np.full(X_test.shape, std_X_train)

    # We compute the predictive mean and variance for the target variables
    # of the test data

    model = model
    standard_pred = model.predict_generator(test_generator(X_test_normalized),X_test.shape[0], verbose=0)
    # standard_pred = standard_pred * std_y_train_flatten + mean_y_train
    standard_pred[standard_pred>=0.5]=1
    standard_pred[standard_pred<=0.5]=0
    print('standard_pred finished')
    np.savetxt('./CPT/prediction_{}_{}.txt'.format(dropout,tau),standard_pred.flatten(),'%.3f')
    rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze()) ** 2.) ** 0.5

    T = 20

    # Yt_hat = np.array([model.predict_generator(test_generator(X_test_normalized), X_test.shape[0], verbose=1) for _ in range(T)])
    # MC_pred = np.mean((Yt_hat * std_y_train + mean_y_train),0)

    MC_pred = 0
    sum=0
    with tqdm(total=T, desc='Prediction', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for i in range(T):
            Yt_hat = model.predict_generator(test_generator(X_test_normalized), X_test.shape[0], verbose=0)
            sum=sum+np.exp(-0.5 * tau * (Yt_hat - y_test) ** 2.)
            MC_pred = MC_pred + Yt_hat / T
            pbar.update(1)
        sum = np.log(sum)

    # MC_pred = np.mean(Yt_hat, 0)
    MC_pred=np.array(MC_pred)
    Yt=MC_pred.copy()
    Yt[Yt>=0.5]=1
    Yt[Yt<0.5]=0
    rmse = np.mean((y_test.squeeze() - Yt.squeeze()) ** 2.,dtype='float32') ** 0.5
    test_ll = -sum  + np.log(T) + 0.5 * np.log(2 * np.pi) - 0.5 * np.log(tau)
    test_ll = np.mean(test_ll.squeeze(), axis=0)
    ll = np.mean(test_ll)
    np.savetxt('./test_ll_ori/test_ll_{}_{}.txt'.format(dropout, tau), test_ll)

    # Compute uncertainty scores
    entropy = -np.apply_along_axis(lambda x: np.sum(x * np.log(x+0.001)), axis=0, arr=MC_pred)
    # predictive_variance = np.apply_along_axis(lambda x: np.mean(x),axis=0, arr=rmse ** 2)

    # Compute overall uncertainty score
    uncertainty_score = entropy + rmse

    # Compute accuracy and uncertainty for each sample
    # accuracy = np.mean(np.argmax(Yt_hat, axis=0) == np.argmax(y_test, axis=0))
    accuracy = np.mean(Yt == y_test)
    uncertainty = np.mean(uncertainty_score)

    # We are done!
    return rmse_standard_pred, rmse, ll, uncertainty, accuracy, MC_pred


def grid_search(X_train, y_train, dropout_rates, tau_values, X_validation, y_validation):
    # we perform the grid-search to select the best hyperparameters based on the highest loglikelihood value
    best_ll = -float('inf')
    best_tau = 0
    best_dropout = 0
    best_network = None
    for dropout_rate in dropout_rates:
        for tau in tau_values:
            print('Grid search step: Tau: ' + str(tau) + ' Dropout rate: ' + str(dropout_rate))
            network = unet(tau=tau, dropout=dropout_rate)
            network.fit(X_train, y_train, batch_size=20, epochs=2, verbose=1)
            ##we obtain the test RMSE and the test LL from the validation sets
            # error, MC_error, ll, uncertainty, accuracy = predict(network, mean_X_train, std_X_train, X_validation,
                                                                 # y_validation, best_tau)
            error, MC_error, ll = network.predict()
            if (ll > best_ll):
                best_ll = ll
                best_network = network
                best_tau = tau
                best_dropout = dropout_rate
                print('Best log_likelihood changed to: ' + str(best_ll))
                print('Best tau changed to: ' + str(best_tau))
                print('Best dropout rate changed to: ' + str(best_dropout))

                ##storing the validation results

                with open('./results/_RESULTS_VALIDATION_RMSE.txt', 'w') as myfile:
                    myfile.write('Dropout_Rate:' + repr(dropout_rate) + 'Tau' + repr(tau) + "::")
                    myfile.write(repr(error) + '\n')

                with open('./result/_RESULTS_VALIDATION_MC_RMSE.txt', "a") as myfile:
                    myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                    myfile.write(repr(MC_error) + '\n')

                with open('./result/_RESULTS_VALIDATION_LL.txt', "a") as myfile:
                    myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
                    myfile.write(repr(ll) + '\n')

        return best_ll, best_tau, best_dropout, best_network
