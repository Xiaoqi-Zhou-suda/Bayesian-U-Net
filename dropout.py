import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
gpus = tf.config.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)
import os
import time
from CNN_dropout import *
from keras.callbacks import LearningRateScheduler
import numpy as np
import tensorflow as tf

print('version = %s\nIs gpu available? %s' % (tf.__version__, tf.config.list_physical_devices('GPU')))
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

_RESULTS_TEST_LL = './results/_RESULTS_TEST_LL.txt'
_RESULTS_TEST_TAU = './results/_RESULTS_TEST_TAU.txt'
_RESULTS_TEST_RMSE = './results/_RESULTS_TEST_RMSE.txt'
_RESULTS_TEST_MC_RMSE = './results/_RESULTS_TEST_MC_RMSE.txt'
_RESULTS_TEST_LOG = "./results/RESULTS_TEST_LOG.txt"
_RESULTS_TEST_ACCURACY = './results/_RESULTS_TEST_ACCURACY.txt'
_RESULTS_TEST_UNCERTAINTY = './results/_RESULTS_TEST_UNCERTAINTY.txt'
label_path = r'./data/CPT.xlsx'
cpt_path = './data/training_data'

splits = 1
errors = []
MC_errors = []
lls = []
accuracys = []
uncertaintys = []
training_data, training_label = read_exploration_file(label_path, cpt_path)
# for split in range(int(splits))
start = time.time()
X_train, y_train, X_train_original, y_train_original, X_validation, y_validation, X_test, y_test,index_train,index_test = get_dataset(training_data, training_label,0.8)
print(y_validation.shape)
# np.savetxt('./real_output_0.4m.txt',y_validation.squeeze().flatten(),'%.3f')
#normailize the training dataset
mean_X_train = np.mean(X_train,axis=0,dtype='float32')
std_X_train=np.std(X_train,0,dtype='float32')
std_X_train[ std_X_train == 0 ] = 1
X_train_normalized = (X_train - np.full(X_train.shape, mean_X_train))/np.full(X_train.shape,std_X_train)
mean_y_train = np.mean(y_train,0)
std_y_train = np.std(y_train,0)
std_y_train[ std_y_train == 0 ] = 1
##define the learning_rate decay
lr_scheduler = LearningRateScheduler(lr_schedule)

# list of hyperparameters which we will try out using grid search
dropout_rates = [0.441]
tau_values = [0.507]

##find the best paras
best_ll = -float('inf')
best_tau = 0
best_dropout = 0
best_network = None
for dropout_rate in dropout_rates:
    for tau in tau_values:
        print('Grid search step: Tau: ' + str(tau) + ' Dropout rate: ' + str(dropout_rate))
        network = unet(tau=tau, dropout=dropout_rate)
        model_checkpoint = ModelCheckpoint('./CPT/MC_{}_{}.hdf5'.format(dropout_rate,tau), monitor='loss', verbose=1, save_best_only=True)
        # if os.path.exists('./CPT/MC_{}_{}.hdf5'.format(dropout_rate,tau)):
        #     network.load_weights('./CPT/MC_{}_{}.hdf5'.format(dropout_rate,tau))
        #     print("checkpoint_loaded")
        history = network.fit_generator(generate_arrats_from_memory(X_train_normalized, y_train, 2),
                                        steps_per_epoch=y_train.shape[0] / 20, epochs=100, verbose=1,
                                        callbacks=[model_checkpoint])
        # network.save('./MC_{}_{}.hdf5'.format(dropout_rate,tau))
        np.savetxt('./CPT/MC_accuracy_{}_{}.txt'.format(dropout_rate,tau), network.history.history['accuracy'])
        np.savetxt('./CPT/MC_loss_{}_{}.txt'.format(dropout_rate,tau), network.history.history['loss'])

        ##we obtain the test RMSE and the test LL from the validation sets
        error, MC_error, ll, uncertainty, accuracy ,MC_pred= predict(network, mean_X_train, std_X_train, X_validation,
                                                             y_validation,dropout=dropout_rate, tau=tau)


        if (ll > best_ll):
            best_ll = ll
            best_network = network
            best_tau = tau
            best_dropout = dropout_rate
            print('Best log_likelihood changed to: ' + str(best_ll))
            print('Best tau changed to: ' + str(best_tau))
            print('Best dropout rate changed to: ' + str(best_dropout))

        ##storing the validation results

        with open('./results/_RESULTS_VALIDATION_RMSE.txt', 'a') as myfile:
            myfile.write('Dropout_Rate:' + repr(dropout_rate) + 'Tau' + repr(tau) + "::")
            myfile.write(repr(error) + '\n')

        with open('./results/_RESULTS_VALIDATION_MC_RMSE.txt', "a") as myfile:
            myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
            myfile.write(repr(MC_error) + '\n')

        with open('./results/_RESULTS_VALIDATION_LL.txt', "a") as myfile:
            myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
            myfile.write(repr(ll) + '\n')
        with open('./results/_RESULTS_VALIDATION_uncertainty.txt', "a") as myfile:
            myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
            myfile.write(repr(uncertainty) + '\n')
        with open('./results/_RESULTS_VALIDATION_accuracy.txt', "a") as myfile:
            myfile.write('Dropout_Rate: ' + repr(dropout_rate) + ' Tau: ' + repr(tau) + ' :: ')
            myfile.write(repr(accuracy) + '\n')

# if os.path.exists('./bestwork.hdf5'):
#     os.remove('./bestwork.hdf5')
# best_network.save('./bestwork.hdf5')

##storing the best results
# error, MC_error, ll, uncertainty, accuracy, MC_pred = predict(best_network, mean_X_train, std_X_train, X_test, y_test, dropout=best_dropout,tau=best_tau)

with open(_RESULTS_TEST_RMSE, 'a') as myfile:
    myfile.write('dropout_rate:{},tau:{},'.format(best_dropout, best_tau) + repr(error) + '\n')
with open(_RESULTS_TEST_MC_RMSE, "a") as myfile:
    myfile.write('dropout_rate:{},tau:{},'.format(best_dropout, best_tau) + repr(MC_error) + '\n')
with open(_RESULTS_TEST_LL, "a") as myfile:
    myfile.write('dropout_rate:{},tau:{},'.format(best_dropout, best_tau) + repr(ll) + '\n')
with open(_RESULTS_TEST_TAU, "a") as myfile:
        myfile.write('dropout_rate:{},tau:{}m'.format(best_dropout,best_tau)+repr(best_tau) + '\n')
with open(_RESULTS_TEST_ACCURACY, "a") as myfile:
    myfile.write('dropout_rate:{},tau:{},'.format(best_dropout, best_tau) + repr(accuracy) + '\n')
with open(_RESULTS_TEST_UNCERTAINTY, "a") as myfile:
    myfile.write('dropout_rate:{},tau:{},'.format(best_dropout, best_tau) + repr(uncertainty) + '\n')

errors += [error]
MC_errors += [MC_error]
lls += [ll]
accuracys += [accuracy]
uncertaintys += [uncertainty]
cost_time = time.time() - start
print("{} time in total".format(cost_time))

with open(_RESULTS_TEST_LOG, "a") as myfile:
    myfile.write('\n')
    myfile.write('dropout_rate:{},best_tau:{},time cost: {}\n'.format(best_dropout, best_tau, cost_time))
    myfile.write('errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(errors), np.std(errors), np.std(errors) / math.sqrt(splits),
        np.percentile(errors, 50), np.percentile(errors, 25), np.percentile(errors, 75)))
    myfile.write('MC errors %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(MC_errors), np.std(MC_errors), np.std(MC_errors) / math.sqrt(splits),
        np.percentile(MC_errors, 50), np.percentile(MC_errors, 25), np.percentile(MC_errors, 75)))
    myfile.write('lls %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(lls), np.std(lls), np.std(lls) / math.sqrt(splits),
        np.percentile(lls, 50), np.percentile(lls, 25), np.percentile(lls, 75)))
    myfile.write('accuracys %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(accuracys), np.std(accuracys), np.std(accuracys) / math.sqrt(splits),
        np.percentile(accuracys, 50), np.percentile(accuracys, 25), np.percentile(accuracys, 75)))
    myfile.write('uncertaintys %f +- %f (stddev) +- %f (std error), median %f 25p %f 75p %f \n' % (
        np.mean(uncertaintys), np.std(uncertaintys), np.std(uncertaintys) / math.sqrt(splits),
        np.percentile(uncertaintys, 50), np.percentile(uncertaintys, 25), np.percentile(uncertaintys, 75)))
