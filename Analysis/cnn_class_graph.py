from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import listdir, makedirs
from os.path import join, isfile, exists
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import shutil
from tqdm import tqdm
from time import time
import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Reshape, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras
# import np_utils
from tensorflow.keras.callbacks import TensorBoard

from sklearn.metrics import confusion_matrix
import statistical_extensions as SE

# pd.options.display.float_format = '{:.1f}'.format
# sns.set()  # Default seaborn look and feel
# plt.style.use('ggplot')

training_dataset_path = 'E:/Data/Accelerometer_Dataset_Rashmika/OA_data/supervised_data/ActiGraph/numpy_window-3600-overlap-0_train/'
test_dataset_path = 'E:/Data/Accelerometer_Dataset_Rashmika/OA_data/supervised_data/ActiGraph/numpy_window-3600-overlap-0_test/'
temp_model_out_folder = 'E:/Projects/Accelerometer_OA_gihan/CNN_ACCL_OA/Model_outputs/CNN_graph_clas/temp_model_out/'
MODEL_FOLDER = 'E:/Projects/Accelerometer_OA_gihan/CNN_ACCL_OA/Model_outputs/CNN_graph_clas/'
TIME_PERIODS = 3600
model_checkpoint_path = 'E:/Projects/Accelerometer_OA_gihan/CNN_ACCL_OA/Model_outputs/CNN_graph_clas/temp_model_out/'



def load_data(filenames):

    X_data = []
    Y_data = []
    ID_user = []
    counter = 0
    for filename in tqdm(filenames):
        npy = np.load(filename, allow_pickle=True)
        X_data.append(npy.item().get('segments'))
        Y_data.append(npy.item().get('activity_classes'))

        user_id = filename.split('/')[-1][:6]
        data_length = npy.item().get('activity_classes').shape[0]
        ID_user.extend([user_id for _ in range(data_length)])

        # counter += 1
        # if counter > 10:
        #     break

    X_data = np.concatenate(X_data, axis=0)
    Y_data = np.concatenate(Y_data, axis=0)
    # print('yydata', Y_data)
    # Data relabeling from index 0 (use only 3 classes)
    Y_data = np.where(Y_data == 1, 0, Y_data)
    Y_data = np.where(Y_data == 2, 1, Y_data)
    Y_data = np.where(Y_data == 3, 2, Y_data)
    Y_data = np.where(Y_data == 4, 2, Y_data)
    # print('yydata',Y_data)
    return X_data, Y_data, ID_user


def plot_model(history, MODEL_FOLDER):
    # summarize history for accuracy and loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['acc'], "g--", label="Accuracy of training data")
    plt.plot(history.history['val_acc'], "g", label="Accuracy of validation data")
    plt.plot(history.history['loss'], "r--", label="Loss of training data")
    plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.savefig(MODEL_FOLDER + 'learning_history.png')
    plt.clf()
    plt.close()

# training_data_files = [join(training_dataset_path, f) for f in listdir(training_dataset_path) if isfile(join(training_dataset_path, f))]
#
# print(training_data_files)
# train_X_data, train_Y_data, train_ID_user = load_data(training_data_files)
# X_train, y_train, ID_train = train_X_data, train_Y_data, train_ID_user
# # # Data -> Model ready
# # print('xxx',X_train)
# # print('yyy',y_train[0])
# # print('first',(X_train.shape))
# num_time_periods, num_sensors = X_train.shape[1], X_train.shape[2]
# num_classes = len(np.unique(y_train))
# # Set input_shape / reshape for Keras
# input_shape = (num_time_periods * num_sensors)
# print('num_classes',num_classes)
# print('input shape',input_shape)
# X_train = X_train.reshape(X_train.shape[0], input_shape)
#
# X_train = X_train.astype("float32")
# y_train = y_train.astype("float32")
#
# # One-hot encoding of y_train labels (only execute once!)
# # y_train = np_utils.to_categorical(y_train, num_classes)
# y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
# # print('onehot',y_train)
#
# print('nummm',num_classes)
# """Model architecture"""
# model_m = Sequential()
# model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
# model_m.add(Conv1D(80, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
# model_m.add(Conv1D(100, 10, activation='relu'))
# model_m.add(MaxPooling1D(3))
# model_m.add(Conv1D(160, 10, activation='relu'))
# model_m.add(Conv1D(180, 10, activation='relu'))
# model_m.add(MaxPooling1D(3))
# model_m.add(Conv1D(220, 10, activation='relu'))
# model_m.add(Conv1D(240, 10, activation='relu'))
# model_m.add(GlobalMaxPooling1D())
# model_m.add(Dropout(0.5))
# model_m.add(Dense(num_classes, activation='softmax'))
#
# callbacks_list = [
#     ModelCheckpoint(
#         filepath=model_checkpoint_path+'/best_model.{epoch:03d}-{val_loss:.2f}.h5',
#         monitor='val_loss', save_best_only=True),
#     TensorBoard(log_dir='logs\\{}'.format(time())),
#     EarlyStopping(monitor='val_loss', patience=10)
# ]
#
# model_m.compile(loss='categorical_crossentropy',
#                 optimizer='adam',
#                 metrics=['accuracy'])
#
# # Hyper-parameters
# BATCH_SIZE = 32
# EPOCHS = 10
#
# history = model_m.fit(X_train,
#                       y_train,
#                       batch_size=BATCH_SIZE,
#                       epochs=EPOCHS,
#                       callbacks=callbacks_list,
#                       validation_split=0.2,
#                       verbose=2)

# plot_model(history, MODEL_FOLDER)

num_classes = 3
input_shape = 10800
model_m =  load_model(join(temp_model_out_folder, 'best_model.008-0.11.h5'))

test_data_files = [join(test_dataset_path, f) for f in listdir(test_dataset_path) if isfile(join(test_dataset_path, f))]
print(test_data_files)
test_X_data, test_Y_data, test_ID_user = load_data(test_data_files)
test_X_data = test_X_data.reshape(test_X_data.shape[0], input_shape).astype("float32")
test_Y_data = test_Y_data.astype("float32")
test_Y_data = tensorflow.keras.utils.to_categorical(test_Y_data, num_classes)
y_pred_test = model_m.predict(test_X_data)
print('output',y_pred_test)

# res_csv = pd.DataFrame()
# actual_1d_list = [list(i)[0] for i in list(test_Y_data)]
# pres_1d_list = [list(i)[0] for i in list(y_pred_test)]
# print('accc',actual_1d_list)
# print('preeed',pres_1d_list)
# res_csv['actual'] = actual_1d_list
# res_csv['pred'] = pres_1d_list
# res_csv.to_csv(join(MODEL_FOLDER, 'actual_vs_predicted_class.csv'))


# plt.figure(figsize=(8, 8))
# plt.scatter(test_Y_data, y_pred_test)
# plt.xlabel('Actual EE')
# plt.ylabel('Predicted EE')
# plt.savefig(join(MODEL_FOLDER, 'actual_vs_predicted_class.png'))
# plt.clf()
# plt.close()


grp_results = []
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(test_Y_data, axis=1)
print('pred_m',max_y_pred_test)
print('act_m',max_y_test)
assert test_Y_data.shape[0] == y_pred_test.shape[0]


# Evaluation matrices

class_names = ['SED', 'LPA', 'MVPA']
cnf_matrix = confusion_matrix(max_y_test, max_y_pred_test)

stats = SE.GeneralStats.evaluation_statistics(cnf_matrix)

assessment_result = 'Classes' + '\t' + str(class_names) + '\t' + '\n'
assessment_result += 'Accuracy' + '\t' + str(stats['accuracy']) + '\t' + str(stats['accuracy_ci']) + '\n'
assessment_result += 'Sensitivity' + '\t' + str(stats['sensitivity']) + '\n'
assessment_result += 'Sensitivity CI' + '\t' + str(stats['sensitivity_ci']) + '\n'
assessment_result += 'Specificity' + '\t' + str(stats['specificity']) + '\n'
assessment_result += 'Specificity CI' + '\t' + str(stats['specificity_ci']) + '\n'

grp_results.append(assessment_result)

SE.GeneralStats.plot_confusion_matrix(cnf_matrix, classes=class_names, title='CM',
                                      output_filename=join(MODEL_FOLDER,  'confusion_matrix.png'))

result_string = '\n'.join(grp_results)
with open(join(MODEL_FOLDER,  'result_report.txt'), "w") as text_file:
    text_file.write(result_string)

