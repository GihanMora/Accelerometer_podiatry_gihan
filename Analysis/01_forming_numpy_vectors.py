import pandas as pd
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
import pickle


def create_segments_and_labels(dataframe, time_steps, step, n_features, label_class, label_2):

    segments = []
    labels = []
    regression_values = []
    for i in range(0, len(dataframe) - time_steps, step):
        xs = dataframe['0'].values[i: i + time_steps]
        ys = dataframe['1'].values[i: i + time_steps]
        zs = dataframe['2'].values[i: i + time_steps]
        # print(xs)
        # print(ys)
        # print(zs)
        # Retrieve the most often used label in this segment
        class_label = dataframe[label_class][i: i + time_steps].mode()[0]
        class_reg = dataframe[label_2][i: i + time_steps].mean()
        segments.append([xs, ys, zs])
        # print(segments)
        labels.append(class_label)
        regression_values.append(class_reg)


    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, time_steps, n_features)
    # print(reshaped_segments)
    labels = np.asarray(labels)
    regression_values = np.asarray(regression_values)

    return {'segments': reshaped_segments, 'activity_classes': labels, 'energy_e': regression_values}


time_window = 6000
N_FEATURES = 3
LABEL_CLASS = 'waist_intensity_ee_based'
LABEL_REG = 'waist_ee'
req_cols = ['0', '1', '2']
input_cols = ['0', '1', '2']
target_cols = ['waist_ee', 'waist_intensity_ee_based']
participants = []
ids = [2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
for i in ids:
    if(i<10):
        st = 'Participant_00'+str(i)
    else:
        st = 'Participant_0' + str(i)
    participants.append(st)
print(participants)
for p in participants:

    INPUT_DATA_FOLDER = 'E:\Data\Accelerometer_Dataset_Rashmika\Podiatry Data\processed_raw_files\\'+p+'\\'
    OUTPUT_FOLDER_ROOT = 'E:\Data\Accelerometer_Dataset_Rashmika\Podiatry Data\\numpy_arrays_test\\'+p+'\\'
    if not exists(OUTPUT_FOLDER_ROOT):
        makedirs(OUTPUT_FOLDER_ROOT)



    raw_files = [f for f in listdir(INPUT_DATA_FOLDER) if isfile(join(INPUT_DATA_FOLDER, f))]
    for f in tqdm(raw_files):
    #     #  84%|████████▎ | 705/842 [5:39:10<1:15:31, 33.08s/it]
        print(f)
        df = pd.read_csv(join(INPUT_DATA_FOLDER, f), usecols=req_cols)
        df['waist_intensity_ee_based'] =[1]*len(df)
        df['waist_ee'] =[1]*len(df)
        STEP_DISTANCE = time_window
        reshaped_outcomes = create_segments_and_labels(df, time_window, STEP_DISTANCE,
                                                   N_FEATURES, LABEL_CLASS, LABEL_REG)
        OUTPUT_FOLDER = join(OUTPUT_FOLDER_ROOT, 'numpy_window-{}-overlap-{}'.format(time_window, int(0)))
        if not exists(OUTPUT_FOLDER):
                            makedirs(OUTPUT_FOLDER)
        out_name = join(OUTPUT_FOLDER, f.replace('.csv', '_test.npy'))
        np.save(out_name, reshaped_outcomes)
