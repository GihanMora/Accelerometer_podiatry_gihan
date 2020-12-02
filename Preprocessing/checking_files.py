import pandas as pd
import os
import datetime
import time

start = time.time()
file_path = 'E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/1/MOS4B21140956 (2015-10-23)RAW.csv'
root_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/'
initial_times = pd.DataFrame()
f_name_list = []
init_dt_list = []
for i in range(1,24):
    if(i==4):continue
    f_path = os.path.join(root_folder,str(i))
    file_list = os.listdir(f_path)
    raw_file_path = ''
    for each_p in file_list:
        if(')RAW.csv' in each_p):
            raw_file_path = os.path.join(root_folder,str(i),each_p)
    print(raw_file_path)

    f = open(raw_file_path,'r')
    file_name = os.path.basename(f.name)
    lines = f.readlines()
    s_t = [int(i) for i in lines[2].strip().split('Start Time ')[1].split(':')]
    s_d = [int(i) for i in lines[3].strip().split('Start Date ')[1].split('/')]
    dt  = datetime.datetime(s_d[2], s_d[1], s_d[0], s_t[0],s_t[1],s_t[2])
    f.close()
    f_name_list.append(file_name)
    init_dt_list.append(str(dt))
    print(file_name,dt)



initial_times['f_name'] = f_name_list
initial_times['initial_date'] = init_dt_list

initial_times.to_csv('E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/metadata/initial_times.csv')
#
# raw_df_orig = pd.read_csv(file_path, skiprows=11, header=None)
# print(raw_df_orig.head())
# print(len(raw_df_orig))

# step = datetime.timedelta(seconds=0.01)
# time_stamps = []
# for t in range(len(raw_df_orig)):
#     time_stamps.append(dt.strftime('%Y-%m-%d %H:%M:%S:%f'))
#     dt += step
#
# raw_df_orig['Time_stamp']=time_stamps
#
# raw_df_orig.to_csv("E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/processed_raw_files/"+file_name)
stop = time.time()
duration = stop-start
print(duration)

# csv_f = pd.read_csv(file_path)

