import pandas as pd
from tqdm import tqdm
from dateutil import parser
import os
from datetime import datetime


root_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/'
data_dict = {}
for i in range(1,24):
    if(i==4):continue
    f_path = os.path.join(root_folder,str(i))
    file_list = os.listdir(f_path)
    raw_file_path = ''

    for each_p in file_list:
        if(')RAW.csv' in each_p):
            raw_file_path = os.path.join(root_folder,str(i),each_p)
            sleep_f_path = os.path.join(root_folder,str(i),'sleep.csv')
            data_dict[each_p]=[raw_file_path,sleep_f_path]
    # print(raw_file_path)

print(data_dict)
initia_time = pd.read_csv('E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/metadata/initial_times.csv')
for _, row in tqdm(initia_time.iterrows(), total=initia_time.shape[0]):
    data_dict[row['f_name']].extend([row['initial_date']])


print(data_dict)

for each_participant in list(data_dict.keys())[16:]:
    raw_f = data_dict[each_participant][0]
    sleep_f = data_dict[each_participant][1]
    # if(sleep_f=='E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/18\\sleep.csv'):continue
    initial_dt = parser.parse(data_dict[each_participant][2])
# file_path = "E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/1/sleep.csv"
    sleep_df = pd.read_csv(sleep_f, skiprows=5)
#
#
# initial_dt = datetime(2015, 8, 7, 7,30,00)
    end_times = []
    begin_times = [initial_dt]

    for _, row in tqdm(sleep_df.iterrows(), total=sleep_df.shape[0]):

        end_d = row['In Bed Date']
        begin_d = row['Out Bed Date']
        end_d = ('-').join(end_d.split('/')[::-1])
        begin_d = ('-').join(begin_d.split('/')[::-1])
        end_dt = end_d+" "+row['In Bed Time']
        begin_dt = begin_d + " " + row['Out Bed Time']
        end_dt = parser.parse(end_dt)
        begin_dt = parser.parse(begin_dt)
        end_times.append(end_dt)
        begin_times.append(begin_dt)

    print(begin_times)
    print(end_times)


    for i in range(len(begin_times)-1):
        print(begin_times[i],end_times[i])
        starting_row = int(((begin_times[i] - begin_times[0]).total_seconds()) * 100)
        number_of_rows = int(((end_times[i] - begin_times[i]).total_seconds()) * 100)
        # print((begin_times[i] - begin_times[0]).total_seconds()/3600)
        # print((end_times[i] - begin_times[i]).total_seconds()/3600)
        print('starting_row',starting_row)
        print('row_len',number_of_rows)
        #
        # ff = 'E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/1/MOS4B21140956 (2015-10-23)RAW.csv'
        raw_df = pd.read_csv(raw_f, skiprows=11 + int(starting_row), nrows=number_of_rows, header=None)
        out_folder = "E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/processed_raw_files/"+each_participant.replace('.csv','')
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_name = (out_folder+'/'+each_participant.replace('.csv','_')+str(begin_times[i]).replace(':','-')+'_to_'+str(end_times[i]).replace(':','-')+'.csv')
        raw_df.to_csv(out_name)
        print("***")






# begin = parser.parse(sleep_f['begin_time'][0])