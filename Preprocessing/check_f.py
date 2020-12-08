import pandas as pd
file_path = "E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/1/sleep_copy.csv"
# f = pd.read_csv(file_path, skiprows=11, header=None)
#
#
import time
import datetime
# sleep_f = pd.read_csv(file_path, skiprows=5)
# # print(sleep_f['begin_time'])
# from dateutil import parser
# begin = parser.parse(sleep_f['begin_time'][0])
# end = parser.parse(sleep_f['end_time'][1])
initial = datetime.datetime(2015, 8, 7, 14,30,00)

# starting_row = ((begin-initial).total_seconds())*100
# number_of_rows = ((end-begin).total_seconds())*100
# print('s',starting_row)
# print('n',number_of_rows)
# ff  = 'E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/1/MOS4B21140956 (2015-10-23)RAW.csv'
# f = pd.read_csv(ff, skiprows=11+int(starting_row),nrows=number_of_rows, header=None)
# f.to_csv("E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/processed_raw_files/pp.csv")
# print((dt-dt_1).total_seconds())
dt = initial
step = datetime.timedelta(seconds=60)
time_stamps = []
for t in range(1000):
    time_stamps.append(dt.strftime('%Y-%m-%d %H:%M:%S:%f'))
    dt += step
print(time_stamps)