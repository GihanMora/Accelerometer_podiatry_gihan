import pandas as pd
from tqdm import tqdm
import datetime
from dateutil import parser
import os

root_folder = 'E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/'
pred_root = 'E:/Data/Accelerometer_Dataset_Rashmika/Podiatry Data/Predictions/'
data_dict = {}

for i in range(1,24):
    if(i==4):continue
    f_path = os.path.join(root_folder,str(i))
    file_list = os.listdir(f_path)
    d_file_path = ''
    if (i < 10):
        st = 'Participant_00' + str(i)
    else:
        st = 'Participant_0' + str(i)

    for each_p in file_list:
        if('60sec AGD Details Epochs' in each_p):
            d_file_path = os.path.join(root_folder,str(i),each_p)
            pred_f_path = os.path.join(pred_root,st,'predictions.csv')
            data_dict[i]=[d_file_path,pred_f_path]
    # print(raw_file_path)

print(data_dict)


for i in range(1,24):
    if(i in [4,13,14,15]):continue
    try:
        from_device = data_dict[i][0]
        from_pred = data_dict[i][1]
        out_file = from_pred.replace('predictions.csv','predictions_merged.csv')
        print(from_device)
        print(from_pred)
        print(out_file)
        # print(len(data_dict[i]))

        pd_device = pd.read_csv(from_device, skiprows=1)
        print(pd_device.columns)
        pd_pred = pd.read_csv(from_pred)
        pd_device_fixed = pd_device[[ 'axis1', 'axis2', 'axis3', 'vm', 'steps', 'lux',
               'inclinometer off', 'inclinometer standing', 'inclinometer sitting',
               'inclinometer lying', 'kcals', 'MET rate']]

        Timestamps = []
        for _, row in tqdm(pd_device.iterrows(), total=pd_device.shape[0]):
            s_d = row['date'].split('/')
            s_d_n = s_d[2]+'/'+s_d[1]+"/"+s_d[0]
            # print(s_d_n)
            s_t = row['epoch']
            end_dt = s_d_n + " " + row['epoch']
            # print(end_dt)
            end_dt = parser.parse(end_dt)

            # print(end_dt)
            Timestamps.append(str(end_dt))
            # dt = datetime.datetime(int(s_d[0]), int(s_d[1]), int(s_d[2]), int(s_t[0]), int(s_t[1]), int(s_t[2]))
            # break
        # pd_device_fixed['Timestamp'] = Timestamps
        pd_device_fixed.insert(0,'Timestamp',Timestamps)


        print(pd_device_fixed.head())

        pd_pred_fixed = pd_pred[['Timestamp','EE','PA_intensitivity','PA_Category']]
        pd_pred_fixed.columns = ['Timestamp','CNN_EE','CNN_PA_intensitivity','CNN_PA_Category' ]
        df3 = pd_device_fixed.merge(pd_pred_fixed, on=['Timestamp'], how='left')
        # df3.fillna()
        df3.to_csv(out_file)
    except SyntaxError:
        print()
