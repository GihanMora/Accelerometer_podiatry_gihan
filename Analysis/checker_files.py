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