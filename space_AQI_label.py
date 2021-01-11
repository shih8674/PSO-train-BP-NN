# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:50:11 2021

@author: ray
"""

import numpy as np
import pandas as pd
result_all = pd.DataFrame()
for month_num in range(1,13):
    df_data = pd.read_csv('./space/space_'+ str(month_num) +'.csv', index_col= False)
    new_frame = pd.read_csv('train_space_frame.csv')
    # df_data_drop = df_data[(True-df_data.isin(['x']))]
    result = pd.DataFrame()
    
    days = int(df_data.shape[0]/132)
    # for num, col in enumerate(df_data.iloc[:,6]):
    data_temp = pd.DataFrame()
    # AQI_result = np.zeros(shape=[5, days])
    
    for day_num in range(days):
        
        data_temp = df_data.iloc[day_num * 132:(day_num * 132)+132, :].reset_index()
        space_ItemId = data_temp["ItemId"]

        for space_num in range(12):
            ItemId = space_ItemId[space_num * 11 : (space_num * 11)+11]
            data_space_temp = data_temp.iloc[space_num * 11 : (space_num * 11)+11].reset_index()
            for Id_num in range(11):
                # print(ItemId[Id_num])
                if(ItemId.iloc[Id_num]==3):
                    # print('ok')
                    AQI_O3 = 0
                    mean_O3 = 0
                    O3 = np.float64(data_space_temp.iloc[Id_num, 9:])
                    mean_O3 = O3.mean()
                    if(mean_O3 < 125):
                        AQI_O3 = 0
                    if(mean_O3 >= 125 and mean_O3 <= 164):
                        AQI_O3 = (mean_O3-125)*(49/39)+101
                    if(mean_O3 > 164):
                        AQI_O3 = (mean_O3-165)*(49/39)+151
                    # print('AQI_O3', AQI_O3)
                    
                if(ItemId.iloc[Id_num]==33):
                    AQI_PM25 = 0
                    mean_PM25 = 0
                    PM25 = np.float64(data_space_temp.iloc[Id_num, 9:])
                    mean_PM25 = PM25.mean()
                    if(mean_PM25 < 15.5):
                        AQI_PM25 = (mean_PM25)*(50/15.4)
                    if(mean_PM25 >=15.5 and mean_PM25 <35.4):
                        AQI_PM25 = (mean_PM25-15.5)*(49/18.9)+51
                    if(mean_PM25 >=35.4 and mean_PM25 <54.4):
                        AQI_PM25 = (mean_PM25-35.5)*(49/18.9)+101
                    if(mean_PM25 >=54.4):
                        AQI_PM25 = (mean_PM25-54.5)*(49/95.9)+151
                    # print('AQI_PM25', AQI_PM25)
                                
                if(ItemId.iloc[Id_num]==4):
                    AQI_PM10 = 0
                    mean_PM10 = 0
                    PM10 = np.float64(data_space_temp.iloc[Id_num, 9:])
                    mean_PM10 = PM10.mean()
                    if(mean_PM10 < 54):
                        AQI_PM10 = (mean_PM10)*(50/54)
                    if(mean_PM10 >=54 and mean_PM10 <125):
                        AQI_PM10 = (mean_PM25-54)*(49/70)+51
                    if(mean_PM10 >=125 and mean_PM10 <254):
                        AQI_PM10 = (mean_PM25-125)*(49/128)+101
                    if(mean_PM10 >=254):
                        AQI_PM10 = (mean_PM25-254)*(49/99)+151
                    # print('AQI_PM10', AQI_PM10, 'mean_PM10', mean_PM10)
                                
                if(ItemId.iloc[Id_num]==2):
                    AQI_CO = 0
                    mean_CO = 0
                    CO = np.float64(data_space_temp.iloc[Id_num, 9:])
                    mean_CO = CO.mean()
                    if(mean_CO < 4.4):
                        AQI_CO = (mean_CO)*(50/4.4)
                    if(mean_CO >=4.4 and mean_CO <9.4):
                        AQI_CO = (mean_CO-4.4)*(49/4.9)+51
                    if(mean_CO >=9.4 and mean_CO <12.4):
                        AQI_CO = (mean_CO-9.4)*(49/2.9)+101
                    if(mean_CO >=12.4):
                        AQI_CO = (mean_CO-12.4)*(49/2.9)+151     
                    # print('AQI_CO', AQI_CO, 'mean CO', mean_CO)
                    
                if(ItemId.iloc[Id_num]==1):
                    AQI_SO2 = 0
                    mean_SO2 = 0
                    SO2 = np.float64(data_space_temp.iloc[Id_num, 9:])
                    # print(SO2)
                    mean_SO2 = SO2.mean()
                    if(mean_SO2 < 35):
                        AQI_SO2 = (mean_SO2)*(50/35)
                    if(mean_SO2 >=35 and mean_SO2 <75):
                        AQI_SO2 = (mean_SO2-35)*(49/39)+51
                    if(mean_SO2 >=75 and mean_SO2 <185):
                        AQI_SO2 = (mean_SO2-75)*(49/109)+101
                    if(mean_SO2 >=185):
                        AQI_SO2 = (mean_SO2-185)*(49/118)+151
                    # print('AQI_SO2', AQI_SO2)
                
            AQI_temp = np.max([AQI_CO, AQI_O3, AQI_PM10, AQI_PM25, AQI_SO2])
            new_frame.iloc[0, (space_num + 1)] = AQI_temp
            # print('day num: ',day_num, 'space_num ',space_num)
            # print('new_frame ',new_frame.iloc[0, (space_num + 1)])
            # print('--------------------------------------------')
            new_frame.iloc[0, 0] = data_space_temp.iloc[0, 8]
            
        result = pd.concat([result, new_frame], axis=0)
        new_frame.iloc[0,:] = 0
    result.to_csv('./space_train/space_'+ str(month_num) +'.csv', index = False)
    result_all = pd.concat([result_all, result], axis=0)

result_all.to_csv('./space_train/train_space_AQI.csv', index = False)

# frame_3day = pd.read_csv('train_frame_3day.csv')
# result_train = pd.DataFrame()

# for train_num in range(363):
#     # col1 = pd.DataFrame(result_all.iloc[train_num, 2:8]).T
#     # col2 = pd.DataFrame(result_all.iloc[train_num+1, 2:8]).T
#     # col3 = pd.DataFrame(result_all.iloc[train_num+2, 2:8]).T
#     # label =  pd.DataFrame(result_all.iloc[train_num+3, 7])
    
#     # result_train_temp = pd.concat([col1, col2, col3, label], axis=1)
#     # result_train = pd.concat([result_train, result_train_temp], axis=0)
    
#     frame_3day.iloc[0, :6] = result_all.iloc[train_num, 2:8].values
#     frame_3day.iloc[0, 6:12] = result_all.iloc[train_num + 1, 2:8].values
#     frame_3day.iloc[0, 12:18]= result_all.iloc[train_num + 2, 2:8].values
#     frame_3day.iloc[0, 18] = result_all.iloc[train_num + 3, 7]
#     result_train =  pd.concat([result_train, frame_3day], axis=0)

# result_train.to_csv('./time_train/train_3day.csv', index = False)