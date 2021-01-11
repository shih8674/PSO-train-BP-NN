# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:50:11 2021

@author: ray
"""

import numpy as np
import pandas as pd
result_all = pd.DataFrame()
for month_num in range(1,13):
    df_data = pd.read_csv('./time/time_'+ str(month_num) +'.csv', index_col= False)
    new_frame = pd.read_csv('train_frame.csv')
    # df_data_drop = df_data[(True-df_data.isin(['x']))]
    result = pd.DataFrame()
    
    days = int(df_data.shape[0]/11)
    # for num, col in enumerate(df_data.iloc[:,6]):
    data_temp = pd.DataFrame()
    AQI_result = np.zeros(shape=[5, days])
    
    for day_num in range(days):
        data_temp = df_data.iloc[day_num*11:(day_num*11)+11, :].reset_index()
        ItemId = data_temp["ItemId"]
        for Id_num in range(11):
            if(ItemId[Id_num]==3):
                AQI_O3 = 0
                mean_O3 = 0
                O3 = data_temp.iloc[Id_num, 8:].astype(float)
                mean_O3 = O3.mean()
                if(mean_O3 < 125):
                    AQI_O3 = 0
                if(mean_O3 >= 125 and mean_O3 <= 164):
                    AQI_O3 = (mean_O3-125)*(49/39)+101
                if(mean_O3 > 164):
                    AQI_O3 = (mean_O3-165)*(49/39)+151
                # print('AQI_O3', AQI_O3)
                
            if(ItemId[Id_num]==33):
                AQI_PM25 = 0
                mean_PM25 = 0
                PM25 = data_temp.iloc[Id_num,8:].astype(float)
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
                            
            if(ItemId[Id_num]==4):
                AQI_PM10 = 0
                mean_PM10 = 0
                PM10 = data_temp.iloc[Id_num, 8:].astype(float)
                mean_PM10 = PM10.mean()
                if(mean_PM10 < 54):
                    AQI_PM10 = (mean_PM10)*(50/54)
                if(mean_PM10 >=54 and mean_PM10 <125):
                    AQI_PM10 = (mean_PM25-54)*(49/70)+51
                if(mean_PM10 >=125 and mean_PM10 <254):
                    AQI_PM10 = (mean_PM25-125)*(49/128)+101
                if(mean_PM10 >=254):
                    AQI_PM10 = (mean_PM25-254)*(49/99)+151
                # print('AQI_PM10', AQI_PM10)
                            
            if(ItemId[Id_num]==2):
                AQI_CO = 0
                mean_CO = 0
                CO = data_temp.iloc[Id_num, 8:].astype(float)
                mean_CO = CO.mean()
                if(mean_CO < 4.4):
                    AQI_CO = (mean_CO)*(50/4.4)
                if(mean_CO >=4.4 and mean_CO <9.4):
                    AQI_CO = (mean_CO-4.4)*(49/4.9)+51
                if(mean_CO >=9.4 and mean_CO <12.4):
                    AQI_CO = (mean_CO-9.4)*(49/2.9)+101
                if(mean_CO >=12.4):
                    AQI_CO = (mean_CO-12.4)*(49/2.9)+151     
                # print('AQI_CO', AQI_CO)
                
            if(ItemId[Id_num]==1):
                AQI_SO2 = 0
                mean_SO2 = 0
                SO2 = data_temp.iloc[Id_num, 8:].astype(float)
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
            
            # if(ItemId[Id_num]==144):
            #     wind_direct = data_temp.iloc[Id_num, 8:].astype(float)
            #     mean_wind_direct = wind_direct.mean()
                
            # if(ItemId[Id_num]==14):
            #     temper = data_temp.iloc[Id_num, 8:].astype(float)
            #     mean_temper = temper.mean()
                
            # if(ItemId[Id_num]==23):
            #     rain = data_temp.iloc[Id_num, 8:].astype(float)
            #     mean_rain = rain.sum()
                
            # if(ItemId[Id_num]==38):
            #     humidity = data_temp.iloc[Id_num, 8:].astype(float)
            #     mean_humidity = humidity.mean()
            
            # if(ItemId[Id_num]==144):
            #     wind_speed = data_temp.iloc[Id_num, 8:].astype(float)
            #     mean_wind_speed = wind_speed.mean()
            
        temp_AQI = np.max([AQI_CO, AQI_O3, AQI_PM10, AQI_PM25, AQI_SO2])
        data_temp_result = np.array([mean_CO, mean_PM10, mean_PM25, mean_SO2, temp_AQI])
        AQI_result[:, day_num] = data_temp_result.T
        new_frame.iloc[0, 2:7] = data_temp_result
        new_frame.iloc[0, 0] = data_temp.iloc[0, 7]
        result = pd.concat([result, new_frame], axis=0)
    
    result.to_csv('./time_train/time_'+ str(month_num) +'.csv', index = False)
    result_all = pd.concat([result_all, result], axis=0)

result_all.to_csv('./time_train/origin_train_only6AQI.csv', index = False)

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