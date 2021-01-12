# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:47:08 2021

@author: ray
"""
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.seterr(divide='ignore', invalid='ignore')


input_data = pd.read_csv('./space_train/train_space_AQI_2.csv')

shape_num = input_data.shape
X_row = input_data.iloc[:, :shape_num[1]-1]
Y_row = input_data.iloc[:, shape_num[1]-1]

train_x, val_test_x, train_y, val_test_y = train_test_split(X_row, Y_row, test_size = 0.3, shuffle = True)
val_x, test_x, val_y, test_y = train_test_split(val_test_x, val_test_y, test_size = 0.5, shuffle = True)

def relu(x):
    if x < 0: return 0
    else: return x

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def plot_fit(val_output, val_y, epoch_num):
    
    plt.figure(figsize=(8.8,4))
    plt.plot(val_y)
    plt.plot(val_output)
    
    plt.title('Predict Result of epoch:' + str(epoch_num))
    plt.ylabel('value')
    plt.xlabel('number')
    plt.legend(['val_y', 'val_output'], loc='best')
      
    plt.show()
    
def NN(X, train_x, train_y, input_kernel, hid_kernel, output_kernel, epoch_num):
    fit_w = X[:input_kernel*hid_kernel]
    fit_wbias = X[input_kernel*hid_kernel: ((input_kernel*hid_kernel)+hid_kernel)]
    fit_v = X[((input_kernel*hid_kernel)+hid_kernel): (((input_kernel*hid_kernel)+hid_kernel)+hid_kernel)]
    fit_vbias = X[(((input_kernel*hid_kernel)+hid_kernel)+hid_kernel): ((((input_kernel*hid_kernel)+hid_kernel)+hid_kernel)+output_kernel)]
    
    data_result = np.zeros(train_x.shape[0])
    
    for input_num in range(train_x.shape[0]):
        input_x = np.zeros(input_kernel)
        for input_for_hid_num in range(hid_kernel):
            if(input_for_hid_num == 0):
                input_x = train_x.iloc[input_num].T
            else:
                input_x = np.hstack([input_x, train_x.iloc[input_num].T])
    
        hid_temp = fit_w * input_x
        hid_result = np.zeros(hid_kernel)
        for hid_num in range(hid_kernel):
            hid_result[hid_num] = relu(np.sum(hid_temp[hid_num * input_kernel : (hid_num * input_kernel) + input_kernel]) + fit_wbias[hid_num])
        

        output_temp = fit_v * hid_result        
        data_result[input_num] = np.sum(output_temp + fit_vbias)
        
    val_y = np.zeros(train_y.shape)
    val_y[:] = train_y[:]
    plot_fit(data_result, val_y, epoch_num)

        
def fitfunction(X, train_x, train_y, input_kernel, hid_kernel, output_kernel):
    
################ calculate NN structure #######################
    fit_w = X[:input_kernel*hid_kernel]
    fit_wbias = X[input_kernel*hid_kernel: ((input_kernel*hid_kernel)+hid_kernel)]
    fit_v = X[((input_kernel*hid_kernel)+hid_kernel): (((input_kernel*hid_kernel)+hid_kernel)+hid_kernel)]
    fit_vbias = X[(((input_kernel*hid_kernel)+hid_kernel)+hid_kernel): ((((input_kernel*hid_kernel)+hid_kernel)+hid_kernel)+output_kernel)]
    
    data_result = np.zeros(train_x.shape[0])
    
    for input_num in range(train_x.shape[0]):
        input_x = np.zeros(input_kernel)
        for input_for_hid_num in range(hid_kernel):
            if(input_for_hid_num == 0):
                input_x = train_x.iloc[input_num].T
            else:
                input_x = np.hstack([input_x, train_x.iloc[input_num].T])
    
        hid_temp = fit_w * input_x
        hid_result = np.zeros(hid_kernel)
        for hid_num in range(hid_kernel):
            hid_result[hid_num] = relu(np.sum(hid_temp[hid_num*input_kernel : (hid_num*input_kernel) + input_kernel]) + fit_wbias[hid_num])
        

        output_temp = fit_v * hid_result        
        data_result[input_num] = np.sum(output_temp + fit_vbias)
                
############## calculate fittness ######################
    val_y = np.zeros(train_y.shape)
    val_y[:] = train_y[:]


    fittness = np.mean(np.abs(val_y-data_result.flatten())/val_y)

    return fittness
    


 # PSO 參數
wmin = 0.02
wmax = 1       #20
c1min = 0.05
c1max = 1.2     # 500
c2min = 0.05
c2max = 1.2     #500


input_layer_kernel = shape_num[1]-1  # number of input layer kernel
hidden_layer_kernel = 6         # number of hidden layer kernel
output_layer_kernel = 1         # number of output layer kernel


dim = input_layer_kernel*hidden_layer_kernel + hidden_layer_kernel + hidden_layer_kernel*output_layer_kernel + output_layer_kernel   #維度
p = dim * 3      #粒子數量
iteration = 30   #迭代次數
X = np.zeros((p, dim))   #粒子位置
V = np.zeros((p, dim))   #粒子速度
pbest = np.zeros((p, dim, iteration+1))   #個體最佳解
gbest = np.zeros(dim)   #群體最佳解
pbest_fit = np.zeros((p, iteration+1))   #個體最佳適應值        
gbest_fit = 10**12     #群體最佳適應值
   
#初始化群體  
def init_Population(p, dim, X, V, pbest_fit, gbest_fit, gbest, pbest, input_num, hidden_num, output_num):

    for i in range(p):
        for j in range(dim):
            V[i,j] = random.uniform(-0.85, 0.85)
            X[i,j] = random.uniform(-0.85, 0.85)
            
        pbest[i, :, 0] = X[i, :]
        # print('particle num: '+str(i))
        pbest_fit[i, 0] = fitfunction(X[i, :], train_x, train_y, input_num, hidden_num, output_num)
        # print(pbest_fit)
        if(pbest_fit[i, 0] < gbest_fit):
            gbest_fit = pbest_fit[i, 0]
            gbest = pbest[i, :, 0]
            print('update gbest, gbest: ', gbest_fit)
        
    return p, dim, X, V, pbest_fit, gbest_fit, gbest, pbest

                
def iterator(wmax, wmin, c1max, c1min, c2max, c2min, p, dim, iteration, X, V, pbest, gbest, pbest_fit, gbest_fit, input_num, hidden_num, output_num):

    particle_bound = 2
    
    fitness = np.zeros(iteration+1)
    fitness[0]= gbest_fit
    for tt in range(iteration):
        t_start = time.time()


        
        w = wmin + (iteration-tt)/iteration*(wmax-wmin)
        c1 = c1min + (iteration-tt)/iteration*(c1max-c1min)
        c2 = c2max + (iteration-tt)/iteration*(c2min-c2max)
        

        
        
        for i in range(p):
            #速度及位置更新
            # print('particle_num: ', i)
            rand1 = random.random()
            rand2 = random.random()
            if (tt != 0):
                V[i] = w*V[i] + c1*rand1*(pbest[i,:,tt+1] - X[i, :]) + c2*rand2*(gbest - X[i, :])
            X[i, :] = X[i, :] + V[i]

            for col_num in range(dim):
                if(X[i, col_num] < -particle_bound or X[i, col_num] > particle_bound):
                    X[i, col_num] = random.uniform(-particle_bound, particle_bound)
                

#        for i in range(p):
            temp = fitfunction(X[i],  train_x, train_y, input_num, hidden_num, output_num)
            if(temp < pbest_fit[i, tt]):    #更新個體最佳及個體最佳適應值
                if(X[i, 0] >= -particle_bound and X[i, 0] <= particle_bound):
                    pbest[i,:,tt+1] = X[i, :]
                    pbest_fit[i, tt+1] = fitfunction(pbest[i,:,tt+1],  train_x, train_y, input_num, hidden_num, output_num)
            else:
                pbest_fit[i, tt+1] = pbest_fit[i, tt]
                pbest[i, :, tt+1] = pbest[i,:,tt]
            #### mutation #####
                mutation_rate = random.uniform(0,1)
                if(mutation_rate < 0.05):
                    X[i, :] = random.uniform(-0.85, 0.85)
                
            if(pbest_fit[i, tt+1] < gbest_fit):    #更新群體最佳及群體最佳適應值
                gbest = pbest[i, :, tt+1]
                gbest_fit = pbest_fit[i, tt+1]
            
        
            
                
                
        fitness[tt+1] = gbest_fit
        NN(gbest, train_x, train_y, input_num, hidden_num, output_num, tt)
        t_end = time.time()
        duration = t_end - t_start
        val_error = fitfunction(gbest, val_x, val_y, input_num, hidden_num, output_num)

        
        print('epoch_num: '+ str(tt) + ', gbest: ', gbest_fit, ', cost time：', duration, 'sec ,val_test mape: ', val_error)

       


    return fitness, gbest, pbest, pbest_fit
#執行
t_start_all = time.time()
p, dim, X, V, pbest_fit, gbest_fit, gbest, pbest = init_Population(p, dim, X, V, pbest_fit, gbest_fit, gbest, pbest, input_layer_kernel, hidden_layer_kernel, output_layer_kernel)
fitness, gbest, pbest, pbest_fit = iterator(wmax, wmin, c1max, c1min, c2max, c2min, p, dim, iteration, X, V, pbest, gbest, pbest_fit, gbest_fit, input_layer_kernel, hidden_layer_kernel, output_layer_kernel)
t_end_all = time.time()
duration_all = t_end_all - t_start_all




############ plot PSO history ###########
plt.figure(figsize = (12, 12))
plt.title("fitness")
plt.xlabel("iteration")
plt.ylabel("fitness")
t = np.array([t for t in range(0,iteration+1)])
fitness = np.array(fitness)
plt.plot(t, fitness)
#plt.ylim(0, 10)
#plt.yscale('log')
plt.show()

########### plot test result ###############
NN(gbest, val_x, val_y, input_layer_kernel, hidden_layer_kernel, output_layer_kernel, 30)
error = fitfunction(gbest, val_x, val_y, input_layer_kernel, hidden_layer_kernel, output_layer_kernel)
print('total time: ', duration_all)
print('val_test mape: ', error)

NN(gbest, test_x, test_y, input_layer_kernel, hidden_layer_kernel, output_layer_kernel, 30)
error = fitfunction(gbest, test_x, test_y, input_layer_kernel, hidden_layer_kernel, output_layer_kernel)
print('total time: ', duration_all)
print('test mape: ', error)

df_gbest = pd.DataFrame(gbest)
df_gbest.to_csv('PSO_gbest.csv')

