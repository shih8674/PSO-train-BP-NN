# PSO-train-BP-NN
This project use PSO to train the single layer NN structure.

If you only want to see the training process and result, 'improved_PSO_train_NN.py' can run directly.

'NN_train_AQI.py' is the traditional NN set up by keras, which can compare with the result of 'improved_PSO_train_NN.py'.

When the 'improved_PSO_train_NN.py' run, the console will represent the number of epoch, the error of val data, the error of test data, and cost time per epoch.

The data is air quality index download from Environmental Protection Administration Executive Yuan, R.O.C. (Taiwan). (https://data.epa.gov.tw/dataset/aqx_p_13)

'space' and 'time' is original data.

'space_train' and 'time_train' is the data after preprocessing.

'AQI_label.py', 'only_6AQI.py', 'space_AQI_label.py',and 'origin_6AQI_label.py' is the preprocessing of data.

The process divided the data into three kinds.
  1.Predict the AQI at different times in the same place.
  2.Predict the AQI at different places in the same time.
  3.Predict AQI from 6 kind of air quality values.
  
  
