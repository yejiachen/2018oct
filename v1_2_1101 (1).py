

# Hint: 此1.0版本為1102第一此報告前，資料觀察、預處理、與建模之完整流程
# 以rolling window on time series，pridict avg. feature 12 week call-off
# 將每個call-off/此item非0之最小call-off為base unit，降低樣本間之距離
# 以80%為train，20%為test，mae為loss function/ evulation method
# 共136周，以call-off 次數占總週數70%以上item做為訓練，共70種item，從136-96

# 期間嘗試利用k-shape、DTM方式將time-series分群，但未成功建立(未來可嘗試)
# 期間嘗試利用one-hot模式建立sparse format，並嘗試使用RNN/Conv1D

# 最終運用0與call-off兩COLUMN format，minimax scaler對各item訓練
# 並使用xgboost under GridCV，對降低梯度消失之發生，並加快運算速度
# 由於各item pattern不一，開的window週數也不一，需用迴圈各自訓練(plot change of mae)

# 未來計畫：
# 繼續嘗試以NN方法預測(試用encoder structre)
# 需要從數學方法去計算出sample size requirement
# 

import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error ,mean_squared_log_error
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
import xgboost as xgb

from google.colab import files
uploaded = files.upload()

save_df = pd.DataFrame({'item': [], 'window': [],
                        
                        #'test_mse': [],'val_mse': [],'train_mse': [],
                        #'test_msle': [],'val_msle': [],'train_msle': [],
                        'mase_test': [],'raw_mae':[],
                        'test_mae': [],'val_mae': [],'train_mae': []})

df_week = pd.read_csv("raw_top70item_hub.csv").set_index("week_firstday")



forecast_day = 12
base_line_len = 8

training_data_rate = 0.75
val_test_rate = 0.5


for item in list(df_week):
    
    for window in range(12,46):
        df = pd.DataFrame(df_week[item])

        base_unit = int(df[df !=0].min(axis=0))

        df = df/base_unit
        df.columns = ["call_off"]
        df["zero"] = ""

        for i in range(df.shape[0]):
            if int(df.iloc[i,0])>=1:
                df.iloc[i,1]=0
            elif int(df.iloc[i,0])==0:
                df.iloc[i,1]=1

        scaler = MinMaxScaler()
        #scaler = StandardScaler()
        train_data = scaler.fit_transform(df)

        feature = train_data.shape[1]

        train_x = np.empty(shape=(0, window, feature))
        train_y_0 = np.empty(shape=(0, forecast_day))
        base_line = np.empty(shape=(0, base_line_len))

        for i in range(len(train_data)-window-forecast_day+1):
            train_x = np.vstack((train_x, train_data[np.newaxis, i:(i+window), :]))
            train_y_0 = np.vstack((train_y_0, train_data[np.newaxis, (i+window):(i+window+forecast_day), 0]))
            base_line = np.vstack((base_line, train_data[np.newaxis, (i+window-base_line_len) :(i+window), 0]))

        # Make mean of future 12 weeks
        train_y = np.mean(train_y_0, axis=1).reshape(len(train_x),1)
        base_line = np.mean(base_line, axis=1).reshape(len(base_line),1)

        # train, val, test split, with 先後順序
        train_x, validation_x = train_x[:int(len(train_x)*training_data_rate),:,:], train_x[int(len(train_x)*training_data_rate):,:,:] 
        train_y, validation_y = train_y[:int(len(train_y)*training_data_rate),:], train_y[int(len(train_y)*training_data_rate):,:] 
        train_baseline, validation_baseline = base_line[:int(len(base_line)*training_data_rate),:], base_line[int(len(base_line)*training_data_rate):,:] 

        validation_x, test_x = validation_x[:int(len(validation_x)*training_data_rate),:,:], validation_x[int(len(validation_x)*training_data_rate):,:,:] 
        validation_y, test_y = validation_y[:int(len(validation_y)*training_data_rate),:], validation_y[int(len(validation_y)*training_data_rate):,:]
        validation_baseline, test_baseline = validation_baseline[:int(len(validation_baseline)*training_data_rate),:], validation_baseline[int(len(validation_baseline)*training_data_rate):,:]

        # reshape 3d to 2d
        train_x = train_x.reshape((train_x.shape[0], -1))
        validation_x = validation_x.reshape((validation_x.shape[0], -1))
        test_x = test_x.reshape((test_x.shape[0], -1))

        # main model
        xgb = XGBRegressor()
        xgb_parameters = { 'objective':['reg:linear'],'learning_rate': [0.001,0.1,0.01],  'max_depth': [2,3,4], 'n_estimators': [200], 'min_child_weight':[1,3],'gamma':[0.99,3]
                         }
        xgb_grid = GridSearchCV(xgb, xgb_parameters)
        eval_set = [(validation_x, validation_y)]
        xgb_grid.fit(train_x, train_y, early_stopping_rounds=20, eval_metric="mae", eval_set=eval_set, verbose=False)

        
        # predict & evaluate
        test_pred = xgb_grid.predict(test_x)
        val_pred = xgb_grid.predict(validation_x)
        train_pred = xgb_grid.predict(train_x)
        
                
        test_mae = mean_absolute_error(test_y, test_pred)
        val_mae = mean_absolute_error(validation_y, val_pred)
        train_mae = mean_absolute_error(train_y, train_pred)
        # evaluate mase from base line,last 8 weeks average
        mase_test = test_mae / (mean_absolute_error(test_y, test_baseline))

        raw_mae =  test_mae* ((scaler.data_max_[0]-scaler.data_min_[0]) + scaler.data_min_[0])* (base_unit)
    
    
        # save to dataframe
        save_df = save_df.append({'item': item, 'window': window, 
                                 'test_mae': round(test_mae,6),  'val_mae': round(val_mae,6), 'train_mae': round(train_mae,6),
                                 'mase_test': round(mase_test,6) , 'raw_mae': round(raw_mae,2)},
                                 ignore_index=True)

        
        print(window, ', mase_test: ', round(mase_test,4) , ", test_mae: ",round(test_mae,4) ,", val_mae: ",round(val_mae,4) ,", raw_mae: ", int(raw_mae))

save_df.to_csv("1102_mase_save_df.csv")

from google.colab import files
files.download("1102_mase_save_df.csv")

save_df.head()

df = pd.read_csv("1102_mase_save_df.csv")
df = df.drop(list(df)[0], axis=1)
df

df["negative_test_mae"] = df["test_mae"]*(-1)
df2 = df.groupby(by="item").apply( lambda x:x.nlargest(5, "negative_test_mae"))
df2.head
