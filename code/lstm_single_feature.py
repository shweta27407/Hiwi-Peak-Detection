# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Specify the path of the datasets
file_path = ['C:/Users/genty/Hiwi-TimeSeries/dataset/DatAmount_dataset_new/1_CMX/CMX1_AL_CP1.csv',
             'C:/Users/genty/Hiwi-TimeSeries/dataset/DatAmount_dataset_new/1_CMX/CMX1_AL_CP2.csv',
             'C:/Users/genty/Hiwi-TimeSeries/dataset/DatAmount_dataset_new/1_CMX/CMX1_S_CP1.csv',
             'C:/Users/genty/Hiwi-TimeSeries/dataset/DatAmount_dataset_new/1_CMX/CMX1_S_CP2.csv',
             'C:/Users/genty/Hiwi-TimeSeries/dataset/DatAmount_dataset_new/2_DMC/DMC2_AL_CP1.csv',
             'C:/Users/genty/Hiwi-TimeSeries/dataset/DatAmount_dataset_new/2_DMC/DMC2_AL_CP2.csv',
             'C:/Users/genty/Hiwi-TimeSeries/dataset/DatAmount_dataset_new/2_DMC/DMC2_S_CP1.csv',
             'C:/Users/genty/Hiwi-TimeSeries/dataset/DatAmount_dataset_new/2_DMC/DMC2_S_CP2.csv']

features =  ['LOAD|1', 'LOAD|2', 'LOAD|3', 'LOAD|6',
       'ENC_POS|1', 'ENC_POS|2', 'ENC_POS|3','ENC_POS|6',
       'CTRL_DIFF2|1', 'CTRL_DIFF2|2', 'CTRL_DIFF2|3', 'CTRL_DIFF2|6',
       'TORQUE|1', 'TORQUE|2', 'TORQUE|3', 'TORQUE|6',
       'DES_POS|1', 'DES_POS|2', 'DES_POS|3', 'DES_POS|6',
       'CTRL_DIFF|1', 'CTRL_DIFF|2', 'CTRL_DIFF|3' ,'CTRL_DIFF|6',
       'CTRL_POS|1', 'CTRL_POS|2', 'CTRL_POS|3', 'CTRL_POS|6',
       'VEL_FFW|1', 'VEL_FFW|2','VEL_FFW|3', 'VEL_FFW|6',
       'CONT_DEV|1','CONT_DEV|2', 'CONT_DEV|3', 'CONT_DEV|6',
       'CMD_SPEED|1', 'CMD_SPEED|2', 'CMD_SPEED|3', 'CMD_SPEED|6',
       'TORQUE_FFW|1', 'TORQUE_FFW|2', 'TORQUE_FFW|3', 'TORQUE_FFW|6',
       'ENC1_POS|1', 'ENC1_POS|2', 'ENC1_POS|3','ENC1_POS|6',
       'ENC2_POS|1', 'ENC2_POS|2', 'ENC2_POS|3', 'ENC2_POS|6']

target = 'CURRENT|6'

output_folder = 'C:/Users/genty/Hiwi-TimeSeries/output/' # change the output folder here
result_df = pd.DataFrame(columns=['Feature', 'Test_RMSE']) # Change the structure of resulting csv file here

def normalize(X, y):
    '''
    This function normalizes the features (X) and target (y)
    '''
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaled_X = scaler_X.fit_transform(X)
    scaled_y = scaler_y.fit_transform(y)

    return scaler_X, scaler_y, scaled_X, scaled_y

def denormalize(train_predict, y_train, test_predict, y_test, scaler_y):
    '''
    This function denormalizes the features (X) and target (y)
    '''
    train_predict_inv = scaler_y.inverse_transform(train_predict)
    y_train_inv = scaler_y.inverse_transform(y_train)
    test_predict_inv = scaler_y.inverse_transform(test_predict)
    y_test_inv = scaler_y.inverse_transform(y_test)
    
    return train_predict_inv, y_train_inv, test_predict_inv, y_test_inv

def create_dataset(X, y, time_step=60):
    '''
    This function creates batches of dataset to be fed to the neural network
    '''
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        if isinstance(X, np.ndarray):
            Xs.append(X[i:(i + time_step)])
            ys.append(y[i + time_step])
        else:
            Xs.append(X.iloc[i:(i + time_step)].values)
            ys.append(y.iloc[i + time_step].values)
    return np.array(Xs), np.array(ys)

def build_model(input_shape):
    '''
    Creates a neural network model
    '''
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

for file in file_path: # Iterates through all files listed in file_path
    df = pd.read_csv(file)
    for f in features: # Iterates through all features listed in features
        print(f)
        X = df[[f]] # Only one feature
        y = df[[target]]
        scaler_X, scaler_y, scaled_X, scaled_y = normalize(X, y)
        X = scaled_X
        y = scaled_y
        train_size = int(len(X) * 0.7)
        test_size = len(X) - train_size
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        time_step = 60
        X_train, y_train = create_dataset(X_train, y_train, time_step)
        X_test, y_test = create_dataset(X_test, y_test, time_step)

        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_model(input_shape)
        history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        train_predict, y_train, test_predict, y_test = denormalize(train_predict, y_train, test_predict, y_test, scaler_y)

        train_r2 = r2_score(y_train, train_predict)
        test_r2 = r2_score(y_test, test_predict)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predict))

        result_df = pd.concat([result_df, pd.DataFrame([{       
            'Feature': f,
            'Test_RMSE': test_rmse
            }])], ignore_index=True)
        
        result_df.sort_values(by='Test_RMSE', ascending=True, inplace=True)
        filenamefinal = os.path.join(output_folder, f'{os.path.basename(file)}_summary_lstm_single_feature.csv')
        result_df.to_csv(filenamefinal, index=False)

                