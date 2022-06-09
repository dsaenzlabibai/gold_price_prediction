

import os
import codecs
import json
import time
import argparse

# Web-scrapping Yahoo Finance data
import yfinance as yf

# Data ETL
import re
import numpy as np
import pandas as pd

# Training & Testing
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

def _scrap_stock_data(stocks_file_path, start="2000-01-01", end="2022-06-01"):
    """Scrap stock data from the Yahoo Finace"""
    obj_text = codecs.open(stocks_file_path, 'r', encoding='utf-8').read()
    stocks = json.loads(obj_text)
    data = yf.download(
        " ".join([obs['stock_ticker'] for obs in stocks]), 
        start=start,
        end=end,
        group_by='ticker')

    data.columns = data.columns.map('_'.join)
    data.columns = [c.lower() for c in data.columns]
    data.columns = [c.replace("^", "") for c in data.columns]
    data.columns = [c.replace("=", "") for c in data.columns]
    data.columns = [c.replace(" ", "_") for c in data.columns]
    data.reset_index()

    stock_tickers = [obs['stock_ticker'] for obs in stocks if obs['feature']]
    stock_tickers = [st.lower() for st in stock_tickers]
    stock_tickers = [st.replace("^", "") for st in stock_tickers]
    stock_tickers = [st.replace("=", "") for st in stock_tickers]
    stock_tickers = [st.replace(" ", "_") for st in stock_tickers]

    stock_ticker_target = [obs['target'] for obs in stocks if not obs['feature']]
    stock_ticker_target = [st.lower() for st in stock_ticker_target]
    stock_ticker_target = [st.replace("^", "") for st in stock_ticker_target]
    stock_ticker_target = [st.replace("=", "") for st in stock_ticker_target]
    stock_ticker_target = [st.replace(" ", "_") for st in stock_ticker_target]

    print("Total number of observations is:", len(data))

    X_features = [col for col in data.columns for st in stock_tickers if re.search(f'^{st}', col)]
    df = data[X_features]
    df['target'] = data[stock_ticker_target[0]].shift(-1)
    df.dropna(inplace=True)

    feature_corr_map = df.iloc[:, :-1].corr()
    feature_corr_map = feature_corr_map - np.eye(feature_corr_map.shape[0])
    to_drop = [column for column in feature_corr_map.columns if any(abs(feature_corr_map[column]) > 0.5)]

    to_drop_chosen = []
    for st in stock_tickers:
        temp = [c for c in to_drop if re.search(f'^{st}', c)]
        to_drop_chosen.extend(temp[:-1])
    df = df.drop(columns=to_drop_chosen, axis=1)

    return df

def conv_sequence_input(data, target_column=-1, window_size=64):
    """Create convolutional input from time sequential data"""
    X = []
    y = []

    for i in range(0 , data.shape[0] - window_size, 1):
        temp2 = []
        for t in range(window_size):
            temp2.append(data[i + t, 0])
        X.append(np.array(temp2).reshape(window_size, 1))
        y.append(data[i + window_size, target_column])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    if X.shape[1] > 1:
        for j in range(1, data.shape[1]):
            temp = []
            for i in range(0 , data.shape[0] - window_size, 1):
                temp2 = []
                for t in range(window_size):
                    temp2.append(data[i + t, j])
                temp.append(np.array(temp2).reshape(window_size, 1))
            X = np.concatenate((X, temp), axis=-1)

    return X, y

def build_model(optimizer='adam', activation='relu', conv_filters=64, kernel_size=3, window_size=64, n_time_features=4, seq_neurons=120, dropout_rate=0.5, dense_neurons=16):
    inputs = keras.layers.Input(shape = (None, window_size, n_time_features,))
    x = keras.layers.TimeDistributed(keras.layers.Conv1D(conv_filters, kernel_size=kernel_size, activation=activation))(inputs)
    x = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(2))(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv1D(conv_filters * 2, kernel_size=kernel_size, activation=activation))(x)
    x = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(2))(x)
    x = keras.layers.TimeDistributed(keras.layers.Conv1D(conv_filters * 3, kernel_size=kernel_size, activation=activation))(x)
    x = keras.layers.TimeDistributed(keras.layers.MaxPooling1D(2))(x)
    x = keras.layers.TimeDistributed(keras.layers.Flatten())(x)

    x = keras.layers.Bidirectional(keras.layers.LSTM(seq_neurons, return_sequences=True))(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(seq_neurons, return_sequences=False))(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    x = keras.layers.Dense(dense_neurons)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation)(x)
    outputs = keras.layers.Dense(1, activation = 'linear')(x)
    
    model = keras.models.Model(inputs = inputs,
                               outputs = outputs)

    model.compile(
        optimizer = optimizer,
        loss = keras.losses.MeanAbsoluteError(),
        metrics = [keras.metrics.RootMeanSquaredError(name = 'rmse'), keras.metrics.MeanAbsoluteError(name='mae')]
    )
    return model

def train_model(X_train, y_train, X_val, y_val, model_ckpt_path, model_logs_path, epochs=100, batch_size=128, optimizer='adam', dense_neurons=512, window_size=64, n_time_features=2):
    """Train the regression model using callbacks and the builder"""
    
    cb = [
        keras.callbacks.EarlyStopping(
            patience = 10, 
            mode = 'min', 
            monitor = 'val_rmse',
            restore_best_weights = True), 
        keras.callbacks.ModelCheckpoint(
            filepath=model_ckpt_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='min',),
        keras.callbacks.TensorBoard(
            log_dir=model_logs_path,
            histogram_freq=1),
        keras.callbacks.ReduceLROnPlateau(
            patience = 5,
            mode = 'min',
            monitor = 'val_rmse',
            factor = .3,
            min_lr = 1e-5)
            ]

    model = KerasRegressor(
        build_model, epochs = epochs, callbacks = cb, batch_size = batch_size, verbose = 1, seq_neurons = 2 * window_size,
        optimizer = optimizer, dense_neurons=dense_neurons, window_size = window_size, n_time_features=n_time_features
        )
    
    model.fit(X_train, y_train, validation_data = (X_val, y_val))

    return model

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--stocks_file_path', type=str)
    parser.add_argument('--start', type=str, default='2000-01-01')
    parser.add_argument('--end', type=str, default='2022-06-01')
    parser.add_argument('--window_size', type=int, default=64)
    parser.add_argument('--model_ckpt_path', type=str)
    parser.add_argument('--model_logs_path', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    df = _scrap_stock_data(stocks_file_path=args.stocks_file_path, start=args.start, end=args.end)

    df_train, df_val = train_test_split(df, test_size=0.15, shuffle = False)
    df_val, df_test = train_test_split(df_val, test_size=0.33, shuffle = False)

    # Normalize data using training information
    scaler = MinMaxScaler()
    scaler.fit(df_train)
    X_train = scaler.transform(df_train)
    X_val = scaler.transform(df_val)

    X_val, X_test = train_test_split(X_val, test_size=0.33, shuffle = False)
    X_train, y_train = conv_sequence_input(X_train, target_column=-1, window_size=args.window_size)
    X_val, y_val = conv_sequence_input(X_val, target_column=-1, window_size=args.window_size)
    X_test, y_test = conv_sequence_input(X_test, target_column=-1, window_size=args.window_size)

    X_train = np.expand_dims(X_train, axis=1)
    X_val = np.expand_dims(X_val, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    model = train_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        model_ckpt_path=args.model_ckpt_path, model_logs_path=args.model_logs_path, 
        epochs=300, batch_size=128, optimizer='adam', dense_neurons=512, 
        window_size=args.window_size, n_time_features=X_train.shape[-1])

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001' in Tensorflow SavedModel Format
        # To export the model as h5 format use model.save('my_model.h5')
        ts = int(time.time())
        model.model.save(os.path.join(args.sm_model_dir, '001'))