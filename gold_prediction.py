
import os
import json
import time
import argparse
import numpy as np

# Training & Testing
import tensorflow as tf
from tensorflow import keras

def _load_training_data(base_dir):
    """Load training data"""
    x_train = np.load(os.path.join(base_dir, 'train/X_train.npy'))
    y_train = np.load(os.path.join(base_dir, 'train/y_train.npy'))
    return x_train, y_train

def _load_validation_data(base_dir):
    """Load validation data"""
    x_val = np.load(os.path.join(base_dir, 'val/X_val.npy'))
    y_val = np.load(os.path.join(base_dir, 'val/y_val.npy'))
    return x_val, y_val

def _load_testing_data(base_dir):
    """Load testing data"""
    x_test = np.load(os.path.join(base_dir, 'test/X_test.npy'))
    y_test = np.load(os.path.join(base_dir, 'test/y_test.npy'))
    return x_test, y_test

def model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=100, batch_size=128, optimizer='adam', activation='relu', conv_filters=64, kernel_size=3, window_size=64, n_time_features=4, seq_neurons=120, dropout_rate=0.5, dense_neurons=16):
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
    outputs = keras.layers.Dense(1, activation = 'linear', name='dense_output')(x)
    
    model = keras.models.Model(inputs = inputs,
                               outputs = outputs)

    model.compile(
        optimizer = optimizer,
        loss = keras.losses.MeanAbsoluteError(),
        metrics = [keras.metrics.RootMeanSquaredError(name = 'rmse')]
    )
    
    cb = [
    keras.callbacks.EarlyStopping(
        patience = 10, 
        mode = 'min', 
        monitor = 'val_rmse',
        restore_best_weights = True), 
    keras.callbacks.ReduceLROnPlateau(
        patience = 5,
        mode = 'min',
        monitor = 'val_rmse',
        factor = .3,
        min_lr = 1e-5)
        ]
    
    model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = epochs, callbacks = cb, batch_size = batch_size, verbose = 1,)
    model.evaluate(X_test, y_test)
    
    return model

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()

    X_train, y_train = _load_training_data(args.train)
    X_val, y_val = _load_validation_data(args.train)
    X_test, y_test = _load_testing_data(args.train)

    model = model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
        epochs=300, batch_size=128, optimizer='adam', dense_neurons=512, 
        window_size=64, seq_neurons=128, n_time_features=X_train.shape[-1])

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001' in Tensorflow SavedModel Format
        # To export the model as h5 format use model.save('my_model.h5')
        ts = int(time.time())
        model.save(os.path.join(args.sm_model_dir, '00000001'))
