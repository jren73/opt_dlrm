import random
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle as pkl
import keras
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
    multiply, concatenate, Flatten, Activation, dot
from tensorflow.keras.optimizers import Adam 
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
import pydot as pyd
from keras.utils.vis_utils import plot_model, model_to_dot
keras.utils.vis_utils.pydot = pyd
import torch
import collections 
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
from utils import get_logger

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

_logger = get_logger(__name__)

#processing data
def data(input_file):
        indices, offsets, lengths = torch.load(input_file)
        print(f"Data file indices = {indices.size()}", f"offsets = {offsets.size()}, lengths = {lengths.size()}) ")
        return indices

def truncate(x, feature_cols=range(3), target_cols=range(3), label_col=3, train_len=100, test_len=20):
        in_, out_, lbl = [], [], []
        for i in range(len(x)-train_len-test_len+1):
                in_.append(x[i:(i+train_len), feature_cols].tolist())
                out_.append(x[(i+train_len):(i+train_len+test_len), target_cols].tolist())
                lbl.append(x[i+train_len, label_col])
        return np.array(in_), np.array(out_), np.array(lbl)

def merge(list1, list2):
      
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def model(N, M, X_input_train, X_output_train):
        n_hidden = N

        #input layer
        input_train = Input(shape=(X_input_train.shape[1], X_input_train.shape[2]-1))
        output_train = Input(shape=(X_output_train.shape[1], X_output_train.shape[2]-1))

        # encoder
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
        n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2, 
        return_state=True, return_sequences=True)(input_train)
        print(encoder_stack_h)
        print(encoder_last_h)
        print(encoder_last_c)

        #batch_norm
        encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

        #decoder
        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        print(decoder_input)

        decoder_stack_h = LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
        return_state=False, return_sequences=True)(
        decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        print(decoder_stack_h)

        #attention layer: Luong attention
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)
        print(attention)

        context = dot([attention, encoder_stack_h], axes=[2,1])
        context = BatchNormalization(momentum=0.6)(context)
        print(context)
        decoder_combined_context = concatenate([context, decoder_stack_h])
        print(decoder_combined_context)
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
        print(out)



def test(X_input_train, X_input_test, X_output_train, X_output_test):
        train_pred_detrend = model.predict(X_input_train[:, :, :2])*x_train_max[:2]
        test_pred_detrend = model.predict(X_input_test[:, :, :2])*x_train_max[:2]
        print(train_pred_detrend.shape, test_pred_detrend.shape)
        train_true_detrend = X_output_train[:, :, :2]*x_train_max[:2]
        test_true_detrend = X_output_test[:, :, :2]*x_train_max[:2]
        print(train_true_detrend.shape, test_true_detrend.shape)

        train_pred_detrend = np.concatenate([train_pred_detrend, np.expand_dims(X_output_train[:, :, 2], axis=2)], axis=2)
        test_pred_detrend = np.concatenate([test_pred_detrend, np.expand_dims(X_output_test[:, :, 2], axis=2)], axis=2)
        print(train_pred_detrend.shape, test_pred_detrend.shape)
        train_true_detrend = np.concatenate([train_true_detrend, np.expand_dims(X_output_train[:, :, 2], axis=2)], axis=2)
        test_true_detrend = np.concatenate([test_true_detrend, np.expand_dims(X_output_test[:, :, 2], axis=2)], axis=2)
        print(train_pred_detrend.shape, test_pred_detrend.shape)


def main():
        parser = argparse.ArgumentParser(description='caching model.\n')
        parser.add_argument('traceFile', type=str,  help='trace file name\n')
        parser.add_argument('n', type=int,  help='input sequence length N\n')
        parser.add_argument('m', type=int,  help='output sequence length\n')
        args = parser.parse_args() 

        traceFile = args.traceFile
        M = args.m
        N = args.n
        gt_trace = traceFile[0:traceFile.rfind(".pt")] + "_cached_trace_opt.csv"

        #dataset = data("dlrm_datasets/embedding_bag/fbgemm_t856_bs65536_9.pt")
        dataset = data(traceFile)
        csvdata = pd.read_csv(gt_trace)
        gt = csvdata[1].tolist()
        #ensure the training and groudtruth has the same size. When we processing groundtruth, we cutout some data
        dataset = dataset[:len(gt)]
         #input sequence length
        #N = 150
        #output sequence length
        #M = 10
        # evalutaion window size
        #W = 150

        X_in, X_out, lbl = truncate(merge(dataset, gt), feature_cols=range(2), target_cols=0, 
                            label_col=1, train_len=N, test_len=M)
        X_input_train = X_in[np.where(lbl==1)]
        X_output_train = X_out[np.where(lbl==1)]
        X_input_test = X_in[np.where(lbl==0)]
        X_output_test = X_out[np.where(lbl==0)]
        print(X_input_train.shape, X_output_train.shape)
        print(X_input_test.shape, X_output_test.shape)

        X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,:-1], dataset[:,-1].astype(int), test_size=0.2, random_state=None, shuffle=True)
        
       
        
        model = model(N, M, X_train, Y_train)
        opt = Adam(lr=0.01, clipnorm=1)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        model.summary()

        ep=200
        es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
        history = model.fit(X_input_train[:, :, :2], X_output_train[:, :, :2], validation_split=0.2, 
                        epochs=epc, verbose=1, callbacks=[es], 
                        batch_size=100)
        train_mae = history.history['mae']
        valid_mae = history.history['val_mae']
        
        model.save('model_caching_seq2seq.h5')
        

if __name__ == "__main__":
    main()