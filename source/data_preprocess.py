import os
import numpy as np
from Bio import SeqIO
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import argparse
import json

############################
def onehot(sequence):
    x=[]
    for seq_record in sequence:
        seq_array = np.array(list(seq_record))
        label_encoder = LabelEncoder()
        integer_encoded_seq = label_encoder.fit_transform(seq_array)
        onehot_encoder = OneHotEncoder(sparse_output=False)
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
        x.append(onehot_encoded_seq)        
    x = np.array(x)
    return x
        
    
def split_train_test_val(peaks):
    testId = np.where(peaks['seqnames'].isin(['chr8','chr9']))[0]
    valId = np.where(peaks['seqnames'].isin(['chr7']))[0]
    trainId = np.where(~peaks['seqnames'].isin(['chr7','chr8','chr9']))[0]
    return trainId, valId, testId    


def save_to_hdf5(h5_file, x_seq, x_read1,x_read2, y_binary, y_continuous, trainId, valId, testId):
    with h5py.File(h5_file,'w') as f:
        f.create_dataset('x_seq',data=x_seq,compression='gzip')
        f.create_dataset('x_read1',data=x_read1,compression='gzip')
        f.create_dataset('x_read2',data=x_read2,compression='gzip')
        f.create_dataset('y_binary',data=y_binary,compression='gzip')
        f.create_dataset('y_continuous',data=y_continuous,compression='gzip')
        f.create_dataset('trainId',data=trainId,compression='gzip')
        f.create_dataset('valId',data=valId,compression='gzip')
        f.create_dataset('testId',data=testId,compression='gzip')

def create_hdf5(data_path,histones):
    
    peaks = pd.read_csv(f'{data_path}/peaks.csv')

    y_binary = np.array(peaks.y_binary)
    y_continuous = np.array(peaks.y_continuous)

    sequence = np.array(peaks.sequence)
    seq_onehot = onehot(sequence)

    x_seq = np.transpose(seq_onehot,(0,2,1))

    x_reads = []
    for histone in histones:
        x_read = pd.read_csv(f'{data_path}/{histone}.csv')
        x_read = x_read.values
        nsample = int(x_read.shape[1]/41)
        x_read = x_read.reshape(x_read.shape[0],nsample,41)
        x_reads.append(x_read)

    trainId, valId, testId = split_train_test_val(peaks)

    h5_file = f'{data_path}/preprocessed_data.h5'
    save_to_hdf5(h5_file,x_seq,x_reads[0],x_reads[1],y_binary,y_continuous,trainId, valId, testId)
    
def load_from_hdf5(h5_file):
    with h5py.File(h5_file,'r') as f:
        x_seq = f.get('x_seq')[:]
        x_read1 = f.get('x_read1')[:]
        x_read2 = f.get('x_read2')[:]
        y_binary = f.get('y_binary')[:]
        y_continuous = f.get('y_continuous')[:]
        testId = f.get('testId')[:]
        trainId = f.get('trainId')[:]
        valId = f.get('valId')[:]
    return x_seq, x_read1,x_read2, y_binary, y_continuous, trainId, valId, testId



def main(json_file):
    with open(json_file, 'r') as f:
        config = json.load(f)

    data_path = config['data_path']
    histones = config['histones']

    create_hdf5(data_path, histones)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run create_hdf5 with config file")
    parser.add_argument('json_file', type=str, help='Path to the JSON configuration file')
    args = parser.parse_args()

    main(args.json_file)
    