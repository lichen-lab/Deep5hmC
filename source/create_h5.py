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

# for binary/cont
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
    

# for gene expression
def save_to_hdf5_gene(h5_file, x_seq, x_read1,x_read2, y_5hmC, y_GE,gene_id):
    with h5py.File(h5_file,'w') as f:
        f.create_dataset('x_seq',data=x_seq,compression='gzip')
        f.create_dataset('x_read1',data=x_read1,compression='gzip')
        f.create_dataset('x_read2',data=x_read2,compression='gzip')
        f.create_dataset('y_5hmC',data=y_5hmC,compression='gzip')
        f.create_dataset('y_GE',data=y_GE,compression='gzip')
        f.create_dataset('gene_id',data=gene_id,compression='gzip')
        
        

def create_hdf5_gene(data_path,histones):
    
    peaks = pd.read_csv(f'{data_path}/genes.csv')

    y_5hmC = np.array(peaks.y_5hmC)
    y_GE = np.array(peaks.y_GE)

    sequence = np.array(peaks.sequence)
    seq_onehot = onehot(sequence)

    x_seq = np.transpose(seq_onehot,(0,2,1))
    
    gene_id = np.array(peaks.gene_id)

    x_reads = []
    for histone in histones:
        x_read = pd.read_csv(f'{data_path}/{histone}_gene.csv')
        x_read = x_read.values
        nsample = int(x_read.shape[1]/41)
        x_read = x_read.reshape(x_read.shape[0],nsample,41)
        x_reads.append(x_read)

    h5_file = f'{data_path}/preprocessed_data.h5'
    save_to_hdf5_gene(h5_file,x_seq,x_reads[0],x_reads[1],y_5hmC,y_GE,gene_id)
    

# for ADCT
def save_to_hdf5_diff(h5_file,x_read1,x_read2,x_read3,x_read4,y_binary,trainId, valId, testId):
    with h5py.File(h5_file,'w') as f:
        f.create_dataset('x_read1',data=x_read1,compression='gzip')
        f.create_dataset('x_read2',data=x_read2,compression='gzip')
        f.create_dataset('x_read3',data=x_read3,compression='gzip')
        f.create_dataset('x_read4',data=x_read4,compression='gzip')
        f.create_dataset('y_binary',data=y_binary,compression='gzip')
        f.create_dataset('trainId',data=trainId,compression='gzip')
        f.create_dataset('valId',data=valId,compression='gzip')
        f.create_dataset('testId',data=testId,compression='gzip')
        
        

def create_hdf5_diff(data_path,histones):
    
    peaks = pd.read_csv(f'{data_path}/peaks.csv')

    y_binary = np.array(peaks.y_binary)

    x_reads = []
    groups = ['AD','CT']
    for group in groups:
        for histone in histones:
            x_read = pd.read_csv(f'{data_path}/{histone}_{group}.csv')
            x_read = x_read.values
            nsample = int(x_read.shape[1]/209)
            x_read = x_read.reshape(x_read.shape[0],nsample,209)
            x_reads.append(x_read)

    trainId, valId, testId = split_train_test_val(peaks)
    h5_file = f'{data_path}/preprocessed_data.h5'
    save_to_hdf5_diff(h5_file,x_reads[0],x_reads[1],x_reads[2],x_reads[3],y_binary,trainId, valId, testId)
     
# ###################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training and evaluation with config file")
    parser.add_argument('json_file', type=str, help='Path to the JSON configuration file')
    args = parser.parse_args()

    with open(args.json_file, 'r') as f:
        config = json.load(f)

    data_path = config['data_path']
    weights_path = config['weights_path']
    prediction_path = config['prediction_path']

    model_type = config['model_type']
    tissue = config['tissue']
    histones = config['histones']

    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    lr = config['lr']
    earlystop_thresh = config['earlystop_thresh']

    if model_type in ['binary', 'cont']:
        data_dir = f'{data_path}/{tissue}/binary_cont'
        create_hdf5(data_dir, histones)
    elif model_type == 'gene':
        data_dir = f'{data_path}/{tissue}/gene'
        create_hdf5_gene(data_dir, histones)
    elif model_type == 'diff':
        data_dir = f'{data_path}/{tissue}/diff'
        create_hdf5_diff(data_dir, histones)
    