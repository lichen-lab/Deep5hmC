import os
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import TensorDataset, DataLoader

class MyDataLoader():      
    def get_dataloader(self, dstype):
        raise NotImplementedError
        
    def train_dataloader(self):
        return self.get_dataloader(dstype='train')

    def val_dataloader(self):
        return self.get_dataloader(dstype='val')
    
    def test_dataloader(self):
        return self.get_dataloader(dstype='test') 

class MyDataset(MyDataLoader):
    def __init__(self,data_path,tissue,histones,batch_size):
        self.data_path = data_path
        self.tissue = tissue
        self.batch_size = batch_size
        self.histones = histones
                
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.read_size = []
        
        h5_file = f'{data_path}/{tissue}/preprocessed_data.h5'
        x_read1, x_read2, x_seq, y_binary, y_continuous, trainId, valId, testId = self.load_from_hdf5(h5_file)
        
        self.train_dataset.extend([torch.from_numpy(x_read1[trainId]),
                                   torch.from_numpy(x_read2[trainId]),
                                   torch.from_numpy(x_seq[trainId]),
                                   torch.from_numpy(y_binary[trainId]).long(), 
                                   torch.from_numpy(y_continuous[trainId])])
        
        self.val_dataset.extend([torch.from_numpy(x_read1[valId]),
                                   torch.from_numpy(x_read2[valId]),
                                   torch.from_numpy(x_seq[valId]),
                                   torch.from_numpy(y_binary[valId]).long(), 
                                   torch.from_numpy(y_continuous[valId])])
        self.test_dataset.extend([torch.from_numpy(x_read1[testId]),
                                   torch.from_numpy(x_read2[testId]),
                                   torch.from_numpy(x_seq[testId]),
                                   torch.from_numpy(y_binary[testId]).long(), 
                                   torch.from_numpy(y_continuous[testId])])
        
        self.read_size = [x_read1.shape,x_read2.shape]
     
    def load_from_hdf5(self,h5_file):
        with h5py.File(h5_file,'r') as f:
            x_seq = f.get('x_seq')[:]
            x_read1 = f.get('x_read1')[:]
            x_read2 = f.get('x_read2')[:]
            y_binary = f.get('y_binary')[:]
            y_continuous = f.get('y_continuous')[:]
            testId = f.get('testId')[:]
            trainId = f.get('trainId')[:]
            valId = f.get('valId')[:]
        return x_read1,x_read2, x_seq, y_binary, y_continuous, trainId, valId, testId
                
    def get_dataloader(self,dstype):
        shuffle = False
        if dstype == 'train':
            shuffle = True
            dataset = TensorDataset(*self.train_dataset)
        elif dstype == 'val':
            dataset = TensorDataset(*self.val_dataset)
        elif dstype == 'test':
            dataset = TensorDataset(*self.test_dataset)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    