import os
import torch
import numpy as np
from Bio import SeqIO
import pandas as pd
import h5py
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
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

class MyDataset_binary(MyDataLoader):
    def __init__(self,batch_size,histones):
        self.batch_size = batch_size
        self.histones = histones
        
        self.peaks_path = '../data/binary_cont/peaks'
        self.seqs_path = '../data/binary_cont/seqs'
        
        # create hdf5 for each histone
        self.create_hdf5(histones)
                
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.read_size = []
        
        for histone in histones:
            h5_file = f'../h5/{histone}.h5'
            x_seq, x_read, y1, y2, trainId, valId, testId = self.load_from_hdf5(h5_file)
            self.train_dataset.append(torch.from_numpy(x_read[trainId]))
            self.val_dataset.append(torch.from_numpy(x_read[valId]))
            self.test_dataset.append(torch.from_numpy(x_read[testId]))
            self.read_size.append(x_read.shape)
        

        self.train_dataset.extend([torch.from_numpy(x_seq[trainId]),torch.from_numpy(y1[trainId]).long()])
        self.val_dataset.extend([torch.from_numpy(x_seq[valId]),torch.from_numpy(y1[valId]).long()])
        self.test_dataset.extend([torch.from_numpy(x_seq[testId]),torch.from_numpy(y1[testId]).long()])
        
    def create_hdf5(self,histones):
        if not os.path.exists('../h5'):
            os.makedirs('../h5')
            
        for histone in histones:
            reads_path = f'../data/binary_cont/{histone}'
            h5_file = f'../h5/{histone}.h5'
        
            if os.path.exists(h5_file):
                print(h5_file+' exists and load data directly.')
            
            else:
                print(h5_file+' does not exist and is being created.')
                x_pos_seq, x_neg_seq = self.load_seqs(self.seqs_path)
                print('seqs loaded')
                x_pos_peak, x_neg_peak = self.load_peaks(self.peaks_path)
                print('peaks loaded')
                x_pos_read, x_neg_read  = self.load_reads(reads_path)
                print('reads loaded')
                
                y1_pos, y1_neg = self.label_binary(x_pos_read,x_neg_read)
                y2_pos, y2_neg = self.label_cont('../data/binary_cont/5hmC_peak')
                print('y loaded')
                    
                x_seq, x_read, x_peak, y1, y2 = self.combine_pos_neg(x_pos_seq, x_pos_peak, x_pos_read, x_neg_seq, x_neg_peak, x_neg_read, y1_pos, y1_neg, y2_pos, y2_neg)
                trainId, valId, testId = self.split_train_test_val(x_peak)
                self.save_to_hdf5(h5_file,x_seq,x_read,y1,y2,trainId,valId,testId)
     
    def load_from_hdf5(self,h5_file):
        with h5py.File(h5_file,'r') as f:
            x_seq = f.get('x_seq')[:]
            x_read = f.get('x_read')[:]
            y1 = f.get('y1')[:]
            y2 = f.get('y2')[:]
            testId = f.get('testId')[:]
            trainId = f.get('trainId')[:]
            valId = f.get('valId')[:]
        return x_seq, x_read, y1, y2, trainId, valId, testId
    
    def load_seqs(self,seqs_path):
        seq_pos_file = os.path.join(seqs_path,'seqs.pos.allchr.fasta')
        seq_neg_file = os.path.join(seqs_path,'seqs.neg.allchr.fasta')
        x_pos_seq = self.onehot(seq_pos_file)
        x_neg_seq = self.onehot(seq_neg_file)
        return x_pos_seq, x_neg_seq
    
    def onehot(self,fafile):
        x=[]
        for seq_record in SeqIO.parse(fafile, "fasta"):
            seq_array = np.array(list(seq_record.seq))
            label_encoder = LabelEncoder()
            integer_encoded_seq = label_encoder.fit_transform(seq_array)
            onehot_encoder = OneHotEncoder(sparse_output=False)
            integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
            onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
            x.append(onehot_encoded_seq)        
        x = np.array(x)
        return x  
    
    def load_peaks(self,peaks_path):
        peak_pos_file = os.path.join(peaks_path,'peaks.pos.allchr.txt')
        peak_neg_file = os.path.join(peaks_path,'peaks.neg.allchr.txt')
        x_pos_peak = pd.read_csv(peak_pos_file, sep="\t")
        x_neg_peak = pd.read_csv(peak_neg_file, sep="\t")
        return x_pos_peak, x_neg_peak
    
    def load_reads(self,reads_path):
        readlists = os.listdir(reads_path)
        readlists = [file for file in readlists if file.startswith('reads')]
        cell_lines = np.arange(1,1+len(readlists)/2).astype(int)
        x_pos_read = np.stack([pd.read_csv(os.path.join(reads_path,f'reads.pos.allchr.{cell_line}.txt'), delimiter='\t',header=None) for cell_line in cell_lines], axis = 1)
        x_neg_read = np.stack([pd.read_csv(os.path.join(reads_path,f'reads.neg.allchr.{cell_line}.txt'), delimiter='\t',header=None) for cell_line in cell_lines], axis = 1)
        return x_pos_read, x_neg_read 
    
    def label_binary(self,x_pos_read,x_neg_read):
        y_pos= np.ones(x_pos_read.shape[0])
        y_pos= np.array(y_pos)
        y_neg= np.zeros(x_neg_read.shape[0])
        y_neg= np.array(y_neg)
        return y_pos, y_neg
    
    def label_cont(self,y_path):
        y_pos_file = os.path.join(y_path,'y.pos.txt')
        y_neg_file = os.path.join(y_path,'y.neg.txt')
        y_pos = pd.read_csv(y_pos_file, sep="\t",header=None)
        y_neg = pd.read_csv(y_neg_file, sep="\t",header=None)
        y_pos = np.squeeze(np.array(y_pos))
        y_neg = np.squeeze(np.array(y_neg))
        return y_pos, y_neg
    
    def combine_pos_neg(self, x_pos_seq, x_pos_peak, x_pos_read, x_neg_seq, x_neg_peak, x_neg_read, y1_pos, y1_neg,y2_pos, y2_neg):
        x_seq = np.concatenate((x_pos_seq,x_neg_seq),axis=0)
        x_seq = np.swapaxes(x_seq,2,1)
        x_read = np.concatenate((x_pos_read,x_neg_read),axis=0)
        x_peak = pd.concat((x_pos_peak,x_neg_peak),axis=0)
        y1 = np.concatenate((y1_pos,y1_neg),axis=0)
        y2 = np.concatenate((y2_pos,y2_neg),axis=0)

        np.random.seed(1234)
        indices = np.arange(len(y1))
        indices = np.random.permutation(indices)

        y1 = y1[indices]
        y2 = y2[indices]
        x_seq = x_seq[indices]
        x_read = x_read[indices]
        x_peak = x_peak.iloc[indices]
        return x_seq, x_read, x_peak, y1, y2
    
    def split_train_test_val(self, x_peak):
        testId = np.where(x_peak['seqnames'].isin(['chr8','chr9']))[0]
        valId = np.where(x_peak['seqnames'].isin(['chr7']))[0]
        trainId = np.where(~x_peak['seqnames'].isin(['chr7','chr8','chr9']))[0]
        return trainId, valId, testId
    
    def save_to_hdf5(self, h5_file, x_seq, x_read, y1, y2, trainId, valId, testId):
        with h5py.File(h5_file,'w') as f:
            f.create_dataset('x_seq',data=x_seq,compression='gzip')
            f.create_dataset('x_read',data=x_read,compression='gzip')
            f.create_dataset('y1',data=y1,compression='gzip')
            f.create_dataset('y2',data=y2,compression='gzip')
            f.create_dataset('trainId',data=trainId,compression='gzip')
            f.create_dataset('valId',data=valId,compression='gzip')
            f.create_dataset('testId',data=testId,compression='gzip')
                
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
    

class MyDataset_cont(MyDataLoader):
    def __init__(self,batch_size,histones):
        self.batch_size = batch_size
        self.histones = histones

        self.peaks_path = '../data/binary_cont/peaks'
        self.seqs_path = '../data/binary_cont/seqs'

        # create hdf5 for each histone
        self.create_hdf5(histones)

        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.read_size = []

        for histone in histones:
            h5_file = f'../h5/{histone}.h5'
            x_seq, x_read, y1, y2, trainId, valId, testId = self.load_from_hdf5(h5_file)
            self.train_dataset.append(torch.from_numpy(x_read[trainId]))
            self.val_dataset.append(torch.from_numpy(x_read[valId]))
            self.test_dataset.append(torch.from_numpy(x_read[testId]))
            self.read_size.append(x_read.shape)


        self.train_dataset.extend([torch.from_numpy(x_seq[trainId]),torch.from_numpy(y2[trainId])])
        self.val_dataset.extend([torch.from_numpy(x_seq[valId]),torch.from_numpy(y2[valId])])
        self.test_dataset.extend([torch.from_numpy(x_seq[testId]),torch.from_numpy(y2[testId])])

    def create_hdf5(self,histones):
        if not os.path.exists('../h5'):
            os.makedirs('../h5')

        for histone in histones:
            reads_path = f'../data/binary_cont/{histone}'
            h5_file = f'../h5/{histone}.h5'

            if os.path.exists(h5_file):
                print(h5_file+' exists and load data directly.')

            else:
                print(h5_file+' does not exist and is being created.')
                x_pos_seq, x_neg_seq = self.load_seqs(self.seqs_path)
                print('seqs loaded')
                x_pos_peak, x_neg_peak = self.load_peaks(self.peaks_path)
                print('peaks loaded')
                x_pos_read, x_neg_read  = self.load_reads(reads_path)
                print('reads loaded')

                y1_pos, y1_neg = self.label_binary(x_pos_read,x_neg_read)
                y2_pos, y2_neg = self.label_cont('../data/binary_cont/5hmC_peak')
                print('y loaded')

                x_seq, x_read, x_peak, y1, y2 = self.combine_pos_neg(x_pos_seq, x_pos_peak, x_pos_read, x_neg_seq, x_neg_peak, x_neg_read, y1_pos, y1_neg, y2_pos, y2_neg)
                trainId, valId, testId = self.split_train_test_val(x_peak)
                self.save_to_hdf5(h5_file,x_seq,x_read,y1,y2,trainId,valId,testId)

    def load_from_hdf5(self,h5_file):
        with h5py.File(h5_file,'r') as f:
            x_seq = f.get('x_seq')[:]
            x_read = f.get('x_read')[:]
            y1 = f.get('y1')[:]
            y2 = f.get('y2')[:]
            testId = f.get('testId')[:]
            trainId = f.get('trainId')[:]
            valId = f.get('valId')[:]
        return x_seq, x_read, y1, y2, trainId, valId, testId

    def load_seqs(self,seqs_path):
        seq_pos_file = os.path.join(seqs_path,'seqs.pos.allchr.fasta')
        seq_neg_file = os.path.join(seqs_path,'seqs.neg.allchr.fasta')
        x_pos_seq = self.onehot(seq_pos_file)
        x_neg_seq = self.onehot(seq_neg_file)
        return x_pos_seq, x_neg_seq

    def onehot(self,fafile):
        x=[]
        for seq_record in SeqIO.parse(fafile, "fasta"):
            seq_array = np.array(list(seq_record.seq))
            label_encoder = LabelEncoder()
            integer_encoded_seq = label_encoder.fit_transform(seq_array)
            onehot_encoder = OneHotEncoder(sparse_output=False)
            integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
            onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
            x.append(onehot_encoded_seq)        
        x = np.array(x)
        return x  

    def load_peaks(self,peaks_path):
        peak_pos_file = os.path.join(peaks_path,'peaks.pos.allchr.txt')
        peak_neg_file = os.path.join(peaks_path,'peaks.neg.allchr.txt')
        x_pos_peak = pd.read_csv(peak_pos_file, sep="\t")
        x_neg_peak = pd.read_csv(peak_neg_file, sep="\t")
        return x_pos_peak, x_neg_peak

    def load_reads(self,reads_path):
        readlists = os.listdir(reads_path)
        readlists = [file for file in readlists if file.startswith('reads')]
        cell_lines = np.arange(1,1+len(readlists)/2).astype(int)
        x_pos_read = np.stack([pd.read_csv(os.path.join(reads_path,f'reads.pos.allchr.{cell_line}.txt'), delimiter='\t',header=None) for cell_line in cell_lines], axis = 1)
        x_neg_read = np.stack([pd.read_csv(os.path.join(reads_path,f'reads.neg.allchr.{cell_line}.txt'), delimiter='\t',header=None) for cell_line in cell_lines], axis = 1)
        return x_pos_read, x_neg_read 

    def label_binary(self,x_pos_read,x_neg_read):
        y_pos= np.ones(x_pos_read.shape[0])
        y_pos= np.array(y_pos)
        y_neg= np.zeros(x_neg_read.shape[0])
        y_neg= np.array(y_neg)
        return y_pos, y_neg

    def label_cont(self,y_path):
        y_pos_file = os.path.join(y_path,'y.pos.txt')
        y_neg_file = os.path.join(y_path,'y.neg.txt')
        y_pos = pd.read_csv(y_pos_file, sep="\t",header=None)
        y_neg = pd.read_csv(y_neg_file, sep="\t",header=None)
        y_pos = np.squeeze(np.array(y_pos))
        y_neg = np.squeeze(np.array(y_neg))
        return y_pos, y_neg

    def combine_pos_neg(self, x_pos_seq, x_pos_peak, x_pos_read, x_neg_seq, x_neg_peak, x_neg_read, y1_pos, y1_neg,y2_pos, y2_neg):
        x_seq = np.concatenate((x_pos_seq,x_neg_seq),axis=0)
        x_seq = np.swapaxes(x_seq,2,1)
        x_read = np.concatenate((x_pos_read,x_neg_read),axis=0)
        x_peak = pd.concat((x_pos_peak,x_neg_peak),axis=0)
        y1 = np.concatenate((y1_pos,y1_neg),axis=0)
        y2 = np.concatenate((y2_pos,y2_neg),axis=0)

        np.random.seed(1234)
        indices = np.arange(len(y1))
        indices = np.random.permutation(indices)

        y1 = y1[indices]
        y2 = y2[indices]
        x_seq = x_seq[indices]
        x_read = x_read[indices]
        x_peak = x_peak.iloc[indices]
        return x_seq, x_read, x_peak, y1, y2

    def split_train_test_val(self, x_peak):
        testId = np.where(x_peak['seqnames'].isin(['chr8','chr9']))[0]
        valId = np.where(x_peak['seqnames'].isin(['chr7']))[0]
        trainId = np.where(~x_peak['seqnames'].isin(['chr7','chr8','chr9']))[0]
        return trainId, valId, testId

    def save_to_hdf5(self, h5_file, x_seq, x_read, y1, y2, trainId, valId, testId):
        with h5py.File(h5_file,'w') as f:
            f.create_dataset('x_seq',data=x_seq,compression='gzip')
            f.create_dataset('x_read',data=x_read,compression='gzip')
            f.create_dataset('y1',data=y1,compression='gzip')
            f.create_dataset('y2',data=y2,compression='gzip')
            f.create_dataset('trainId',data=trainId,compression='gzip')
            f.create_dataset('valId',data=valId,compression='gzip')
            f.create_dataset('testId',data=testId,compression='gzip')

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

class MyDataset_gene(MyDataLoader):
    def __init__(self,batch_size,histones):
        self.batch_size = batch_size
        self.histones = histones

        self.peaks_path = '../data/gene/genes'
        self.seqs_path = '../data/gene/seqs'

        # create hdf5 for each histone
        self.create_hdf5(histones)

        self.test_dataset = []
        self.read_size = []

        for histone in histones:
            h5_file = f'../h5/{histone}_gene.h5'
            x_seq, x_read = self.load_from_hdf5(h5_file)
            self.test_dataset.append(torch.from_numpy(x_read))
            self.read_size.append(x_read.shape)

        self.test_dataset.append(torch.from_numpy(x_seq))


    def create_hdf5(self,histones):
        if not os.path.exists('../h5'):
            os.makedirs('../h5')

        for histone in histones:
            reads_path = f'../data/gene/{histone}'
            h5_file = f'../h5/{histone}_gene.h5'

            if os.path.exists(h5_file):
                print(h5_file+' exists and load data directly.')

            else:
                print(h5_file+' does not exist and is being created.')
                x_pos_seq = self.load_seqs(self.seqs_path)
                print('seqs loaded')
                x_pos_read = self.load_reads(reads_path)
                print('reads loaded')
                self.save_to_hdf5(h5_file,x_pos_seq,x_pos_read)

    def load_from_hdf5(self,h5_file):
        with h5py.File(h5_file,'r') as f:
            x_seq = f.get('x_seq')[:]
            x_read = f.get('x_read')[:]
        return x_seq, x_read

    def load_seqs(self,seqs_path):
        seq_file = os.path.join(seqs_path,'seqs.genes.pos.allchr.fasta')
        x_seq = self.onehot(seq_file)
        x_seq = np.swapaxes(x_seq,2,1)
        return x_seq

    def onehot(self,fafile):
        x=[]
        for seq_record in SeqIO.parse(fafile, "fasta"):
            seq_array = np.array(list(seq_record.seq))
            label_encoder = LabelEncoder()
            integer_encoded_seq = label_encoder.fit_transform(seq_array)
            onehot_encoder = OneHotEncoder(sparse_output=False)
            integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
            onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
            x.append(onehot_encoded_seq)        
        x = np.array(x)
        return x  

    def load_reads(self,reads_path):
        readlists = os.listdir(reads_path)
        readlists = [file for file in readlists if file.startswith('reads')]
        cell_lines = np.arange(1,1+len(readlists)/2).astype(int)
        x_read = np.stack([pd.read_csv(os.path.join(reads_path,f'reads.genes.pos.allchr.{cell_line}.txt'), delimiter='\t',header=None) for cell_line in cell_lines], axis = 1)
        return x_read

    def save_to_hdf5(self, h5_file, x_seq, x_read):
        with h5py.File(h5_file,'w') as f:
            f.create_dataset('x_seq',data=x_seq,compression='gzip')
            f.create_dataset('x_read',data=x_read,compression='gzip')

    def get_dataloader(self,dstype):
        shuffle = False
        if dstype == 'test':
            dataset = TensorDataset(*self.test_dataset)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

class MyDataset_diff(MyDataLoader):
    def __init__(self,batch_size,histones):
        self.histones = histones
        self.batch_size = batch_size
        seqs_path = os.path.join('../data/diff/seqs')
        reads_path = os.path.join('../data/diff/reads')
        peaks_path = os.path.join('../data/diff/peaks')
        h5_file = f'../h5/{self.histones}_diff.h5'

        if not os.path.exists('../h5/'):
            os.makedirs('../h5/')

        if os.path.exists(h5_file):
            self.x_seq, self.x_read_dif,self.x_read_ADCT, self.y, self.trainId, self.valId, self.testId = self.load_from_hdf5(h5_file)
            print(h5_file+' exists and load data directly.')

        else:
            print(h5_file+' does not exist and is being created.')
            x_pos_seq, x_neg_seq = self.load_seqs(seqs_path)
            print('seqs loaded')
            x_pos_peak, x_neg_peak = self.load_peaks(peaks_path)
            print('peaks loaded')
            x_pos_read_dif, x_neg_read_dif = self.load_read(reads_path,'dif')
            x_pos_read_AD, x_neg_read_AD = self.load_read(reads_path,'AD')
            x_pos_read_CT, x_neg_read_CT = self.load_read(reads_path,'CT')
            print('reads loaded')
            y_pos, y_neg = self.label(x_pos_peak,x_neg_peak)
            self.x_seq, self.x_read_dif,self.x_read_ADCT, self.x_peak, self.y = self.combine_pos_neg(x_pos_seq,x_pos_peak,x_pos_read_dif,x_pos_read_AD,x_pos_read_CT,x_neg_seq,x_neg_peak,x_neg_read_dif,x_neg_read_AD,x_neg_read_CT,y_pos,y_neg)
            self.trainId, self.valId, self.testId = self.split_train_test_val(self.x_peak)
            self.save_to_hdf5(h5_file,self.x_seq,self.x_read_dif,self.x_read_ADCT,self.y,self.trainId,self.valId,self.testId)

            self.x_peak.to_csv(os.path.join(peaks_path,'x_peak.csv'),index=False)

        self.read_size = [self.x_read_dif.shape,self.x_read_ADCT.shape]

    def load_from_hdf5(self,h5_file):
        with h5py.File(h5_file,'r') as f:
            x_seq = f.get('x_seq')[:]
            x_read_dif = f.get('x_read_dif')[:]
            x_read_ADCT = f.get('x_read_ADCT')[:]
            y = f.get('y')[:]
            testId = f.get('testId')[:]
            trainId = f.get('trainId')[:]
            valId = f.get('valId')[:]
        return x_seq, x_read_dif,x_read_ADCT, y, trainId, valId, testId

    def load_seqs(self,seqs_path):
        seq_pos_file = os.path.join(seqs_path,f'seqs.pos.allchr.fasta')
        seq_neg_file = os.path.join(seqs_path,f'seqs.neg.allchr.fasta')
        x_pos_seq = self.onehot(seq_pos_file)
        x_neg_seq = self.onehot(seq_neg_file)
        return x_pos_seq, x_neg_seq

    def onehot(self,fafile):
        x=[]
        for seq_record in SeqIO.parse(fafile, "fasta"):
            seq_array = np.array(list(seq_record.seq))
            label_encoder = LabelEncoder()
            integer_encoded_seq = label_encoder.fit_transform(seq_array)
            onehot_encoder = OneHotEncoder(sparse_output=False,categories=[[0,1,2,3]])
            integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
            onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
            x.append(onehot_encoded_seq)        
        x = np.array(x)
        return x  

    def load_peaks(self,peaks_path):
        peak_pos_file = os.path.join(peaks_path,f'peaks.pos.allchr.txt')
        peak_neg_file = os.path.join(peaks_path,f'peaks.neg.allchr.txt')
        x_pos_peak = pd.read_csv(peak_pos_file, sep="\t")
        x_neg_peak = pd.read_csv(peak_neg_file, sep="\t")
        return x_pos_peak, x_neg_peak

    def load_read(self,reads_path,group):
        readlists = os.listdir(reads_path)
        readlists = [file for file in readlists if file.endswith('.txt')]
        x_pos_read = np.concatenate([pd.read_csv(os.path.join(reads_path,f'{group}.reads.pos.allchr.{histone}.txt'), delimiter='\t',header=None) for histone in self.histones],axis=1)
        x_neg_read = np.concatenate([pd.read_csv(os.path.join(reads_path,f'{group}.reads.neg.allchr.{histone}.txt'), delimiter='\t',header=None) for histone in self.histones],axis=1)
        return x_pos_read, x_neg_read

    def label(self,x_pos_peak,x_neg_peak):
        y_pos= np.ones(x_pos_peak.shape[0])
        y_pos= np.array(y_pos)
        y_neg= np.zeros(x_neg_peak.shape[0])
        y_neg= np.array(y_neg)
        return y_pos, y_neg

    def combine_pos_neg(self,x_pos_seq,x_pos_peak,x_pos_read_dif,x_pos_read_AD,x_pos_read_CT,x_neg_seq,x_neg_peak,x_neg_read_dif,x_neg_read_AD,x_neg_read_CT,y_pos,y_neg):
        x_seq = np.concatenate((x_pos_seq,x_neg_seq),axis=0)
        x_seq = np.swapaxes(x_seq,2,1)

        x_peak = pd.concat((x_pos_peak,x_neg_peak),axis=0)

        x_read_dif = np.concatenate((x_pos_read_dif,x_neg_read_dif),axis=0)
        x_read_dif = x_read_dif[:,np.newaxis,:]

        x_pos_read_ADCT = np.stack([x_pos_read_AD,x_pos_read_CT],axis=1)
        x_neg_read_ADCT = np.stack([x_neg_read_AD,x_neg_read_CT],axis=1)
        x_read_ADCT = np.concatenate((x_pos_read_ADCT,x_neg_read_ADCT),axis=0)

        y = np.concatenate((y_pos,y_neg),axis=0)

        np.random.seed(1234)
        indices = np.arange(len(y))
        indices = np.random.permutation(indices)

        y = y[indices]
        x_seq = x_seq[indices]
        x_read_dif = x_read_dif[indices]
        x_read_ADCT = x_read_ADCT[indices]
        x_peak = x_peak.iloc[indices]

        return x_seq, x_read_dif, x_read_ADCT, x_peak, y

    def split_train_test_val(self, x_peak):
        testId = np.where(x_peak['seqnames'].isin(['chr8','chr9']))[0]
        valId = np.where(x_peak['seqnames'].isin(['chr7']))[0]
        trainId = np.where(~x_peak['seqnames'].isin(['chr7','chr8','chr9']))[0]
        return trainId, valId, testId

    def save_to_hdf5(self, h5_file, x_seq, x_read_dif,x_read_ADCT, y, trainId, valId, testId):
        with h5py.File(h5_file,'w') as f:
            f.create_dataset('x_seq',data=x_seq,compression='gzip')
            f.create_dataset('x_read_dif',data=x_read_dif,compression='gzip')
            f.create_dataset('x_read_ADCT',data=x_read_ADCT,compression='gzip')
            f.create_dataset('y',data=y,compression='gzip')
            f.create_dataset('trainId',data=trainId,compression='gzip')
            f.create_dataset('valId',data=valId,compression='gzip')
            f.create_dataset('testId',data=testId,compression='gzip')

    def get_dataloader(self,dstype): 
        shuffle = False
        if dstype == 'train':
            indices = self.trainId
            shuflle = True
        elif dstype =='val':
            indices = self.valId
        elif dstype == 'test':
            indices = self.testId
        else:
            raise ValueError("dstype must be one of ['train','val', 'test']")

        x_seq = self.x_seq[indices]
        x_read_dif = self.x_read_dif[indices]
        x_read_ADCT = self.x_read_ADCT[indices]
        y = self.y[indices]

        dataset = TensorDataset(torch.from_numpy(x_read_dif), torch.from_numpy(x_read_ADCT),torch.from_numpy(x_seq),torch.from_numpy(y).long())
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)