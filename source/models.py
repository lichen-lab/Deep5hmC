import torch
import torch.nn.functional as F
import torch.nn as nn


#########################################################################################
#########################################################################################
# Modules
#########################################################################################
#########################################################################################
class Seqs_CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(53*256, 512)
        self.Linear2 = nn.Linear(512, 256)
    
    def forward(self, x): 
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop(x)
        
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop(x)
        
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.Drop(x)
        
        x = x.view(-1, 53*256)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Drop(x)  
        
        x = self.Linear2(x)
        x = F.relu(x)
        x = self.Drop(x)
        
        return x


#########################################################################################
class Reads_CNN1(nn.Module):
    def __init__(self,input_read_size):
        super().__init__()
        
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        MaxPool_kernel = (2,2) if input_read_size[1]>3 else (1,2)
        
        self.Maxpool = nn.MaxPool2d(kernel_size=MaxPool_kernel, stride=2)
        self.Drop = nn.Dropout(p=0.5)
        
        linear_input_size = self.linear_input_size(input_read_size)
        
        self.Linear1 = nn.Linear(linear_input_size, 256)

    def forward(self,xx): 
        xx = xx.unsqueeze(1)
        xx = self.Conv1(xx)
        xx = F.relu(xx)
        xx = self.Maxpool(xx)
        xx = self.Drop(xx)
        
        xx = self.Conv2(xx)
        xx = F.relu(xx)
        xx = self.Maxpool(xx)
        xx = self.Drop(xx)
        
        xx = self.Conv3(xx)
        xx = F.relu(xx)
        xx = self.Drop(xx)
        
        xx = xx.view(xx.size(0),-1)
        xx = self.Linear1(xx)
        xx = F.relu(xx)
        xx = self.Drop(xx)  
        
        return xx
    
    def linear_input_size(self,input_read_size):
        xx = torch.empty(1,1,input_read_size[1],input_read_size[2])
        xx = self.Conv1(xx)
        xx = self.Maxpool(xx)
        
        xx = self.Conv2(xx)
        xx = self.Maxpool(xx)
        
        xx = self.Conv3(xx)
        
        xx = xx.view(xx.size(0),-1)
        return xx.shape[1]
    
#########################################################################################    
class Seqs_CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=8)
        self.Conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)
        self.Conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.Drop = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(53*256, 512)
        self.Linear2 = nn.Linear(512, 256)
        self.BatchNorm1 = nn.BatchNorm1d(64)
        self.BatchNorm2 = nn.BatchNorm1d(128)
        self.BatchNorm3 = nn.BatchNorm1d(256)
        self.BatchNorm4 = nn.BatchNorm1d(512)
        self.BatchNorm5 = nn.BatchNorm1d(256)
    
    def forward(self, x): 
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.BatchNorm1(x)
        x = self.Maxpool(x)
        x = self.Drop(x)
        
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.BatchNorm2(x)
        x = self.Maxpool(x)
        x = self.Drop(x)
        
        x = self.Conv3(x)
        x = F.relu(x)
        x = self.BatchNorm3(x)
        x = self.Drop(x)
        
        x = x.view(-1, 53*256)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.BatchNorm4(x)
        x = self.Drop(x)  
        
        x = self.Linear2(x)
        x = F.relu(x)
        x = self.BatchNorm5(x)
        x = self.Drop(x)
        
        return x
     
#########################################################################################       
class Reads_CNN2(nn.Module):
    def __init__(self,input_read_size):
        super().__init__()
        
        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        MaxPool_kernel = (2,2) if input_read_size[1]>3 else (1,2)
        
        self.Maxpool = nn.MaxPool2d(kernel_size=MaxPool_kernel, stride=2)
        self.Drop = nn.Dropout(p=0.5)
        
        linear_input_size = self.linear_input_size(input_read_size)
        
        self.Linear1 = nn.Linear(linear_input_size, 256)
        
        self.BatchNorm1 = nn.BatchNorm2d(32)
        self.BatchNorm2 = nn.BatchNorm2d(64)
        self.BatchNorm3 = nn.BatchNorm2d(128)
        self.BatchNorm4 = nn.BatchNorm1d(256)

    def forward(self,xx): 
        xx = xx.unsqueeze(1)
        xx = self.Conv1(xx)
        xx = F.relu(xx)
        xx = self.BatchNorm1(xx)
        xx = self.Maxpool(xx)
        xx = self.Drop(xx)
        
        xx = self.Conv2(xx)
        xx = F.relu(xx)
        xx = self.BatchNorm2(xx)
        xx = self.Maxpool(xx)
        xx = self.Drop(xx)
        
        xx = self.Conv3(xx)
        xx = F.relu(xx)
        xx = self.BatchNorm3(xx)
        xx = self.Drop(xx)
        
        xx = xx.view(xx.size(0),-1)
        xx = self.Linear1(xx)
        xx = F.relu(xx)
        xx = self.BatchNorm4(xx)
        xx = self.Drop(xx)  
        
        return xx
    
    def linear_input_size(self,input_read_size):
        xx = torch.empty(1,1,input_read_size[1],input_read_size[2])
        xx = self.Conv1(xx)
        xx = self.Maxpool(xx)
        
        xx = self.Conv2(xx)
        xx = self.Maxpool(xx)
        
        xx = self.Conv3(xx)
        
        xx = xx.view(xx.size(0),-1)
        return xx.shape[1]
#########################################################################################    
class MFBLayer(nn.Module):
    def __init__(self,n1,n2,mfb_out_dim,mfb_factor_num):
        super().__init__()
        self.mfb_out_dim = mfb_out_dim
        self.mfb_factor_num =mfb_factor_num
        joint_emb_size = mfb_out_dim*mfb_factor_num
        self.Drop = nn.Dropout(p=0.5)
        self.Linear1 = nn.Linear(n1, joint_emb_size)
        self.Linear2 = nn.Linear(n2, joint_emb_size)
        self.Drop = nn.Dropout(p=0.5)
        
    def forward(self,x,xx):
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Drop(x)
        
        xx = self.Linear2(xx)
        xx = F.relu(xx)
        xx = self.Drop(xx)
        
        xcomb = torch.mul(x,xx)
        xcomb = self.Drop(xcomb)
        xcomb = xcomb.view(xcomb.shape[0],1,self.mfb_out_dim,self.mfb_factor_num)
        xcomb = torch.sum(xcomb,3,keepdim=True)
        xcomb = torch.squeeze(xcomb)
        xcomb = torch.sqrt(F.relu(xcomb)-torch.sqrt(F.relu(-xcomb)))
        xcomb = F.normalize(xcomb)
        return xcomb

#########################################################################################
class Deep5hmC_binary(nn.Module):
    def __init__(self,input_read_size,input_dim1=256,input_dim2=256,mfb_out_dim=512,mfb_factor_num=10):
        super(Deep5hmC_binary, self).__init__()
        self.model_name = 'Deep5hmC_binary'
        
        #seqs module  
        self.Seqs_CNN = Seqs_CNN1()
        
        #read module
        self.Reads_CNN1 = Reads_CNN1(input_read_size=input_read_size[0])
        self.Reads_CNN2 = Reads_CNN1(input_read_size=input_read_size[1])
        
        #combine module
        self.compact_bilinear_pooling = MFBLayer(input_dim1,input_dim2*2,mfb_out_dim,mfb_factor_num)
        
        #predict module
        self.combined_fc1 = nn.Linear(mfb_out_dim, 128)
        self.combined_fc2 = nn.Linear(128, 2)
        self.Drop = nn.Dropout(p=0.5)        
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self,read1,read2,seq):
        # seqs module
        x = self.Seqs_CNN(seq)
        
        # read module
        xx1 = self.Reads_CNN1(read1)
        xx2 = self.Reads_CNN2(read2)
        #combine reads
        xx = torch.cat((xx1,xx2), 1)
        
        # combine module
        xcomb = self.compact_bilinear_pooling(x,xx)
        
        # predict module
        xcomb = self.combined_fc1(xcomb)
        xcomb = F.relu(xcomb)
        xcomb = self.Drop(xcomb)
        xcomb = self.combined_fc2(xcomb)
        xcomb = self.logSoftmax(xcomb)
        
        return xcomb
    
#########################################################################################
class Deep5hmC_cont(nn.Module):
    def __init__(self,input_read_size,input_dim1=256,input_dim2=256,mfb_out_dim=512,mfb_factor_num=10):
        super(Deep5hmC_cont, self).__init__()
        self.model_name = 'Deep5hmC_cont'
        
        #seqs module  
        self.Seqs_CNN = Seqs_CNN2()
        
        #read module
        self.Reads_CNN1 = Reads_CNN2(input_read_size=input_read_size[0])
        self.Reads_CNN2 = Reads_CNN2(input_read_size=input_read_size[1])
        
        #combine module
        self.compact_bilinear_pooling = MFBLayer(input_dim1,input_dim2*2,mfb_out_dim,mfb_factor_num)
        
        #predict module
        self.combined_fc1 = nn.Linear(mfb_out_dim, 128)
        self.combined_fc2 = nn.Linear(128, 1)
        self.Drop = nn.Dropout(p=0.5)        

    def forward(self,read1,read2,seq):
        # seqs module
        x = self.Seqs_CNN(seq)
        
        # read module
        xx1 = self.Reads_CNN1(read1)
        xx2 = self.Reads_CNN2(read2)
        #combine reads
        xx = torch.cat((xx1,xx2), 1)
        
        # combine module
        xcomb = self.compact_bilinear_pooling(x,xx)
        
        # predict module
        xcomb = self.combined_fc1(xcomb)
        xcomb = F.relu(xcomb)
        xcomb = self.Drop(xcomb)
        xcomb = self.combined_fc2(xcomb)
        
        return xcomb
    
#########################################################################################
class Deep5hmC_gene(nn.Module):
    def __init__(self,input_read_size,input_dim1=256,input_dim2=256,mfb_out_dim=512,mfb_factor_num=10):
        super(Deep5hmC_gene, self).__init__()
        self.model_name = 'Deep5hmC_gene'
        
        #seqs module  
        self.Seqs_CNN = Seqs_CNN2()
        
        #read module
        self.Reads_CNN1 = Reads_CNN2(input_read_size=input_read_size[0])
        self.Reads_CNN2 = Reads_CNN2(input_read_size=input_read_size[1])
        
        #combine module
        self.compact_bilinear_pooling = MFBLayer(input_dim1,input_dim2*2,mfb_out_dim,mfb_factor_num)
        
        #predict module
        self.combined_fc1 = nn.Linear(mfb_out_dim, 128)
        self.combined_fc2 = nn.Linear(128, 1)
        self.Drop = nn.Dropout(p=0.5)        

    def forward(self,read1,read2,seq):
        # seqs module
        x = self.Seqs_CNN(seq)
        
        # read module
        xx1 = self.Reads_CNN1(read1)
        xx2 = self.Reads_CNN2(read2)
        #combine reads
        xx = torch.cat((xx1,xx2), 1)
        
        # combine module
        xcomb = self.compact_bilinear_pooling(x,xx)
        
        # predict module
        xcomb = self.combined_fc1(xcomb)
        xcomb = F.relu(xcomb)
        xcomb = self.Drop(xcomb)
        xcomb = self.combined_fc2(xcomb)
        
        return xcomb    
#########################################################################################
class Deep5hmC_diff(nn.Module):
    def __init__(self,input_read_size):
        super(Deep5hmC_diff, self).__init__()
        self.model_name = 'Deep5hmC_diff'
        
        #read module
        self.Reads_CNN1 = Reads_CNN1(input_read_size=input_read_size[0])
        self.Reads_CNN2 = Reads_CNN1(input_read_size=input_read_size[1])
        
        #predict module
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 2)
        self.Drop = nn.Dropout(p=0.5)        
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self,read1,read2):
        # read module
        xx1 = self.Reads_CNN1(read1)
        xx2 = self.Reads_CNN2(read2)

        #combine reads
        xx = torch.cat((xx1,xx2), 1)
        
        # predict module
        xx = self.fc1(xx)
        xx = F.relu(xx)
        xx = self.Drop(xx)
        xx = self.fc2(xx)
        xx = self.logSoftmax(xx)
        
        return xx
