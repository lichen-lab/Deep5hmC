import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau,LambdaLR
import matplotlib.pyplot as plt
import time
import os

class Trainer():
    def __init__(self,num_epochs, lr, earlystop_thresh, warmup_epochs=10):
        self.num_epochs = num_epochs
        self.lr = lr
        self.earlystop_thresh = earlystop_thresh
        #self.warmup_epochs = warmup_epochs
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lossfun_cls = nn.NLLLoss()
        self.lossfun_reg = nn.MSELoss()
        self.gpu_cnt = torch.cuda.device_count()
        
        self.H = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        
        if not os.path.exists('../weights/'):
            os.makedirs('../weights/')
        
    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        
        self.trainSteps = len(self.train_dataloader.dataset) // data.batch_size
        self.valSteps = len(self.val_dataloader.dataset) // data.batch_size
        
    def prepare_model(self, model):
        model.trainer = self
        if self.gpu_cnt >=1:
            model.to(self.device)
            if self.gpu_cnt > 1:
                print('[INFO] using multiple GPUs')
                model = nn.DataParallel(model).module
            else:
                print('[INFO] using GPU')
        self.model = model
        
    def warmup(self,epoch):
        if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
        else:
            return 1.0    
    

    def fit_binary(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = ReduceLROnPlateau(self.optim, verbose=True,patience=30)
        # self.warmup_scheduler = LambdaLR(self.optim, lr_lambda=self.warmup) 
        
        best_loss = 1e9
        best_epoch = -1
        
        print("[INFO] training the network...")
        startTime = time.time()
        

        for e in range(0,self.num_epochs):
            self.model.train()
            totalTrainLoss = 0
            totalValLoss = 0
            trainCorrect = 0
            valCorrect = 0

            #training
            for (readx1,readx2,seqx, y) in self.train_dataloader:
                (readx1,readx2,seqx, y) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float),y.to(self.device))
                self.pred = self.model(readx1,readx2,seqx)
                self.loss = self.lossfun_cls(self.pred, y)
                self.optim.zero_grad()
                self.loss.backward()
                self.optim.step()
                totalTrainLoss += self.loss
                trainCorrect += (self.pred.argmax(1) == y).type(torch.float).sum().item()

            #validation     
            with torch.no_grad():
                self.model.eval()
                for (readx1,readx2,seqx, y) in self.val_dataloader:
                    (readx1,readx2,seqx, y) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float),y.to(self.device))
                    self.pred = self.model(readx1,readx2,seqx)
                    totalValLoss += self.lossfun_cls(self.pred, y)
                    valCorrect += (self.pred.argmax(1) == y).type(torch.float).sum().item()

            self.avgTrainLoss = totalTrainLoss / self.trainSteps
            self.avgValLoss = totalValLoss / self.valSteps
            self.trainCorrect = trainCorrect / len(self.train_dataloader.dataset)
            self.valCorrect = valCorrect / len(self.val_dataloader.dataset)

            self.H["train_loss"].append(self.avgTrainLoss.cpu().detach().numpy())
            self.H["train_acc"].append(self.trainCorrect)
            self.H["val_loss"].append(self.avgValLoss.cpu().detach().numpy())
            self.H["val_acc"].append(self.valCorrect)

            if e%5==0:
                print("[INFO] EPOCH: {}/{}".format(e + 1, self.num_epochs))
                print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(self.avgTrainLoss, self.trainCorrect))
                print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(self.avgValLoss, self.valCorrect))

            #early stopping
            if self.avgValLoss < best_loss:
                best_loss = self.avgValLoss
                best_epoch = e
                self.best_model = f"../weights/{self.model.model_name}.pth"
                torch.save(self.model.state_dict(), f"../weights/{self.model.model_name}.pth")
            elif e - best_epoch > self.earlystop_thresh:
                print("Early stopped training at epoch %d" % e)
                break
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))     
        
        
    def fit_cont(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = ReduceLROnPlateau(self.optim, verbose=True,patience=30)
        # self.warmup_scheduler = LambdaLR(self.optim, lr_lambda=self.warmup) 
        
        best_loss = 1e9
        best_epoch = -1
        
        print("[INFO] training the network...")
        startTime = time.time()
        
        for e in range(0,self.num_epochs):
            self.model.train()
            totalTrainLoss = 0
            totalValLoss = 0

            #training
            for (readx1,readx2,seqx, y) in self.train_dataloader:
                (readx1,readx2,seqx, y) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float),y.to(self.device,dtype=torch.float))
                self.pred = self.model(readx1,readx2,seqx)
                self.loss = self.lossfun_reg(self.pred, y.unsqueeze(1))
                self.optim.zero_grad()
                self.loss.backward()
                self.optim.step()
                totalTrainLoss += self.loss
        # if e <= self.warmup_epochs:
        #     self.warmup_scheduler.step()
        # print(self.optim.param_groups[0]['lr'])


            #validation     
            with torch.no_grad():
                self.model.eval()
                for (readx1,readx2,seqx, y) in self.val_dataloader:
                    (readx1,readx2,seqx, y) = (readx1.to(self.device,dtype=torch.float),readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float),y.to(self.device,dtype=torch.float))
                    self.pred = self.model(readx1,readx2,seqx)
                    totalValLoss += self.lossfun_reg(self.pred, y.unsqueeze(1))
                # self.scheduler.step(totalValLoss)

            self.avgTrainLoss = totalTrainLoss / self.trainSteps
            self.avgValLoss = totalValLoss / self.valSteps

            self.H["train_loss"].append(self.avgTrainLoss.cpu().detach().numpy())
            self.H["val_loss"].append(self.avgValLoss.cpu().detach().numpy())

            if e%5==0:
                print("[INFO] EPOCH: {}/{}".format(e + 1, self.num_epochs))
                print("Train loss: {:.6f}".format(self.avgTrainLoss))
                print("Val loss: {:.6f}".format(self.avgValLoss))

            #early stopping
            if self.avgValLoss < best_loss:
                best_loss = self.avgValLoss
                best_epoch = e
                self.best_model = f"../weights/{self.model.model_name}.pth"
                torch.save(self.model.state_dict(), f"../weights/{self.model.model_name}.pth")
            elif e - best_epoch > self.earlystop_thresh:
                print("Early stopped training at epoch %d" % e)
                break
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))    

    def fit_diff(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = ReduceLROnPlateau(self.optim, verbose=True,patience=30)
        # self.warmup_scheduler = LambdaLR(self.optim, lr_lambda=self.warmup) 
        
        best_loss = 1e9
        best_epoch = -1
        
        print("[INFO] training the network...")
        startTime = time.time()
        

        for e in range(0,self.num_epochs):
            self.model.train()
            totalTrainLoss = 0
            totalValLoss = 0
            trainCorrect = 0
            valCorrect = 0

            #training
            for (readx1,readx2,seqx, y) in self.train_dataloader:
                (readx1,readx2,seqx, y) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float),y.to(self.device))
                self.pred = self.model(readx1,readx2)
                self.loss = self.lossfun_cls(self.pred, y)
                self.optim.zero_grad()
                self.loss.backward()
                self.optim.step()
                totalTrainLoss += self.loss
                trainCorrect += (self.pred.argmax(1) == y).type(torch.float).sum().item()

            #validation     
            with torch.no_grad():
                self.model.eval()
                for (readx1,readx2,seqx, y) in self.val_dataloader:
                    (readx1,readx2,seqx, y) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float),y.to(self.device))
                    self.pred = self.model(readx1,readx2)
                    totalValLoss += self.lossfun_cls(self.pred, y)
                    valCorrect += (self.pred.argmax(1) == y).type(torch.float).sum().item()

            self.avgTrainLoss = totalTrainLoss / self.trainSteps
            self.avgValLoss = totalValLoss / self.valSteps
            self.trainCorrect = trainCorrect / len(self.train_dataloader.dataset)
            self.valCorrect = valCorrect / len(self.val_dataloader.dataset)

            self.H["train_loss"].append(self.avgTrainLoss.cpu().detach().numpy())
            self.H["train_acc"].append(self.trainCorrect)
            self.H["val_loss"].append(self.avgValLoss.cpu().detach().numpy())
            self.H["val_acc"].append(self.valCorrect)

            if e%5==0:
                print("[INFO] EPOCH: {}/{}".format(e + 1, self.num_epochs))
                print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(self.avgTrainLoss, self.trainCorrect))
                print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(self.avgValLoss, self.valCorrect))

            #early stopping
            if self.avgValLoss < best_loss:
                best_loss = self.avgValLoss
                best_epoch = e
                self.best_model = f"../weights/{self.model.model_name}.pth"
                torch.save(self.model.state_dict(), f"../weights/{self.model.model_name}.pth")
            elif e - best_epoch > self.earlystop_thresh:
                print("Early stopped training at epoch %d" % e)
                break
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))     
                
    def plot(self):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(self.H["train_loss"], label="train_loss")
        plt.plot(self.H["val_loss"], label="val_loss")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        


                
