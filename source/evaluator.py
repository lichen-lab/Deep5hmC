import torch
import numpy as np
from sklearn.metrics import roc_curve,auc,f1_score,recall_score,precision_score,accuracy_score,precision_recall_curve, average_precision_score,mean_squared_error
from scipy.stats import wilcoxon,pearsonr,spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import os


class Evaluator():
    def __init__(self,model,best_model):
        print("[INFO] resume best model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(best_model))
        self.model = model.to(self.device)
        
    def prepare_data(self, data):
        self.test_dataloader = data.test_dataloader()
            
    def prediction_binary(self,data):
        self.prepare_data(data)
        
        print("[INFO] evaluating network...")
        with torch.no_grad():

            self.model.eval()

            preds = []
            predsProb = []
            ys = []

            for (readx1,readx2,seqx, y) in self.test_dataloader:
                (readx1,readx2,seqx, y) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float),y.to(self.device))
                pred = self.model(readx1,readx2,seqx)
                preds.extend(pred.argmax(axis=1).cpu().numpy())
                predsProb.extend(pred[:,1].cpu().numpy())
                ys.extend(y.cpu().numpy())

            self.preds = np.array(preds)
            self.predsProb = np.array(predsProb)
            self.ys = np.array(ys)


            pred_res = pd.DataFrame({'y_prob' : self.predsProb,
                        'y_pred' : self.preds,
                        'y' : self.ys})

            if not os.path.exists('../pred'):
                os.makedirs('../pred')
            pred_res.to_csv(f'../pred/pred_binary.csv',index=False)
            
    def prediction_cont(self,data):
        self.prepare_data(data)           
        
        print("[INFO] evaluating network...")
        with torch.no_grad():

            self.model.eval()

            preds = []
            predsProb = []
            ys = []

            for (readx1,readx2,seqx, y) in self.test_dataloader:
                (readx1,readx2,seqx, y) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float),y.to(self.device,dtype=torch.float))
                pred = self.model(readx1,readx2,seqx)
                preds.extend(pred.cpu().numpy())
                ys.extend(y.cpu().numpy())

            self.preds = np.squeeze(np.array(preds))
            self.ys = np.array(ys)

            pred_res = pd.DataFrame({'y_pred' : self.preds,
                                     'y' : self.ys})

            if not os.path.exists('../pred'):
                os.makedirs('../pred')
            pred_res.to_csv(f'../pred/pred_cont.csv',index=False)
            
        
    def prediction_gene(self,data):
        self.prepare_data(data)
        
        print("[INFO] evaluating network...")
        with torch.no_grad():
            
            self.model.eval()
            
            preds = []

            for (readx1,readx2,seqx) in self.test_dataloader:
                (readx1,readx2,seqx) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float))
                pred = self.model(readx1,readx2,seqx)
                preds.extend(pred.cpu().numpy())
            
            self.preds = np.squeeze(np.array(preds))
            
        observed = ['5hmC','RNA']
        
        for iobserved in range(len(observed)):
            observed_path = f'../data/gene/{observed[iobserved]}_gene'
            observed_counts_file = os.path.join(observed_path,'counts.gene.txt')
            y_df = pd.read_csv(observed_counts_file,delimiter='\t')
            y = np.array(np.log(y_df.sum(axis=1)+1))

            genes_file = os.path.join(observed_path,'genes.txt')
            genes = pd.read_csv(genes_file,delimiter='\t')
            genes['y'] = y

            gene_cut_path = os.path.join('../data/gene/genes')
            genes_cut_file = os.path.join(gene_cut_path,'genes.pos.allchr.txt')
            genes_cut = pd.read_csv(genes_cut_file,delimiter='\t')

            genes_cut['y_pred'] = np.exp(preds)
            genes_pred = genes_cut.groupby('gene_id')['y_pred'].sum().reset_index()
            genes_pred['y_pred'] = np.log(genes_pred.y_pred+1)
            pred_res = pd.merge(genes_pred,genes,left_on='gene_id',right_on='genes.gene_id',how='inner')
            pred_res = pred_res.loc[:,['gene_id','y_pred','y']]
            pred_res = pred_res.replace([np.inf, -np.inf], np.nan).dropna(how='any', axis=0, subset=None)

            if not os.path.exists('../pred'):
                os.makedirs('../pred')
            pred_res.to_csv(f'../pred/pred_gene_{observed[iobserved]}.csv',index=False)            
            
    
    def prediction_diff(self,data):
        self.prepare_data(data)
        
        print("[INFO] evaluating network...")
        with torch.no_grad():
              
            self.model.eval()
            
            preds = []
            predsProb = []
            ys = []

            for (readx1,readx2,seqx, y) in data.test_dataloader():
                (readx1,readx2,seqx, y) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float),y.to(self.device))
                pred = self.model(readx1,readx2)
                preds.extend(pred.argmax(axis=1).cpu().numpy())
                predsProb.extend(pred[:,1].cpu().numpy())
                ys.extend(y.cpu().numpy())

            self.preds = np.array(preds)
            self.predsProb = np.array(predsProb)
            self.ys = np.array(ys)
            

            pred_res = pd.DataFrame({'y_prob' : self.predsProb,
                        'y_pred' : self.preds,
                        'y' : self.ys})
            
            if not os.path.exists('../pred'):
                os.makedirs('../pred')
            pred_res.to_csv(f'../pred/pred_diff.csv',index=False)            
            

            
    def eval_model_binary(self,data,verbose=0):
        self.prediction_binary(data)
        
            
        pred_res = pd.read_csv(f'../pred/pred_binary.csv')
        y_test_prob = pred_res['y_prob']
        y_test_classes = pred_res['y_pred']
        y_test = pred_res['y']

        fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
        auc_test = auc(fpr, tpr)
        acc_test = accuracy_score(y_test_classes, y_test)
        f1_test = f1_score(y_test_classes, y_test, average='binary')
        recall_test = recall_score(y_test_classes, y_test, average='binary')
        precision_test = precision_score(y_test_classes, y_test, average='binary')
        R_test = spearmanr(y_test, y_test_prob)[0]
        precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
        auprc = auc(recall,precision)

        self.acc_test=round(acc_test,3)
        self.auc_test=round(auc_test,3)
        self.f1_test=round(f1_test,3)
        self.precision_test=round(precision_test,3)
        self.recall_test=round(recall_test,3)
        self.R_test=round(R_test,3)
        self.auprc = round(auprc,3)

        if verbose==1:
            print('Test: acc %.3f, auc %.3f, f1 %.3f, precision %.3f, recall %.3f,R %.3f, auprc %.3f\n' % (self.acc_test, self.auc_test, self.f1_test, self.precision_test, self.recall_test, self.R_test, self.auprc))
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % auc_test)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC')
            plt.legend(loc="lower right")
            plt.show()

        return [self.acc_test, self.auc_test, self.f1_test, self.precision_test, self.recall_test, self.R_test, self.auprc]
                
            
        
    def eval_model_cont(self,data,verbose=0):
        self.prediction_cont(data)
        pred_res = pd.read_csv(f'../pred/pred_cont.csv')
        y_preds = pred_res['y_pred']
        y_test = pred_res['y']

        R_test = spearmanr(y_test, y_preds)[0]
        MSE = mean_squared_error(y_test,y_preds)
        self.R_test=round(R_test,3)
        self.MSE=round(MSE,3)
        if verbose==1:
            print('Test: MSE %.3f, R %.3f\n' % (self.MSE,self.R_test))
        return self.MSE, self.R_test


            
    def eval_model_gene(self,data,verbose=1):
        self.prediction_gene(data)
                   
        observed = ['5hmC','RNA']
        
        output = []
        for iobserved in range(len(observed)):
            pred_res = pd.read_csv(f'../pred/pred_gene_{observed[iobserved]}.csv')
            R_test = spearmanr(pred_res.y_pred,pred_res.y)[0]
            MSE = mean_squared_error(pred_res.y_pred,pred_res.y)
            output.extend([MSE,R_test])  
        
        if verbose==1:
            print('Test(5hmC): MSE %.3f, R %.3f\nTest(RNA): MSE %.3f, R %.3f\n' % (output[0],output[1],output[2],output[3]))
        return output[0],output[1],output[2],output[3]
    
            
    def eval_model_diff(self,data,verbose=0):
        self.prediction_diff(data)
        
        pred_res = pd.read_csv(f'../pred/pred_diff.csv')
        y_test_prob = pred_res['y_prob']
        y_test_classes = pred_res['y_pred']
        y_test = pred_res['y']

        fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
        auc_test = auc(fpr, tpr)
        acc_test = accuracy_score(y_test_classes, y_test)
        f1_test = f1_score(y_test_classes, y_test, average='binary')
        recall_test = recall_score(y_test_classes, y_test, average='binary')
        precision_test = precision_score(y_test_classes, y_test, average='binary')
        R_test = spearmanr(y_test, y_test_prob)[0]
        precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
        auprc = auc(recall,precision)

        self.acc_test=round(acc_test,3)
        self.auc_test=round(auc_test,3)
        self.f1_test=round(f1_test,3)
        self.precision_test=round(precision_test,3)
        self.recall_test=round(recall_test,3)
        self.R_test=round(R_test,3)
        self.auprc = round(auprc,3)

        if verbose==1:
            print('Test: acc %.3f, auc %.3f, f1 %.3f, precision %.3f, recall %.3f,R %.3f, auprc %.3f\n' % (self.acc_test, self.auc_test, self.f1_test, self.precision_test, self.recall_test, self.R_test, self.auprc))
            plt.figure()
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % auc_test)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC')
            plt.legend(loc="lower right")
            plt.show()

        return [self.acc_test, self.auc_test, self.f1_test, self.precision_test, self.recall_test, self.R_test, self.auprc]

        

            
        
        
            
        
            
