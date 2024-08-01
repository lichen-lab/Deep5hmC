import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score, accuracy_score, precision_recall_curve, average_precision_score, mean_squared_error
from scipy.stats import wilcoxon, spearmanr
import matplotlib.pyplot as plt
import pandas as pd
import os

class Evaluator():
    def __init__(self, prediction_path, tissue, model, best_model):
        print("[INFO] resume best model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(best_model))
        self.model = model.to(self.device)

        self.tissue = tissue
        self.prediction_path = prediction_path
        self.pred_dir = os.path.join(self.prediction_path, self.tissue)
        if not os.path.exists(self.pred_dir):
            os.makedirs(self.pred_dir)
        
    def prepare_data(self, data):
        self.test_dataloader = data.test_dataloader()

    def prediction(self, data, model_type):
        self.prepare_data(data)
        
        print("[INFO] evaluating network...")
        with torch.no_grad():
            self.model.eval()

            preds = []
            predsProb = []
            ys = []

            for (readx1, readx2, seqx, y_binary, y_continuous) in self.test_dataloader:
                (readx1, readx2, seqx, y_binary, y_continuous) = (
                    readx1.to(self.device, dtype=torch.float),
                    readx2.to(self.device, dtype=torch.float),
                    seqx.to(self.device, dtype=torch.float),
                    y_binary.to(self.device),
                    y_continuous.to(self.device, dtype=torch.float)
                )
                y = y_binary if model_type == 'binary' else y_continuous
                pred = self.model(readx1, readx2, seqx)
                if model_type == 'binary':
                    preds.extend(pred.argmax(axis=1).cpu().numpy())
                    predsProb.extend(pred[:,1].cpu().numpy())
                else:
                    preds.extend(pred.cpu().numpy())
                ys.extend(y.cpu().numpy())

            self.preds = np.array(preds)
            self.ys = np.array(ys)
            if model_type == 'binary':
                self.predsProb = np.array(predsProb)
                pred_res = pd.DataFrame({'y_prob': self.predsProb, 'y_pred': self.preds, 'y': self.ys})
                pred_res.to_csv(os.path.join(self.pred_dir, 'pred_binary.csv'), index=False)
            else:
                self.preds = np.squeeze(self.preds)
                pred_res = pd.DataFrame({'y_pred': self.preds, 'y': self.ys})
                pred_res.to_csv(os.path.join(self.pred_dir, 'pred_cont.csv'), index=False)
            
            
        
        
    def prediction_diff(self,data):
        self.prepare_data(data)
        
        print("[INFO] evaluating network...")
        with torch.no_grad():
              
            self.model.eval()
            
            preds = []
            predsProb = []
            ys = []

            for (readx1,readx2, y) in data.test_dataloader():
                (readx1,readx2, y) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),y.to(self.device))
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
            
            pred_res.to_csv(os.path.join(self.pred_dir, 'pred_diff.csv'), index=False)        
            
            
    def prediction_gene(self,data):
        self.prepare_data(data)
        
        print("[INFO] evaluating network...")
        with torch.no_grad():
            
            self.model.eval()
            
            preds = []
            ys_5hmC = []
            ys_GE = []
            genes_id = []

            for (readx1,readx2,seqx,y_5hmC, y_GE, gene_id) in self.test_dataloader:
                (readx1,readx2,seqx) = (readx1.to(self.device,dtype=torch.float), readx2.to(self.device,dtype=torch.float),seqx.to(self.device,dtype=torch.float))
                pred = self.model(readx1,readx2,seqx)
                
                preds.extend(pred.cpu().numpy())
                ys_5hmC.extend(y_5hmC.numpy())
                ys_GE.extend(y_GE.numpy())
                genes_id.extend(gene_id.numpy())
            
            self.preds = np.squeeze(np.array(preds))
            self.ys_5hmC = np.squeeze(np.array(ys_5hmC))
            self.ys_GE = np.squeeze(np.array(ys_GE))
            self.genes_id= np.squeeze(np.array(genes_id))
            
                       
            output = pd.DataFrame({'y_pred' : np.exp(self.preds),
                        'y_5hmC' : self.ys_5hmC,
                        'y_GE' : self.ys_GE,
                        'gene_id': self.genes_id})

            pred_res = output.groupby('gene_id').agg({
                'y_pred': 'sum',
                'y_5hmC': 'first',
                'y_GE': 'first'
            }).reset_index()

            pred_res['y_pred'] = np.log(pred_res['y_pred']+1)

            pred_res = pred_res.replace([np.inf, -np.inf], np.nan).dropna(how='any', axis=0, subset=None)

            pred_res.to_csv(os.path.join(self.pred_dir, 'pred_gene.csv'), index=False)        
            
    
    
    def eval_model(self, data, model_type, verbose=0):
        if model_type in ['binary','cont']:
            self.prediction(data, model_type)
        elif model_type=='diff':
            self.prediction_diff(data)
        elif model_type=='gene':
            self.prediction_gene(data)
        
        if model_type in ['binary','diff']:
            pred_res = pd.read_csv(os.path.join(self.pred_dir, f'pred_{model_type}.csv'))
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
            auprc = auc(recall, precision)

            self.acc_test = round(acc_test, 3)
            self.auc_test = round(auc_test, 3)
            self.f1_test = round(f1_test, 3)
            self.precision_test = round(precision_test, 3)
            self.recall_test = round(recall_test, 3)
            self.R_test = round(R_test, 3)
            self.auprc = round(auprc, 3)

            if verbose == 1:
                print('Test: acc %.3f, auroc %.3f, auprc %.3f, f1 %.3f, precision %.3f, recall %.3f, R %.3f\n' % (self.acc_test, self.auc_test, self.auprc, self.f1_test, self.precision_test, self.recall_test, self.R_test))
                plt.figure()
                lw = 2
                plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc_test)
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC')
                plt.legend(loc="lower right")
                plt.show()

            return [self.acc_test, self.auc_test, self.auprc, self.f1_test, self.precision_test, self.recall_test, self.R_test]

        elif model_type=='cont':
            pred_res = pd.read_csv(os.path.join(self.pred_dir, 'pred_cont.csv'))
            y_preds = pred_res['y_pred']
            y_test = pred_res['y']

            R_test = spearmanr(y_test, y_preds)[0]
            MSE = mean_squared_error(y_test, y_preds)
            self.R_test = round(R_test, 3)
            self.MSE = round(MSE, 3)
            if verbose == 1:
                print('Test: MSE %.3f, R %.3f\n' % (self.MSE, self.R_test))
            return self.MSE, self.R_test
        
        elif model_type=='gene':
            pred_res = pd.read_csv(os.path.join(self.pred_dir, 'pred_gene.csv'))
            y_preds = pred_res['y_pred']
            y_5hmC = pred_res['y_5hmC']
            y_GE = pred_res['y_GE']
            
            R1 = spearmanr(y_5hmC, y_preds)[0]
            R2 = spearmanr(y_GE, y_preds)[0]
            MSE1 = mean_squared_error(y_5hmC, y_preds)
            MSE2 = mean_squared_error(y_GE, y_preds)
            self.R_5hmC = round(R1, 3)
            self.R_GE = round(R2,3)
            self.MSE_5hmC = round(MSE1, 3)
            self.MSE_GE = round(MSE2, 3)
            if verbose == 1:
                print('Test: MSE_5hmC %.3f, R_5hmC %.3f, MSE_GE %.3f, R_GE %.3f' % (self.MSE_5hmC, self.R_5hmC, self.MSE_GE, self.R_GE))
            return self.MSE_5hmC, self.R_5hmC, self.MSE_GE, self.R_GE