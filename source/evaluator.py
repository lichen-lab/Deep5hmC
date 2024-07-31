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
            
    def eval_model(self, data, model_type, verbose=0):
        self.prediction(data, model_type)
        
        if model_type == 'binary':
            pred_res = pd.read_csv(os.path.join(self.pred_dir, 'pred_binary.csv'))
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

        else:
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