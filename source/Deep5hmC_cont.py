from dataloader import *
from models import *
from trainer import *
from evaluator import *
import json
import argparse


# load data for predicting continuous 5hmC read counts
data_cont = MyDataset_cont(64*8,['H3K4me1','H3K4me3'])

# load Deep5hmC_cont
model_cont = Deep5hmC_cont(input_read_size=data_cont.read_size)


# train Deep5hmC_binary
trainer = Trainer(num_epochs=200, lr=1e-3, earlystop_thresh=20)

trainer.fit_cont(model_cont,data_cont)


# evaluate Deep5hmC_binary
evaluator = Evaluator(model_cont, '../parameters/Deep5hmC_cont.pth')

evaluator.eval_model_cont(data_cont,verbose=1)
