from dataloader import *
from models import *
from trainer import *
from evaluator import *

# load data for predicting binary 5hmC peaks
data_binary = MyDataset_binary(64*8,['H3K4me1','H3K4me3'])

# load Deep5hmC_binary
model_binary = Deep5hmC_binary(input_read_size=data_binary.read_size)

# train Deep5hmC_binary
trainer = Trainer(num_epochs=5, lr=1e-3, earlystop_thresh=20)

trainer.fit_binary(model_binary,data_binary)

# evaluate Deep5hmC_binary
evaluator = Evaluator(model_binary, '../parameters/Deep5hmC_binary.pth')

evaluator.eval_model_binary(data_binary,verbose=1)