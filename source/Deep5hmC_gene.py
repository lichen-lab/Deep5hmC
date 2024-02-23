from dataloader import *
from models import *
from trainer import *
from evaluator import *

# load data for predicting DhMRs
data_diff = MyDataset_diff(64*2,['H3K4me3','H3K27ac'])

# load Deep5hmC_diff
model_diff = Deep5hmC_diff(input_read_size=data_diff.read_size)

# train Deep5hmC_diff
trainer = Trainer(num_epochs=200, lr=1e-3, earlystop_thresh=20)

trainer.fit_diff(model_diff,data_diff)

# Evaluate Deep5hmC_diff
evaluator = Evaluator(model_diff, '../parameters/Deep5hmC_diff.pth')

evaluator.eval_model_diff(data_diff,verbose=1)