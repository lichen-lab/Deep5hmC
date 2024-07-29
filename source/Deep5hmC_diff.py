from dataloader import *
from models import *
from trainer import *
from evaluator import *

# load data for predicting gene expression
data_gene = MyDataset_gene(64*8,['H3K4me1','H3K4me3'])

# load Deep5hmC_gene
model_gene = Deep5hmC_gene(input_read_size=data_gene.read_size)

# evaluate Deep5hmC_gene (Note: Using pretrained Deep5hmC_cont)
evaluator = Evaluator(model_gene, '../weights/Deep5hmC_cont.pth')

evaluator.eval_model_gene(data_gene,verbose=1)
