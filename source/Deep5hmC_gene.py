from dataloader import *
from models import *
from trainer import *
from evaluator import *
import json
import argparse

parser = argparse.ArgumentParser(description="Run model training and evaluation with config file")
parser.add_argument('json_file', type=str, help='Path to the JSON configuration file')
args = parser.parse_args()

with open(args.json_file, 'r') as f:
    config = json.load(f)

data_path = config['data_path']
weights_path = config['weights_path']
prediction_path = config['prediction_path']

model_type = config['model_type']
tissue = config['tissue']
histones = config['histones']

batch_size = config['batch_size']
num_epochs = config['num_epochs']
lr = config['lr']
earlystop_thresh = config['earlystop_thresh']



data = MyDataset_gene(data_path,tissue,histones,batch_size)

model = Deep5hmC_gene(input_read_size=data.read_size) #Deep5hmC_gene = Deep5hmC_cont

evaluator = Evaluator(prediction_path = prediction_path, tissue = tissue, model=model, best_model = f'../weights/EB/Deep5hmC_cont.pth')
#evaluator = Evaluator(prediction_path = prediction_path, tissue = tissue, model=model, best_model = f'../pretrained/Deep5hmC_cont.pth')

evaluator.eval_model(data, model_type = model_type, verbose=1)