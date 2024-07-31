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

model_types = config['model_types']
tissue = config['tissue']
histones = config['histones']

batch_size = config['batch_size']
num_epochs = config['num_epochs']
lr = config['lr']
earlystop_thresh = config['earlystop_thresh']

# load data for predicting continuous 5hmC peaks
data = MyDataset(data_path,tissue,histones, batch_size)

# load continuous
model = Deep5hmC_cont(input_read_size=data.read_size)

# train Deep5hmC_cont
trainer = Trainer(weights_path = weights_path, tissue = tissue, num_epochs=num_epochs, lr=lr, earlystop_thresh=earlystop_thresh)

trainer.fit(model = model, data = data, model_type = model_types[1])

# evaluate Deep5hmC_cont
evaluator = Evaluator(prediction_path = prediction_path, tissue = tissue, model=model, best_model = trainer.best_model)
# evaluator = Evaluator(prediction_path = prediction_path, tissue = tissue, model=model, best_model = f'../pretrained/Deep5hmC_cont.pth') # evaluate pretrained model

evaluator.eval_model(data, model_type = model_types[1], verbose=1)
