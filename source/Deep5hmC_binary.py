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


# load data for predicting binary 5hmC peaks
data_binary = MyDataset(f'{data_path}/{tissue}', batch_size, histones)

    # load Deep5hmC_binary
    model_binary = Deep5hmC_binary(input_read_size=data_binary.read_size)

    # train Deep5hmC_binary
    trainer = Trainer(num_epochs=num_epochs, lr=lr, earlystop_thresh=earlystop_thresh)

    trainer.fit_binary(model_binary,data_binary)

    # evaluate Deep5hmC_binary
    evaluator = Evaluator(model_binary, best_model = '../parameters/Deep5hmC_binary.pth')

    evaluator.eval_model_binary(data_binary,verbose=1)
