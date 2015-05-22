import sys
PROJ_DIR = '/home/ubuntu/AE_experiments'
if PROJ_DIR not in sys.path:
    sys.path.append('/home/ubuntu/AE_experiments')

import matplotlib
matplotlib.use('Agg') #for running on AWS without ssh -X
import os, pdb
import cPickle as pickle
import train_AE


#home directory for all experiments:
#DATA_DIR = '/Users/vmisra/data/AE_experiments' #local
DATA_DIR = '/home/ubuntu/data/AE_experiments' #AWS

#subdirectories and paths for all experiments:
MODELS_DIR = os.path.join(DATA_DIR,"models")
FUEL_DIR = os.path.join(DATA_DIR,"fuel")
paths_YAML_pretrains = ['layer0_skeleton.yaml', 'layer1_skeleton.yaml']
path_YAML_finetune = 'finetune.yaml'

#parameters
dir_models = os.path.join(MODELS_DIR,"A1")

params = { 
    'dir_models': dir_models,
    'dir_fuel'  : FUEL_DIR,
    'paths_YAML_pretrains' : paths_YAML_pretrains,
    'path_YAML_finetune' : path_YAML_finetune,
    'train_stop': 50000,
    'valid_stop': 60000,
    'n_units' : [784, 10],
    'corruptions' : [0],
    'enc_activations' : ['"tanh"'],
    'dec_activations' : ['"tanh"'],
    'pretrain_batch_size' : 100,
    'pretrain_epochs' : 10,
    'monitoring_batches' : 5,
    'finetune_batch_size' : 100,
    'finetune_epochs' : 100
}
path_params = os.path.join(dir_models,"params.pkl")
pickle.dump(params,open(path_params,'w'))

#training and dumping of model files
trainer = train_AE.train_AE(**params)
trainer.pretrain()
trainer.finetune()