import sys, os, pdb

#home code directory:
#PROJ_DIR = '/Users/vmisra/Dropbox/Research/compression/AE_experiments' #local
PROJ_DIR = '/home/ubuntu/AE_experiments' #AWS
#home directory for all experiments:
#DATA_DIR = '/Users/vmisra/data/AE_experiments' #local
DATA_DIR = '/home/ubuntu/data/AE_experiments' #AWS
#subdirectories and paths for all experiments:
MODELS_DIR = os.path.join(DATA_DIR,"models")
FUEL_DIR = os.path.join(DATA_DIR,"fuel")

#set stdout to print and to log to a file.

if PROJ_DIR not in sys.path:
    sys.path.append(PROJ_DIR)

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath,'w')
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        pass

#######Now the actual experiment code
import matplotlib
matplotlib.use('Agg') #for running on AWS without ssh -X
import cPickle as pickle
import train_AE


names = ['B11']
n_unitss = [[784,1000,250,64,5]]
corruptionss = [[.2,.2,.2,.2]]
enc_activationss = [['"sigmoid"']*len(corruptions) for corruptions in corruptionss]
dec_activationss = [['"sigmoid"']*len(corruptions) for corruptions in corruptionss]

for (name,n_units, corruptions, enc_activations, dec_activations) in zip(names,n_unitss,corruptionss,enc_activationss, dec_activationss):
    
    dir_models = os.path.join(MODELS_DIR,name)
    if not os.path.exists(dir_models):
        os.makedirs(dir_models)

    sys.stdout = Logger(os.path.join(dir_models,'stdout.log'))

    params = { 
    'dir_models': dir_models,
    'dir_fuel'  : FUEL_DIR,
    'paths_YAML_pretrains' : ['layer'+str(x)+'_skeleton.yaml' for x in range(len(corruptions))],
    'path_YAML_finetune' : 'finetune_simpletrain.yaml',
    'train_stop': 50000,
    'valid_stop': 60000,
    'n_units' : n_units,
    'corruptions' : corruptions,
    'enc_activations' : enc_activations,
    'dec_activations' : dec_activations,
    'pretrain_lr' : 0.1,
    'pretrain_batch_size' : 100,
    'pretrain_epochs' : 15,
    'monitoring_batches' : 5,
    'finetune_lr' : 0.1,
    'finetune_batch_size' : 100,
    'finetune_epochs' : 300,
    'pretrain_cost_YAML' : '!obj:train_AE.XtropyReconstructionCost_batchsum',
    'finetune_cost_YAML' : '!obj:train_AE.XtropyReconstructionCost_batchsum'
    }
    path_params = os.path.join(dir_models,"params.pkl")
    pickle.dump(params,open(path_params,'w'))

    #training and dumping of model files
    trainer = train_AE.train_AE(**params)
    trainer.pretrain()
    trainer.finetune()

# ### EXP A2a-A2d
# #parameters
# names = ['A2a','A2b','A2c','A2d']
# n_unitss = [[784,1500,10],
#            [784,1000,10],
#            [784,500,10],
#            [784,250,10]]

# ### EXP A2e
# #parameters
# dir_models = os.path.join(MODELS_DIR,"A2e")
# if not os.path.exists(dir_models):
#     os.makedirs(dir_models)

# params = { 
#     'dir_models': dir_models,
#     'dir_fuel'  : FUEL_DIR,
#     'paths_YAML_pretrains' : paths_YAML_pretrains,
#     'path_YAML_finetune' : path_YAML_finetune,
#     'train_stop': 50000,
#     'valid_stop': 60000,
#     'n_units' : [784, 125, 10],
#     'corruptions' : [0,0],
#     'enc_activations' : ['"tanh"','"tanh"'],
#     'dec_activations' : ['"tanh"','"tanh"'],
#     'pretrain_batch_size' : 100,
#     'pretrain_epochs' : 10,
#     'monitoring_batches' : 5,
#     'finetune_batch_size' : 100,
#     'finetune_epochs' : 100
# }
# path_params = os.path.join(dir_models,"params.pkl")
# pickle.dump(params,open(path_params,'w'))

# ### EXP A2f
# #parameters
# dir_models = os.path.join(MODELS_DIR,"A2f")
# if not os.path.exists(dir_models):
#     os.makedirs(dir_models)

# params = { 
#     'dir_models': dir_models,
#     'dir_fuel'  : FUEL_DIR,
#     'paths_YAML_pretrains' : paths_YAML_pretrains,
#     'path_YAML_finetune' : path_YAML_finetune,
#     'train_stop': 50000,
#     'valid_stop': 60000,
#     'n_units' : [784, 64, 10],
#     'corruptions' : [0,0],
#     'enc_activations' : ['"tanh"','"tanh"'],
#     'dec_activations' : ['"tanh"','"tanh"'],
#     'pretrain_batch_size' : 100,
#     'pretrain_epochs' : 10,
#     'monitoring_batches' : 5,
#     'finetune_batch_size' : 100,
#     'finetune_epochs' : 100
# }
# path_params = os.path.join(dir_models,"params.pkl")
# pickle.dump(params,open(path_params,'w'))


# ###EXP A
# #parameters
# dir_models = os.path.join(MODELS_DIR,"A1")
# if not os.path.exists(dir_models):
#     os.makedirs(dir_models)

# params = { 
#     'dir_models': dir_models,
#     'dir_fuel'  : FUEL_DIR,
#     'paths_YAML_pretrains' : paths_YAML_pretrains,
#     'path_YAML_finetune' : path_YAML_finetune,
#     'train_stop': 50000,
#     'valid_stop': 60000,
#     'n_units' : [784, 10],
#     'corruptions' : [0],
#     'enc_activations' : ['"tanh"'],
#     'dec_activations' : ['"tanh"'],
#     'pretrain_batch_size' : 100,
#     'pretrain_epochs' : 10,
#     'monitoring_batches' : 5,
#     'finetune_batch_size' : 100,
#     'finetune_epochs' : 100
# }
# path_params = os.path.join(dir_models,"params.pkl")
# pickle.dump(params,open(path_params,'w'))





