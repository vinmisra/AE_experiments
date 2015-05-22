#Script manages training various types of stacked denoising autoencoders

from pylearn2.config import yaml_parse
from pylearn2.models.mlp import PretrainedLayer, MLP
from pylearn2.space import VectorSpace
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

import os
import pdb

Pretrained_DAE_Layer_Encode = PretrainedLayer
class Pretrained_DAE_Layer_Decode(PretrainedLayer):
    def fprop(self, state_below):
        return self.layer_content.decode(state_below)
    def get_input_space(self):
        return VectorSpace(self.layer_content.nhid)
    def get_output_space(self):
        return VectorSpace(self.layer_content.nvis)

class MeanSquaredReconstructionCost(DefaultDataSpecsMixin, Cost):
    supervised = False

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        X = data
        X_hat = model.reconstruct(X)
        loss = ((X-X_hat)**2).mean(axis=1)
        return loss.mean()


class MLP_autoencoder(MLP):
    def reconstruct(self, inputs):
        """
        Reconstruct the inputs after corrupting and mapping through the
        encoder and decoder.
        Parameters
        ----------
        inputs : tensor_like or list of tensor_likes
            Theano symbolic (or list thereof) representing the input
            minibatch(es) to be corrupted and reconstructed. Assumed to be
            2-tensors, with the first dimension indexing training examples
            and the second indexing data dimensions.
        Returns
        -------
        reconstructed : tensor_like or list of tensor_like
            Theano symbolic (or list thereof) representing the corresponding
            reconstructed minibatch(es) after encoding/decoding.
            NO CORRUPTION at present
        """
        return self.fprop(inputs)

class train_AE():
    def __init__(self,
                 dir_models,  
                 dir_fuel,  
                 paths_YAML_pretrains,
                 path_YAML_finetune,
                 train_stop = 50000,
                 valid_stop = 60000,
                 n_units = [784, 1000, 10],
                 corruptions = [0.3, 0.3],
                 enc_activations = ['"tanh"','"tanh"'],
                 dec_activations = ['"tanh"','"tanh"'],
                 pretrain_batch_size = 100,
                 pretrain_epochs = 10,
                 monitoring_batches = 5,
                 finetune_batch_size = 100,
                 finetune_epochs = 100
                 ):
        n_layers = len(n_units)-1
        dim_layers = zip(n_units[:-1],n_units[1:])
        paths_pretrained = []

        for layer_idx in range(n_layers):
            paths_pretrained.append('"'+os.path.join(dir_models,str(layer_idx)+'.pkl')+'"')

        self.__dict__.update(locals())
        del self.layer_idx
        del self.self

        self.genPretrainYAML_list()
        self.genFinetuneYAML()

#generates a list of YAML for the denoising AE pretraining.
    def genPretrainYAML_list(self):
        self.YAML_pretrain = []

        for idx in range(self.n_layers):
            #add all the simple one-off parameters
            YAML_dict = {
                "train_stop" : self.train_stop,
                "nvis" : self.dim_layers[idx][0],
                "nhid" : self.dim_layers[idx][1],
                "corruption_level" : self.corruptions[idx],
                "act_enc" : self.enc_activations[idx],
                "act_dec" : self.dec_activations[idx],
                "batch_size" : self.pretrain_batch_size,
                "monitoring_batches" : self.monitoring_batches,
                "max_epochs" : self.pretrain_epochs,
                "save_path" : self.paths_pretrained[idx]
            }

            #add the load_path parameters as well
            for layer_idx in range(self.n_layers):
                YAML_dict["load_path_"+str(layer_idx)] = self.paths_pretrained[layer_idx]

            #different YAML skeletons for layer 0 and the later layers.
            YAML_raw = open(self.paths_YAML_pretrains[idx],'r').read()

            #substitute in, and add to list of ready-to-go
            self.YAML_pretrain.append(YAML_raw % YAML_dict)

#generates YAML for finetuning the overall AE
    def genFinetuneYAML(self):
        #build up layers_spec string
        base_layers_spec = \
        """!obj:train_AE.Pretrained_DAE_Layer_%(encode_or_decode)s {
        layer_name: '%(name)s',
        layer_content: !pkl: %(load_path)s
        } """

        layers_spec = ""
        for idx in range(self.n_layers):
            layer_dict = { 
                "name": "encode_"+str(idx),
                "load_path": self.paths_pretrained[idx],
                "encode_or_decode":"Encode"}
            if idx == 0:
                layers_spec = base_layers_spec % layer_dict
            else:
                layers_spec = layers_spec + ", \n" + base_layers_spec % layer_dict
                
        for idx in range(self.n_layers)[::-1]:
            layer_dict = {
                "name": "decode_"+str(idx),
                "load_path": self.paths_pretrained[idx],
                "encode_or_decode":"Decode"
            }
            layers_spec = layers_spec + " , \n" + base_layers_spec % layer_dict

        YAML_raw = open(self.path_YAML_finetune,'r').read()
        YAML_dict = { 
            "train_stop" : self.train_stop,
            "batch_size" : self.finetune_batch_size,
            "layers_spec" : layers_spec,
            "valid_stop" : self.valid_stop,
            "max_epochs" : self.pretrain_epochs,
            "save_path" : os.path.join(self.dir_models,"finetune.pkl"),
            "mnist_train_X_path": os.path.join(self.dir_fuel,"mnist_train_X.pkl"),
            "mnist_valid_X_path": os.path.join(self.dir_fuel,"mnist_valid_X.pkl"),
            "mnist_test_X_path": os.path.join(self.dir_fuel,"mnist_test_X.pkl")
        }

        self.YAML_finetune = YAML_raw % YAML_dict

    def pretrain(self):
        for yaml in self.YAML_pretrain:
            train = yaml_parse.load(yaml)
            train.main_loop()

    def finetune(self):
        print(self.YAML_finetune)
        train = yaml_parse.load(self.YAML_finetune)
        train.main_loop()