#Script manages training various types of stacked denoising autoencoders

from pylearn2.config import yaml_parse
from pylearn2.models.mlp import PretrainedLayer, MLP
from pylearn2.space import VectorSpace
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

import theano
import theano.tensor as T

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

class XtropyReconstructionCost(DefaultDataSpecsMixin, Cost):
    supervised = False

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        X = data
        X_hat = model.reconstruct(X)
        loss = -T.mean(X*T.log(X_hat) + (1-X)*T.log(1-X_hat))
        return loss

class XtropyReconstructionCost_batchsum(DefaultDataSpecsMixin, Cost):
    supervised = False

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        X = data
        X_hat = model.reconstruct(X)
        loss = -T.sum(X*T.log(X_hat) + (1-X)*T.log(1-X_hat),axis=1)
        return T.mean(loss)

class XtropyReconstructionCost_batchsum_tanh(DefaultDataSpecsMixin, Cost):
    supervised = False

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        X = data
        X_hat = model.reconstruct(X*.5)
        X_hat_norm = (X_hat + 1)*.5
        lossa = -T.sum((1-X)*T.log(1-X_hat_norm), axis=1)
        lossb =  -T.sum( X*T.log(X_hat_norm), axis=1)
        return T.mean(lossa+lossb)

class MLP_autoencoder_Dropout(MLP):
    def __init__(self, input_include_probs=None, input_scales=None, **kwargs):
        super(MLP_autoencoder_Dropout, self).__init__(**kwargs)
        self.input_include_probs = input_include_probs
        self.input_scales = input_scales

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
        return self.dropout_fprop(state_below=inputs, 
            input_include_probs = self.input_include_probs,
            input_scales = self.input_scales,
            default_input_include_prob = 1,
            default_input_scale=1)

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
        return self.fprop(state_below=inputs)

class train_AE(object):
    def __init__(self,
                 dir_models,  
                 dir_fuel,  
                 paths_YAML_pretrains,
                 path_YAML_finetune,
                 path_YAML_solotrain,
                 train_stop = 50000,
                 valid_stop = 60000,
                 n_units = [784, 1000, 10],
                 corruptions = [0.3, 0.3],
                 enc_activations = ['"tanh"','"tanh"'],
                 dec_activations = ['"tanh"','"tanh"'],
                 pretrain_batch_size = 100,
                 pretrain_epochs = 10,
                 pretrain_lr=0.1,
                 monitoring_batches = 5,
                 finetune_batch_size = 100,
                 finetune_epochs = 100,
                 finetune_lr=0.1,
                 solotrain_batch_size = 100,
                 solotrain_epochs = 100,
                 solotrain_lr=0.1,
                 pretrain_cost_YAML=['!obj:train_AE.MeanSquaredReconstructionError'],
                 finetune_cost_YAML='!obj:train_AE.MeanSquaredReconstructionError',
                 solotrain_cost_YAML='!obj:train_AE.MeanSquaredReconstructionError',
                 irange=[.05,.05],
                 input_probs = [0,0],
                 input_scales =[1,1],
                 no_pretrain_activations = None
                 ):
        n_layers = len(n_units)-1
        dim_layers = zip(n_units[:-1],n_units[1:])
        paths_pretrained = []

        for layer_idx in range(n_layers):
            paths_pretrained.append('"'+os.path.join(dir_models,str(layer_idx)+'.pkl')+'"')

        self.__dict__.update(locals())
        del self.layer_idx
        del self.self

        
        

#generates YAML for finetuning the overall AE, WITH NO PRETRAINING.
    def genSolotrain_YAML(self):
        #build up layers_spec string
        base_layers_YAML = {
        "relu" : \
        """!obj:pylearn2.models.mlp.RectifiedLinear {
        layer_name: '%(name)s',
        dim: %(dim)i,
        irange: %(irange)f
        } """,
        "tanh" : \
        """!obj:pylearn2.models.mlp.Tanh {
        layer_name: '%(name)s',
        dim: %(dim)i,
        irange: %(irange)f
        } """,
        "sigmoid" : \
        """!obj:pylearn2.models.mlp.Sigmoid {
        layer_name: '%(name)s',
        dim: %(dim)i,
        irange: %(irange)f
        } """
        }

        layers_spec = ""
        dims = self.n_units[1:] + self.n_units[-2::-1]

        for idx in range(len(dims)):
            layer_dict = { 
                "name": str(idx),
                "dim": dims[idx],
                "irange": self.irange[idx]
                }
            base_layer_YAML = base_layers_YAML[self.no_pretrain_activations[idx]]
            if idx == 0:
                layers_spec = base_layer_YAML % layer_dict
            else:
                layers_spec = layers_spec + ", \n" + base_layer_YAML % layer_dict
        

        ##Build up input_probs string
        input_probs_string = ''
        for idx in range(len(dims)):
            input_probs_string += "'"+str(idx)+"'"
            input_probs_string += ":"+str(self.input_probs[idx])
            if idx < len(dims)-1:
                input_probs_string += ','

        ##Build up input_scales string
        input_scales_string = ''
        for idx in range(len(dims)):
            input_scales_string += "'"+str(idx)+"'"
            input_scales_string += ":"+str(self.input_scales[idx])
            if idx < len(dims)-1:
                input_scales_string += ','

        YAML_raw = open(self.path_YAML_solotrain,'r').read()
        YAML_dict = { 
            "train_stop" : self.train_stop,
            "batch_size" : self.solotrain_batch_size,
            "layers_spec" : layers_spec,
            "valid_stop" : self.valid_stop,
            "max_epochs" : self.solotrain_epochs,
            "save_path" : os.path.join(self.dir_models,"solotrain.pkl"),
            "mnist_train_X_path": os.path.join(self.dir_fuel,"mnist_train_X.pkl"),
            "mnist_valid_X_path": os.path.join(self.dir_fuel,"mnist_valid_X.pkl"),
            "mnist_test_X_path": os.path.join(self.dir_fuel,"mnist_test_X.pkl"),
            "cost": self.solotrain_cost_YAML,
            "lr": self.solotrain_lr,
            "input_probs" : input_probs_string,
            "input_scales" : input_scales_string
        }

        self.YAML_solotrain = YAML_raw % YAML_dict

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
                "save_path" : self.paths_pretrained[idx],
                "cost" : self.pretrain_cost_YAML[idx],
                "lr" : self.pretrain_lr
            }

            #add the load_path parameters as well
            for layer_idx in range(self.n_layers):
                YAML_dict["load_path_"+str(layer_idx)] = self.paths_pretrained[layer_idx]
                YAML_dict["irange_"+str(layer_idx)] = self.irange[layer_idx]

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
            "max_epochs" : self.finetune_epochs,
            "save_path" : os.path.join(self.dir_models,"finetune.pkl"),
            "mnist_train_X_path": os.path.join(self.dir_fuel,"mnist_train_X.pkl"),
            "mnist_valid_X_path": os.path.join(self.dir_fuel,"mnist_valid_X.pkl"),
            "mnist_test_X_path": os.path.join(self.dir_fuel,"mnist_test_X.pkl"),
            "cost": self.finetune_cost_YAML,
            "lr": self.finetune_lr
        }

        self.YAML_finetune = YAML_raw % YAML_dict

    def pretrain(self):
        self.genPretrainYAML_list()
        for yaml in self.YAML_pretrain:
            print(yaml)
            train = yaml_parse.load(yaml)
            train.main_loop()

    def finetune(self):
        self.genFinetuneYAML()
        print(self.YAML_finetune)
        train = yaml_parse.load(self.YAML_finetune)
        train.main_loop()

    def solotrain(self):
        self.genSolotrain_YAML()
        print(self.YAML_solotrain)
        train = yaml_parse.load(self.YAML_solotrain)
        train.main_loop()
