#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:30:35 2018

@author: Alex Grigorevskiy
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
import time
from logger import Logger
import os
import mnist_data

class MLP(nn.Module):
    
    def __init__(self, input_dim=None, output_dim=None,layer_sizes='', self_norm_nn= False, 
                 batch_norm = False,
                 dp_prob=0.0, last_layer_activations=None):
        """
        
        """
        
        super(MLP, self).__init__()
        
        assert (self_norm_nn and not batch_norm) or (not self_norm_nn), "Can not have batch_norm and self-normalizing network!"
        
        self.self_norm_nn = self_norm_nn
        self.dp_prob = dp_prob
        self.batch_norm = batch_norm
        
        tmp_list = [ int(ll) for ll in layer_sizes.split(',') ]
        layer_sizes_list = [input_dim,]
        layer_sizes_list.extend( tmp_list); layer_sizes_list.append( output_dim )
        
        self.layer_sizes_list = layer_sizes_list
        self.last_layer_ind = len(layer_sizes_list) - 2
        
        
        self.layers =  nn.ModuleList( [ nn.Linear(layer_sizes_list[ii], layer_sizes_list[ii+1], bias=True )\
                for ii in range( len(layer_sizes_list) - 1 ) ] )
    
        # Inner layer activations:
        self.activation = torch.nn.ReLU() if (self.self_norm_nn == False) else torch.nn.SELU()
        
        # Last layer activation:
        if last_layer_activations == None or last_layer_activations == 'relu':
            self.last_layer_activation = torch.nn.ReLU()
        elif last_layer_activations == 'selu':
            self.last_layer_activation = torch.nn.SELU()
        elif last_layer_activations == 'linear':
            self.last_layer_activation = lambda xx: xx
        elif last_layer_activations == 'sigmoid':
            self.last_layer_activation = torch.nn.Sigmoid()
        
        #import pdb; pdb.set_trace()
        if self.batch_norm: # add also Batch normalization parameters
            self.batch_norm_layers = nn.ModuleList( [ nn.BatchNorm1d(layer_sizes_list[ii+1])\
                for ii in range( len(layer_sizes_list) - 2 ) ] ) # exclude first and last linea layers
    
        self.dp_prob = dp_prob
        if self.dp_prob > 0:
            self.drop = nn.Dropout(dp_prob) if (self.self_norm_nn == False) else nn.AlphaDropout(dp_prob)
            
        self.initialize_weights()
        
        
    def initialize_weights(self,):
        
        for ll in self.layers:
            weight_init.normal(ll.weight, 0, math.sqrt( 1 / ll.in_features ))
            if ll.bias is not None:
                ll.bias.data.zero_()
            
            
    def forward(self, xx):
        """
        
        """
        self.batch_pass = [ xx.data.numpy() ]
        
        for (ii,ll) in enumerate(self.layers):
            xx = ll(xx)
            
            if ii != self.last_layer_ind:
                xx = self.activation(xx)
                #import pdb; pdb.set_trace()
                if self.batch_norm:
                    xx = self.batch_norm_layers[ii]( xx ) # exclude the first layer
            
                if self.dp_prob > 0:
                    xx = self.drop(xx)
                    
                self.batch_pass.append( xx.data.numpy() )
            else: # for the last layer do not use batch norm or dropout
                xx = self.last_layer_activation(xx)
                
        return xx


class EstimateNorm(object):
    
    def __init__(self, layer_size_list, layer_dims_list):
        """
        Inputs:
        ------------------
        layer_size_list: list
            All layers, including input but NOT the final output layer.s
        layer_dims_list: list of lists
            Dimensions to monitor on each layer. External list length must
            be equal to the number of layers
            
        """
        #import pdb; pdb.set_trace()
        assert len(layer_size_list) == len(layer_dims_list), "List length must be equal"
        
        for ii,ll in enumerate(layer_size_list):
            assert np.all(np.array(layer_dims_list[ii]) < layer_size_list[ii]), """Monitoring dimensions must be lower 
                            than the total number of dimensions"""
        
        self.layer_size_list = layer_size_list
        self.layer_dims_list = layer_dims_list

        self._buffer = {}
    def start_evaluation(self, ):
        """
        Empty buffers. Should be run on the beginning of the evaluation epoch.
        """
        #import pdb; pdb.set_trace()
        self._buffer = {}
        
        for li,ll in enumerate(self.layer_dims_list):
            for dim in ll:
                key = str(li) + '_' + str(dim)
                self._buffer[key] = np.array(())
            
            
    def process_minibatch(self, model):
        """
        
        """
        #import pdb; pdb.set_trace()
        for ii, ll in enumerate(model.batch_pass):
            for dim in self.layer_dims_list[ii]:
                key = str(ii) + '_' + str(dim)
                self._buffer[key] = np.concatenate( (self._buffer[key], model.batch_pass[ii][:, dim] ) )
            

def log_weights_and_grads(model, logger, global_step):
    """
    
    """
    
    for ii,ll in enumerate(model.layers):
        ww = ll.weight.data.numpy() # extract weight and biases
        
        logger.scalar_summary("Weights/FrobNorm/weights_{}".format(ii), np.linalg.norm(ww), global_step)
        if ll.bias is not None:
            bb = ll.bias.data.numpy()
            logger.scalar_summary("Weights/FrobNorm/bias_{}".format(ii), np.linalg.norm(bb), global_step)
        

        # Gradients
        w_grad = ll.weight.grad.data.cpu().numpy()
        
        
        logger.scalar_summary("Gradients/FrobNorm/grad_weight_{}".format(ii), np.linalg.norm(w_grad), global_step)
        if ll.bias is not None:
            b_grad = ll.bias.grad.data.cpu().numpy()
            logger.scalar_summary("Gradients/FrobNorm/grad_biases_{}".format(ii), np.linalg.norm(b_grad), global_step)
        
        
def model_eval(data, data_descr_str, model, logger, epoch_no, p_comp_estimate_norm = False, p_layer_dims_list=None):   
    """
    
    """

    model.eval() # model to evaluation mode
    
    # Estimate norm
    if p_comp_estimate_norm:
        estimate_norm = EstimateNorm(model.layer_sizes_list[:-1], p_layer_dims_list)
        estimate_norm.start_evaluation()
    # Loss function
    loss_func = nn.CrossEntropyLoss()
    
    
    
    loss_epoch = 0; denom_loss_epoch = 0; total_points = 0; accuracy = 0
    for (mb_i, mini_batch) in enumerate(data):
        batch_xx = mini_batch[0]
        batch_yy = mini_batch[1]
        
        
        output = model(Variable(batch_xx))
        
        loss = loss_func(output, Variable(batch_yy)) #/ batch_yy.shape[0]
        
        loss_epoch += loss.data[0]; denom_loss_epoch += 1; total_points += batch_yy.shape[0]
        
        #import pdb; pdb.set_trace()
        _, predicted = torch.max(output.data , 1)
        accuracy += np.sum( batch_yy.numpy() == predicted.numpy() )
        
        
        if p_comp_estimate_norm:
            estimate_norm.process_minibatch(model)
            
     # End of iteration
    loss_epoch = loss_epoch / denom_loss_epoch
    accuracy = accuracy / total_points
    
    logger.scalar_summary("Eval_" + data_descr_str, loss_epoch, epoch_no)
    logger.scalar_summary("Accuracy_" + data_descr_str, accuracy, epoch_no)
    
    if p_comp_estimate_norm:
        for key in estimate_norm._buffer.keys():
            logger.histo_summary(tag="key_{}".format(key), values=estimate_norm._buffer[key],
                               step=epoch_no)
            logger.scalar_summary("Mean_{}".format(key), np.mean(estimate_norm._buffer[key]),epoch_no)
            logger.scalar_summary("Std_{}".format(key), np.std(estimate_norm._buffer[key]),epoch_no)
            
    #tag="Epoch_{}_key_{}".format(epoch_no, key)
    return loss_epoch, accuracy

def main(p_model_save_folder, p_lr=0.01, p_layers='', p_selfnorm=False, p_batch_norm = False, p_last_layer_activation='relu', p_drop_prob=0.0 ):
    
    if os.path.exists(p_model_save_folder): 
        os.system("rm -rf {}".format(p_model_save_folder) )
        
    p_epoch_num = 50
    p_optimizer = "momentum"
    p_momentum = 0.9
    p_weight_decay = 0
    p_summary_freq = 100
    p_eval_dims_list = [[500,], [18,], [17,] , [], [15,], [], [], [], [11,] ] # ignore the output error
    p_save_model = False
    p_schedule= True
    # Instantiate logger
    logger = Logger(p_model_save_folder) 
    
    #import pdb; pdb.set_trace()
    
    # Model
    model = MLP(input_dim=28*28, output_dim=10,layer_sizes=p_layers, self_norm_nn= p_selfnorm, 
                batch_norm = p_batch_norm, dp_prob=p_drop_prob, last_layer_activations=p_last_layer_activation)
    
    # Loss function
    loss_func = nn.CrossEntropyLoss()
    
    # Data
    train_iter, test_iter = mnist_data.get_mnist_data( data_dir = None, normalized = p_selfnorm or p_batch_norm, train_batch_size=128, test_batch_size=128, train_shuffle=True)
    
    
    # Optimizer
    if p_optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=p_lr,
                               weight_decay=p_weight_decay)
    elif p_optimizer == "momentum":
        optimizer = optim.SGD(model.parameters(),
                              lr=p_lr, momentum=p_momentum,
                              weight_decay=p_weight_decay)
        scheduler = MultiStepLR(optimizer, milestones=[20, ], gamma=0.5)
        print("Momentum: ", p_momentum)
    
    
    global_step = 0
    loss_name = 'ce_loss'
    for epoch in range(p_epoch_num):
        
        loss_epoch = 0 
        denom_loss_epoch = 0
        loss_run = 0 # running loss for running summary
        denom_loss_run = 0
        epoch_time = time.time()
        
        
        for (mb_i, mini_batch) in enumerate(train_iter):
            
            
            model.train() # model to the traininig mode
            
            if p_schedule:
                scheduler.step()
                
            batch_xx = mini_batch[0]
            batch_yy = mini_batch[1]
        
            optimizer.zero_grad()
            output = model(Variable(batch_xx))
         
            #import pdb; pdb.set_trace()
            loss = loss_func(output, Variable(batch_yy)) #/ batch_yy.shape[0]
        
            loss.backward()
            optimizer.step()
            
            loss_run += loss.data[0]
            loss_epoch += loss.data[0]
            
            denom_loss_epoch += 1; denom_loss_run += 1
            global_step += 1
            
            
            if (mb_i % p_summary_freq) == 0:
                #import pdb; pdb.set_trace()
                loss_run = loss_run / denom_loss_run
                
                print('[%d, %5d] loss( %s ): %.7f' % (epoch, mb_i, loss_name, loss_run ) )#running loss
                logger.scalar_summary('ce_loss_run', loss_run, global_step)
                log_weights_and_grads(model, logger, global_step)
    
                loss_run = 0; denom_loss_run = 0;
                
        # End of epoch
        #import pdb; pdb.set_trace()
        epoch_time = time.time() - epoch_time
        loss_epoch = loss_epoch / denom_loss_epoch # mean epoch error
        
        print('Total epoch {} finished in {} seconds with TRAINING loss {}: {}'
              .format(epoch, epoch_time, loss_name, loss_epoch))
        
        logger.scalar_summary("Training_loss_per_epoch", loss_epoch, epoch)
        loss_epoch = 0; denom_loss_epoch = 0
        
        
        # Compute evaluation data ->
        if epoch % 3 == 0 or epoch == (p_epoch_num - 1):
            #import pdb; pdb.set_trace()
            l_tmp1, acc_tmp_1 = model_eval(train_iter, 'train_data', model, logger, epoch, p_comp_estimate_norm = True, p_layer_dims_list=p_eval_dims_list)
            l_tmp2, acc_tmp_2 = model_eval(test_iter, 'test_data', model, logger, epoch, p_comp_estimate_norm = False, p_layer_dims_list=None)
            
            
            print('Epoch {} train loss ({} ): {},   accuracy:  {}'.format(epoch, loss_name, l_tmp1, acc_tmp_1 ))
            print('Epoch {} test loss ({} ): {},   accuracy:  {}'.format(epoch, loss_name, l_tmp2, acc_tmp_2 ))
            
        # Compute evaluation costs <-
        
        # Save model
        if epoch % 3 == 0 or epoch == p_epoch_num - 1:
            if p_save_model:
                print("Saving model to {}".format(p_model_save_folder + "/model.epoch_"+str(epoch)))
                torch.save(model.state_dict(), p_model_save_folder + "/model.epoch_"+str(epoch))
    # Save last model
    if p_save_model:
        print("Saving model to {}".format(p_model_save_folder + "/model.last"))
        torch.save(model.state_dict(), p_model_save_folder + "/model.last")
        
        
if __name__ == '__main__':
    # Self-norm NN.
    # p_lr=0.02, p_layers = '100,100'.
    main('model_sn',p_lr=0.02, p_layers='100,100,50,50,50,20,20,20', p_selfnorm=True, p_batch_norm = False, p_last_layer_activation='linear', p_drop_prob=0.0 )
    #main('model_bn',p_lr=0.02, p_layers='100,100,50,50,50,20,20,20', p_selfnorm=False, p_batch_norm = True, p_last_layer_activation='linear', p_drop_prob=0.0 )
    #main('model_un',p_lr=0.02, p_layers='100,100,50,50,50,20,20,20', p_selfnorm=False, p_batch_norm = False, p_last_layer_activation='linear', p_drop_prob=0.0 )
    
# Tensorboard can be run:
#    tensorboard --logdir=1_self-norm:./model_sn,2_batch-norm:./model_bn,3_no_norm:./model_un --port 2000
    
    