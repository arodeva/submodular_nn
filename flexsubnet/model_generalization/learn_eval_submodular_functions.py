import torch
import copy
import torch.optim as optim
from models.NeuralSubmodularCollections import neural_subm_two_level,neural_subm_one_level
import matplotlib.pyplot as plt
from common.utils import *
from common import logger, set_log
import os
import math

def evaluate_standalone(X,y_true,model,Notebook=False): 
    h = generate_h(X,model.in_d-1)

    y_pred  = model(X,h)[0] if (name_of(model).startswith('subnet_flex_dual')) else model(X,h)
    y_pred = y_pred.squeeze(1)

    _, indices = torch.sort(y_true)
    assert y_pred.shape == y_true.shape
    
    True_loss = ((y_pred[indices]-y_true[indices])**2).sum().item()/indices.shape[0]

    if Notebook is True:
        X_size = torch.sum(X, dim=[1,2])
        values,indices = torch.sort(X_size)
        plt.plot(values.to('cpu').detach().numpy(),y_pred[indices].to('cpu').detach().numpy(),color='r')
        plt.plot(values.to('cpu').detach().numpy(),y_true[indices].to('cpu').detach().numpy(),color='b')
        plt.show()
        print("True Loss ", True_loss)

        
    return  True_loss

class learning_neural_submodular_function_synthetic(object):
    
    def __init__(self,av,TrainingData, ValidationData):
        
        MakeCopy(av,self)
        self.X_train = TrainingData['X_train']
        self.y_train = TrainingData['y_train']
        
        self.X_dev = ValidationData['X_dev']
        self.y_dev = ValidationData['y_dev']
        
        self.b_size = int(self.X_train.shape[0]/av.train_num_batches)
        self.weight_decay = av.weight_decay
        self.av = av
        self.nsf = av.submodular_function
        set_log(av)
        
        
    def train(self):
        


        if self.load_init_model is True:
            model_subm = self.read_init_model()
            if self.method=="subnet_flex_dual_non_monotone":
                model_subm.thrs = self.thrs

            model_subm =  model_subm.to(self.device)
            model_subm.device = self.device
            self.init_model = model_subm.to(self.device)
        else:
            
            if self.method=="subnet_flex_dual_non_monotone":
                model_subm = self.nsf(self.num_features,
                             self.in_d_umnn, 
                             self.hidden_layers_umnn, 
                             nb_steps=self.nb_steps,
                             dev=self.device,thrs=self.thrs,two_integrals=self.two_integrals).to(self.device)
            else:
                model_subm = self.nsf(self.num_features,
                            self.in_d_umnn, 
                            self.hidden_layers_umnn, 
                            nb_steps=self.nb_steps,
                            dev=self.device).to(self.device)
            
            self.init_model = copy.deepcopy(model_subm)

            if self.convex_generation == True:
                assert self.av.ORIGINAL == False
                self.init_model = self.generate_init_condition()
                model_subm = copy.deepcopy(self.init_model)
                
        print(self.count_params())

        optim_subm = torch.optim.Adam(model_subm.parameters(),
                                   self.learning_rate,
                                   weight_decay=self.weight_decay)
    
        X_train_, y_train_ = self.X_train.to(self.device), self.y_train.to(self.device)
        
        
        last_error = float('Inf')
        
        for epoch in range(0, self.epochs):
            
          
            
            idx = torch.randperm(X_train_.shape[0])
            
            X_train = X_train_[idx]
            y_train = y_train_[idx]
            
            total_loss = 0.

                
            for i in range(0, X_train.shape[0], self.b_size):
            
                x = X_train[i:i + self.b_size].requires_grad_()
                y = y_train[i:i + self.b_size].requires_grad_()
                h = generate_h(x, self.in_d_umnn-1)
                
                if  name_of(model_subm).startswith('subnet_flex_dual'):
                    y_pred, x_input_to_first_integrand, OuterIntegral_Second_Out  = model_subm(x,h)

                    y_pred = y_pred.squeeze(1).to(self.device)
                    loss = torch.sum((y_pred - y)**2) + torch.sum((x_input_to_first_integrand - OuterIntegral_Second_Out)**2)
                    loss_1 = torch.sum((y_pred - y)**2)
                else:
                    y_pred  = model_subm(x,h)                
                    y_pred = y_pred.squeeze(1).to(self.device)
                    loss = torch.sum((y_pred - y)**2) 
                    loss_1 = torch.sum((y_pred - y)**2)

                optim_subm.zero_grad()
                loss.backward()
                optim_subm.step()
        
                total_loss += loss_1.item()
            
            logger.info("epoch %d",epoch)    
            logger.info("Average Training loss: %0.4f", total_loss/X_train.shape[0] )
            self.trained_model = model_subm 
            current_error = evaluate_standalone(self.X_train.to(self.device),
                                                self.y_train.to(self.device),
                                                model_subm,self.Notebook)
            logger.info("True loss: %0.4f", current_error )

            
            if epoch == 0 and self.load_init_model == False:
                if  self.check_no_flag(current_error, y_pred) == True:
                    
                    self.save(init=True, FLAG=False)
                    
                else:

                    self.save(init = True, FLAG = True)
                    return 0
            
            if epoch%50 ==0 or epoch>=self.epochs-10 and self.av.if_intermed_save == True:
                self.current_epoch = epoch
                self.save(intermediate=True)
                
            if current_error<last_error and epoch>=5 and self.av.if_save == True:
                last_error = current_error
                self.save()
        
        return 1
    
                
    def check_no_flag(self, total_loss,y_pred):
        
        out = True
        print("checking flag")
        if total_loss > 1e10 or torch.std(y_pred) < 1e-4 or math.isnan(total_loss):
            out = False
            print("flagged")
        return out
    
    def save(self,init=False,intermediate=False, FLAG = False):
        
        training_output =  Namespace()
        training_output.av = self.av
        training_output.trained_model = self.trained_model
        config = configure(self.machine)
        
        if init == False and intermediate == False:
            path = config['save_path_for_trained_model']
            os.makedirs(path,exist_ok=True)
            filename = path + self.task + '_' + self.phi_concave_func + '_'+ self.set_func_str  + '_output'
            
        if init == True and FLAG == False:
            
            training_output.trained_model = None
            training_output.init_model = self.init_model
            
            path = config['save_path_for_initial_model']
            os.makedirs(path,exist_ok=True)
            filename = path + self.task + '_' + self.phi_concave_func + '_'+ self.set_func_str +'_output_init_Fixed'
       
        if init == True and FLAG == True:
            
            training_output.trained_model = None
            training_output.init_model = self.init_model
            
            path = config['save_path_for_initial_model']
            os.makedirs(path,exist_ok=True)
            filename = path + self.task + '_' + self.phi_concave_func + '_'+ self.set_func_str +'_output_init_FLAGGED'

    
    
        if intermediate == True:
            path = config['save_path_for_intermediate_model']
            os.makedirs(path,exist_ok=True)
            filename = path + self.task + '_' + self.phi_concave_func+ '_'+ \
                                self.set_func_str  + '_output_intermediate_epoch_'+ str(self.current_epoch)

            
            
            
        if self.av.Notebook == True:
            filename = filename+'_Notebook'
            
        save_into_pickle(training_output, filename)
        
        
    def read_init_model(self):
        config = configure(self.machine)
        path = config['save_path_for_initial_model']
        filename = path + self.task + '_' + self.phi_concave_func + '_'+ self.set_func_str +'_output_init_Fixed'
        training_output = read_from_pickle(filename)
        return training_output.init_model
    
    def count_params(self):
        
        c = 0
        for p in self.init_model.parameters():
            c = c + torch.numel(p)
        
        return c 
    
class evaluate_trained_model(object):
    def __init__(self,av):
        MakeCopy(av,self)
        config = configure(self.machine)
        path = config['save_path_for_trained_model']
           
        filename = path + self.task + '_' + self.phi_concave_func + '_'+ self.set_func_str  + '_output'
        
        if av.Notebook == True:
            filename = filename+'_Notebook'
        
        training_output = read_from_pickle(filename)
        self.model = training_output.trained_model
        self.task = av.task
        
    def predict(self, TestData):
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.X_test = TestData['X_test'].to(self.device)
        y_true = TestData['y_test'].to(self.device)
        
        h = generate_h(self.X_test,self.model.in_d-1)
        # y_pred  = model(X,h)[0] if name_of(model)=='subnet_flex_dual' else model(X,h)

        y_pred  = (self.model(self.X_test,h)[0]).squeeze(1).to(self.device) \
                    if name_of(self.model).startswith('subnet_flex_dual') \
                    else (self.model(self.X_test,h)).squeeze(1).to(self.device)

        y_true_sorted, indices = torch.sort(y_true)
        y_pred_sorted = y_pred[indices]
        
        self.y_true = y_true_sorted
        self.y_pred = y_pred_sorted
        
        
    def plot_predictions(self,format_fig=False):
        
        plt.plot(self.y_pred.to('cpu').detach().numpy(),color='r')
        plt.plot(self.y_true.to('cpu').detach().numpy(),color='b')
        plt.show()
    
    def compute_error(self):
    
        assert self.y_pred.shape == self.y_true.shape
        error = torch.mean((self.y_true-self.y_pred)**2) 
        return error
