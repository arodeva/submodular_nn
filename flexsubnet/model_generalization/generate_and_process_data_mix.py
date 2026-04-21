import torch
import torch.nn as nn
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import configparser 
from common.utils import read_from_pickle, scale_tensor, all_pairs_dist, configure, Namespace, save_into_pickle, MakeTranspose3DTensor
from matplotlib import pyplot as plt
import submodlib.functions as subf
from sklearn.metrics import pairwise_distances
import numpy as np

# subm_dict={'torch_log':torch.log, 'torch_sqrt': torch.sqrt, 'fac_loc':'fac_loc',
#            'DPP':'DPP','DisSum':'DisSum','Gcut':'Gcut'}
subm_dict={'torch_log':torch.log, 'torch_sqrt': torch.sqrt, 'fac_loc':'fac_loc',
           'DPP':'DPP' ,'Gcut':'Gcut'}

wt_vec = torch.rand(5)
wt_vec = (wt_vec+1)/torch.sum(wt_vec+1)


def load_synthetic_dataset(av):
    machine = av.machine 
    config = configure(machine)

    generated_data_filename = config['synthetic_data_path'] + av.generated_data_filename + '_' + av.phi_concave_func
    
    dataset_ = read_from_pickle(generated_data_filename) 
    
    return dataset_
class synthetic_dataset(object):

    def __init__(self,av):
    
        self.av = av
        self.number_of_elements_ground_set =  av.total_number_of_elements_in_ground_set ## n = |V|
        self.dim_element = av.num_features
#         self.phi_concave_func =  subm_dict[av.phi_concave_func]
        self.folds = av.folds
        self.data_mat = torch.abs(torch.randn(self.number_of_elements_ground_set, self.dim_element))
        
        for self.phi_concave_func  in ['Gcut']:
            DataMat = self.data_mat.numpy()
            S = 1-pairwise_distances(DataMat, metric="cosine")
            self.DisSum = subf.disparitySum.DisparitySumFunction(self.number_of_elements_ground_set,
                                                                   "dense", sijs=S)                       
            self.Gcut =  subf.graphCut.GraphCutFunction(self.number_of_elements_ground_set,
                                                        'dense', 0.01, separate_rep=False, ggsijs=S)

    def create_dataset(self):
                        
        number_of_elements_ground_set =  self.number_of_elements_ground_set
        dim_element =  self.dim_element

        data_mat_tensor = self.data_mat.unsqueeze(2).expand(number_of_elements_ground_set,
                                                       dim_element,
                                                       number_of_elements_ground_set)

        lower_triangular_mat = torch.tril(torch.ones(number_of_elements_ground_set,number_of_elements_ground_set)).t()
        lower_triangular_tensor= lower_triangular_mat.unsqueeze(1).expand(number_of_elements_ground_set,
                                                                           dim_element,
                                                                           number_of_elements_ground_set)

        self.data_tensor_padded = torch.abs(torch.mul(lower_triangular_tensor,data_mat_tensor))
        
        F_S = {}
        jj=0
        for self.phi_concave_func in list(subm_dict.keys()):
            jj = jj + 1
            print(jj)

            if self.phi_concave_func in ['Gcut']:
                print(self.phi_concave_func)
                F_S_list = [self.Gcut.evaluate(set(range(i)))  for i in range(number_of_elements_ground_set)]

                self.F_S_value = torch.tensor(F_S_list)
    #             self.F_S_value = scale_tensor(self.F_S_value, 10)
                F_S[self.phi_concave_func] = scale_tensor(self.F_S_value, 10)

            if self.phi_concave_func in ['torch_log', 'torch_sqrt']:
                print(self.phi_concave_func)
                self.F_S_value = subm_dict[self.phi_concave_func](torch.sum(self.data_tensor_padded,dim=[0,1]))

                F_S[self.phi_concave_func] = scale_tensor(self.F_S_value, 10)


            if self.phi_concave_func == 'fac_loc':
                print(self.phi_concave_func)
                X = all_pairs_dist(self.data_mat, self.data_mat).to(self.av.device)    

                self.F_S_value = torch.empty(number_of_elements_ground_set)

                for i in range(0,X.shape[0]):

                    self.F_S_value[i] = torch.sum(torch.max(X[0:i+1 :],0,keepdim=True)[0])

    #             self.F_S_value = scale_tensor(self.F_S_value, 10)

                F_S[self.phi_concave_func] = scale_tensor(self.F_S_value, 10)

            if self.phi_concave_func == 'DPP':
                print(self.phi_concave_func)
                data_mat = self.data_mat/torch.norm(self.data_mat)
                X = torch.mm(data_mat, data_mat.t()).to(self.av.device)    

                self.F_S_value = torch.empty(number_of_elements_ground_set)

                for i in range(0,X.shape[0]):


                    
                    DPP_mat = X[0:i+1, 0:i+1]+ (1+1e-6)*torch.eye(i+1).to(self.av.device)
                    self.F_S_value[i] = torch.log(torch.linalg.det(DPP_mat))

                F_S[self.phi_concave_func] = scale_tensor(self.F_S_value, 10)
    
        x = 0
        i= -1
        for self.phi_concave_func in list(subm_dict.keys()):
            i =  i+1
            x = x  + wt_vec[i] * F_S[self.phi_concave_func] 
    
        self.F_S_value =  x
        self.w =  wt_vec
        
    def split_into_training_test(self):


        data_mat, data_tensor_padded, F_S_value, folds =  self.data_mat, self.data_tensor_padded, self.F_S_value, self.folds

        num_elem =  data_mat.shape[0]
        tr_size = int(folds[0]* num_elem)
        dev_size = int((folds[1]+folds[0])* num_elem)
        test_size =   num_elem
        folds = self.folds



        permuted_index = torch.randperm(num_elem)
        self.fold_indices = {'train':permuted_index[0:tr_size], 
                              'dev': permuted_index[tr_size:dev_size], 
                              'test': permuted_index[dev_size:test_size],
                              'all':permuted_index}


        # data_mat_training = data_mat[permuted_index[0:tr_size],:]
        data_tensor_padded_training = data_tensor_padded[:,:,permuted_index[0:tr_size]]
        F_S_training = F_S_value[permuted_index[0:tr_size]]

        # data_mat_dev = data_mat[permuted_index[tr_size:dev_size],:]
        data_tensor_padded_dev = data_tensor_padded[:,:,permuted_index[tr_size:dev_size]]
        F_S_dev = F_S_value[permuted_index[tr_size:dev_size]]

        # data_mat_test = data_mat[permuted_index[dev_size:test_size],:]
        data_tensor_padded_test = data_tensor_padded[:,:,permuted_index[dev_size:test_size]]
        F_S_test = F_S_value[permuted_index[dev_size:test_size]]

        # self.data_mat_folds = (data_mat_training,data_mat_dev,data_mat_test)
        self.data_tensor_padded_folds = (data_tensor_padded_training,data_tensor_padded_dev,data_tensor_padded_test)
        self.F_S_folds = (F_S_training, F_S_dev, F_S_test)

        ## new
        self.data_tensor_padded_folds = (MakeTranspose3DTensor(data_tensor_padded_training),
                                         MakeTranspose3DTensor(data_tensor_padded_dev),
                                         MakeTranspose3DTensor(data_tensor_padded_test))

    def save(self):

        syn_dataset = Namespace()
        syn_dataset.data_tensor_padded_folds = self.data_tensor_padded_folds
        syn_dataset.F_S_folds = self.F_S_folds
        syn_dataset.w =  self.w
        machine = self.av.machine 
        config = configure(machine)
        
        self.generated_data_filename = self.av.generated_data_filename + '_' + self.av.phi_concave_func
        save_file = config['synthetic_data_path'] + self.generated_data_filename

        save_into_pickle(syn_dataset, save_file)


 
