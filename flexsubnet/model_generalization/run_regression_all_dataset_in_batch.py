import argparse
import torch
torch.manual_seed(torch.tensor(760671))
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import model_generalization.generate_and_process_data  as gen_data
from common.utils import Namespace, save_into_pickle, configure, read_from_pickle, create_data_folders
from model_generalization.learn_eval_submodular_functions import learning_neural_submodular_function_synthetic, evaluate_trained_model
from models.NeuralSubmodularCollections \
     import neural_subm_two_level, neural_subm_one_level, DeepSet,subnet_flex_recurrent,\
     mixture_submodular,deep_subm_baseline,subnet_flex_dual,subnet_flex_dual_non_monotone,SetTransformer,subnet_flex_dual_alpha
from common import logger, set_log
###########################################################

def load_synthetic_dataset(av):
    machine = av.machine
    config = configure(machine)

    generated_data_filename = config['synthetic_data_path'] + av.generated_data_filename + '_' + av.phi_concave_func

    dataset_ = read_from_pickle(generated_data_filename)

    return dataset_



SetFuncDict={'DeepSet': DeepSet, 'neural_subm_one_level': neural_subm_one_level, 
             'subnet_flex_recurrent': subnet_flex_recurrent,'mixture_submodular':mixture_submodular, 
             'deep_subm_baseline': deep_subm_baseline, 
             'SetTransformer':SetTransformer,
             'subnet_flex_dual':subnet_flex_dual, 'subnet_flex_dual_non_monotone':subnet_flex_dual_non_monotone,
             'subnet_flex_dual_non_monotone':subnet_flex_dual_non_monotone,
             'subnet_flex_dual_alpha':subnet_flex_dual_alpha}




def generate_dataset(av):
    

        
    if av.load_synthetic_dataset is False:
        synthetic_data = gen_data.synthetic_dataset(av)
        synthetic_data.create_dataset()
        synthetic_data.split_into_training_test()
        synthetic_data.save()
        
        
    if av.load_synthetic_dataset is True:
        print("loading dataset")
        synthetic_data = load_synthetic_dataset(av)

        
    X_train, X_validation, X_test = synthetic_data.data_tensor_padded_folds
    y_train, y_validation, y_test = synthetic_data.F_S_folds

    return X_train, X_validation, X_test, y_train, y_validation, y_test
    

def learn_and_eval(av,TrainingData, ValidationData, TestData):
    
    submodular_function = learning_neural_submodular_function_synthetic(av,TrainingData,ValidationData)
    NO_FLAG = submodular_function.train()
    if NO_FLAG == 0:
        return -1
    
    eval_obj = evaluate_trained_model(av)
    eval_obj.predict(TestData)
    if av.Notebook is True:
        eval_obj.plot_predictions()
        
    err = eval_obj.compute_error()
    print(err)
    return err

 



def main_regression_command_line(av = None):
    
    seeds = {'torch_log':314874, 'DPP': 317043, 'torch_sqrt': 310501, 'fac_loc':315823, 'Gcut':315169,
             'MIX_1': 315671,
              'MIX_2': 320546,
            'MIX_3': 316175,'NGcut8Lambda':314212,'NGcut6Lambda':314132,'NGcut9Lambda':314532,'LOG_SQRT':314204,'LOG_DPP':320666}

    thrs_dict = {'torch_log':None, 'DPP': None, 'torch_sqrt': None, 'fac_loc':None, 'Gcut':None,
             'MIX_1': None,
              'MIX_2': None,
            'MIX_3': None,'LOG_SQRT':None,'LOG_DPP':None,
            'NGcut8Lambda':45000}
    ap = argparse.ArgumentParser()
    ap.add_argument("--patience",                         type=int,   default=1000,help="patience parameter")
    ap.add_argument("--num_features",                     type=int,  default=10)
    ap.add_argument("--total_number_of_elements_in_ground_set",      type=int,  default=10000)
    ap.add_argument("--phi_concave_func",                       type=str, default='NON_')
    ap.add_argument("--machine",                                type=str,   default='NONE')
    
    ap.add_argument("--in_d_umnn",                              type=int,   default=3)
    ap.add_argument("--hidden_layers_umnn",                     type=int,   default=1)
    ap.add_argument("--device",                                 type=str,   default='cuda:0')
    ap.add_argument("--train_num_batches",                      type=int,   default=50)
    ap.add_argument("--method",                    type=str,   default='NONE_')
    ap.add_argument("--load_synthetic_dataset",                 action='store_true')
    ap.add_argument("--load_init_model",                        action='store_true')
    ap.add_argument("--only_dataset_generation",                 action='store_true')
    ap.add_argument("--num_valid_init_models",                        type=int, default =5)
    ap.add_argument("--learning_rate",                        type=float, default =1e-2)
    ap.add_argument("--weight_decay",                        type=float, default =1e-3)
    ap.add_argument("--ORIGINAL",                          action ='store_true')
    ap.add_argument("--convex_generation",                         action ='store_true')
    ap.add_argument("--loadFromMachineID",                   type=str,   default='NONE')
    ap.add_argument("--two_integrals",                   action='store_true')
    ap.add_argument("--epochs",                              type=int,   default=401)



    phi_concave_func = av.phi_concave_func 
    
    av =  ap.parse_args()
    create_data_folders(av.machine)
    
    av.phi_concave_func= phi_concave_func
    
    av.Tag = "Syn_seed_"
    av.thrs = thrs_dict[dataset]


    if av.load_synthetic_dataset == False:
        torch.manual_seed(seeds[av.phi_concave_func])
        
    av.hidden_layers_umnn =  [av.hidden_layers_umnn]
    av.Notebook = False
    av.seed = seeds[av.phi_concave_func]
    av.folds = [0.33,0.33,0.33]
    av.nb_steps = 100
    av.if_save = True
    av.if_intermed_save = True
    av.MAX_ITER = 100
    av.submodular_function = SetFuncDict[av.method]
    av.Notebook =  False

    av.set_func_str =  av.method + "_init_condition_check_LR_" + str(av.learning_rate) + "weight_decay_" + \
                                                                                        str(av.weight_decay) +\
                                                        "hidden" + str(av.hidden_layers_umnn[0])

    print(av.device)
    av.generated_data_filename =  "synthetic_dataset_seed"+ str(av.seed)
    X_train, X_validation, X_test, y_train, y_validation, y_test =  generate_dataset(av)    
    
    if av.only_dataset_generation == True:
        return
    
    TrainingData, ValidationData, TestData = {}, {}, {}
    TrainingData['X_train'] =  X_train
    TrainingData['y_train'] =  y_train
    TestData['X_test'] =  X_test
    TestData['y_test'] =  y_test
    ValidationData['X_dev'] = X_validation
    ValidationData['y_dev'] = y_validation
    
    if av.load_init_model == False and av.load_synthetic_dataset==False:
        torch.manual_seed(torch.randint(1,1000000,(1,)))
    
    err_mat = []
    iter = 1
    for ITER_ in range(av.MAX_ITER):    
        torch.cuda.empty_cache()    

        av.task = av.Tag + str(av.seed) + "_" + str(iter) 
            # + "_convex_gen_" + str(int(av.convex_generation))

        av.logpath = 'logDir_'+ av.method +'/logfile' + av.task+ '_'+ av.set_func_str + '_'+ av.phi_concave_func 

        err = learn_and_eval(av,TrainingData,ValidationData, TestData)
        
        if err!=-1:
            err_mat.append(err)
            iter = iter +1
            if iter == av.num_valid_init_models+1:
                break
            
    mean_err = sum(err_mat) / len(err_mat)
    
    save_error_file = configure(av.machine)['score_path'] \
                        + av.task + '_' + av.phi_concave_func + '_'+ av.set_func_str + '_score_file' 
    
    av.logpath = 'logDir_'+ av.method +'/logfile' + av.task + '_'+ av.set_func_str + '_'+ av.phi_concave_func + '_score_file' 
    set_log(av)
    
    for iter in  range(av.num_valid_init_models):
       logger.info("Error at iteration: %d: %0.4f", iter, err_mat[iter] )

    logger.info("MeanError: %0.4f ", mean_err)
    save_into_pickle(err_mat,save_error_file)

        
if __name__ == "__main__":

    for dataset in ['torch_log','DPP', 'fac_loc','Gcut','LOG_DPP','LOG_SQRT','NGcut8Lambda']:

        
        torch.cuda.empty_cache()    
        av = Namespace()
        av.phi_concave_func = dataset
        main_regression_command_line(av)
