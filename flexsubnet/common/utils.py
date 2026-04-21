import torch
import matplotlib.pyplot as plt
import configparser
import pickle
import os
import random
CONFIGFILE = 'common/config.ini'


def name_of(x):
    return type(x).__name__


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def create_data_folders(machine):
    config = configure(machine)
    for path in config.keys():
        os.makedirs(config[path], exist_ok=True )


def compute_precision_acc(labels, predictions):
    sorted_predictions = sorted(((e, i) for i, e in enumerate(predictions)), reverse=True)
    precision = 0.0
    r = int(sum(labels))
    j = 1
    for i in range(len(sorted_predictions)):
        if labels[sorted_predictions[i][1]] == 1.0:
            precision += 1.0 * j / (i+1) / r
            j += 1

    return precision


def union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list

def generate_skewed_distr(n,p):
    alpha = torch.tensor([(1-p)/(n-1)]*n)
    m = random.randint(0,n-1)
    alpha[m] = p
    a  = torch.distributions.dirichlet.Dirichlet(alpha)
    return a.sample()

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def delete_row(A,i):
    return torch.cat((A[:i],A[i+1:]),0)

def generate_test_permutation(k):
    II =  torch.eye(k)
    return 1.0*II[torch.randperm(k),:]

def find_not_in_set(U,S):
    Ind = torch.ones(U.shape[0],dtype=bool)
    Ind[S] = False
    return U[Ind]


def listOfTuple2List(X):
    return  [X[i][0] for i in range(len(X))]
    

def all_pairs_dist(A, B):

    sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], B.shape[0])
    sqrB = torch.sum(torch.pow(B, 2), 1, keepdim=True).expand(B.shape[0], A.shape[0]).t()

    d = torch.sqrt(torch.abs(sqrA - 2*torch.mm(A, B.t()) + sqrB)) + 1000*torch.eye(sqrA.shape[0])
    return  1/(d+0.01)

def MakeTranspose3DTensor(X):
    X =torch.transpose(X, 0, 2)
    return torch.transpose(X, 1, 2)

def generate_h(x,d):
    return 0*torch.randn((x.shape[0],d))

def scale_tensor(x,fac):
    return fac*(x-torch.min(x))/(torch.max(x)-torch.min(x))

def plot_fn(y, y_pred):

    plt.plot(y, 'r', label="Submodular function")
    plt.plot(y_pred, 'g', label="Approximation")
    plt.legend(loc="lower left")
    plt.show()

def normz_grad(model):
    return torch.norm(torch.tensor([torch.norm(p.grad) for p in model.parameters()]))

def reciprocal_func(x):
    return -1.0 / (x**(0.33)+0.01)+1

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def update(self, **kwargs):
        self.__dict__.update(kwargs)

def MakeCopy(FromA,ToB):
    for a in FromA.__dict__.keys():
#         print(a)
        ToB.__dict__[a] = FromA.__dict__[a]
#     print(ToB.__dict__)

def configure(machine):
    x = configparser.ConfigParser()
    x.read(CONFIGFILE)
    #machine = x['MACHINE']['machine']
    config = x[machine]
    return config   

def save_into_pickle(data, filename):
    filename = filename + '.pickle'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def read_from_pickle(filename):
    filename = filename + '.pickle'
    with open(filename, 'rb') as f:
        x = pickle.load(f)
        return x
    
def send_to_device(device,*args):
    for arg in args:
        arg.to(device)


