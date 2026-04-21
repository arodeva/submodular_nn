import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.UMNN.NeuralIntegral import NeuralIntegral
from models.UMNN.ParallelNeuralIntegral import ParallelNeuralIntegral
from common.utils import find_not_in_set, generate_h, delete_row
zerotensor = torch.tensor(0) 

def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

class graph_subm_func(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers,nb_steps=50, dev="cpu"):
        pass

class SetTransformer(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers,nb_steps=50, dev="cpu"):
        super(SetTransformer, self).__init__()
        dim_input = 1
        num_inds=2
        self.in_d = 1
        dim_hidden=hidden_layers[0]
        num_heads=1
        num_outputs =1 
        dim_output =1
        ln=False
        
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)).to(dev)
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output),
                ).to(dev)

    def forward(self, X,h):
        X  = torch.min(X,dim=2)[0]
        X = X.unsqueeze(2)
        return self.dec(self.enc(X)).squeeze(-1)
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

######################################################################
class DeepSet(nn.Module):
    def __init__(self, num_feature, in_d, hidden_layers,nb_steps=50, dev="cpu"):
        super(DeepSet, self).__init__()
        
        self.init_layer = nn.Linear(num_feature,in_d)
        self.num_feature = num_feature
        self.net = []
        self.in_d = in_d
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  
#         self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net).to(dev)
        self.device = dev
         
         
    def forward(self, x, h):
        x = x.to(self.device)
        if len(list(x.shape)) == 2:
            x = x.unsqueeze(1)
        x  = self.init_layer(x)
        x = torch.sum(x,dim=1)
        x  = self.net(x)
        return x
 
class IntegrandNN_zero_stable(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN_zero_stable, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

        self.final_layer = nn.Linear(1,1, bias = True)
        self.reset_params()

             
        self.relu = nn.ReLU() 
        
    def reset_params(self):
        self.final_layer.weight.data =  torch.abs(self.final_layer.weight.data)
        
    def forward(self, x, h):
        x  = self.net(torch.cat((x, h), 1)) 
        # out =  torch.cat((x,x,x, x, x),1)
        return self.relu(self.final_layer(x))

class IntegrandNN(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

        self.final_layer = nn.Linear(5,1, bias = False)
        self.reset_params()

             
        self.relu = nn.ReLU() 
        
    def reset_params(self):
        self.final_layer.weight.data =  torch.abs(self.final_layer.weight.data)
        
    def forward(self, x, h):
        x  = self.net(torch.cat((x, h), 1)) +1

        out =  torch.cat((x**1.5,x**2,x**2.5, x**3, x**3.5),1)
        return self.relu(self.final_layer(1/(out+1)))

class IntegrandNN_outer_level(nn.Module):
    def __init__(self, in_d, hidden_layers):
        super(IntegrandNN_outer_level, self).__init__()
        self.net = []
        hs = [in_d] + hidden_layers + [1]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  
        self.net.append(nn.ELU())
        self.net = nn.Sequential(*self.net)

        self.final_layer = nn.Linear(5,1, bias = False)
        self.reset_params()

             
        self.relu = nn.ReLU() 
        
    def reset_params(self):
        self.final_layer.weight.data =  torch.abs(self.final_layer.weight.data)
        
    def forward(self, x, h):
        x  = self.net(torch.cat((x, h), 1)) +1

        out =  torch.cat((x**1.1,x**1.1,x**1.5, x**2, x**2.5),1)
        return self.relu(self.final_layer(1/(out+1)))



class neural_subm_two_level(nn.Module):
    def __init__(self, num_feature, in_d,
                 hidden_layers, nb_steps=50, dev="cpu"):
        super(neural_subm_two_level, self).__init__()
        
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand = IntegrandNN(in_d, hidden_layers)
        self.InnerIntegrand = IntegrandNN(in_d, hidden_layers)
        
    def forward(self, x, h, xrandom = zerotensor, hrandom = zerotensor):
        x = torch.sum(x,dim=1).unsqueeze(1)
        x0 = torch.zeros(x.shape).to(self.device)
        x = x.to(self.device)
        h = h.to(self.device)
        
        amax = torch.zeros(xrandom.shape).to(self.device)+100        
        hrandom =  hrandom.to(self.device)
        xrandom =  xrandom.to(self.device)
            
        OuterIntegralOut = ParallelNeuralIntegral.apply(x0, x, self.OuterIntegrand,
                                                        _flatten(self.OuterIntegrand.parameters()), h, self.nb_steps)

        InnerIntegralOut = 0
        
        return OuterIntegralOut, InnerIntegralOut
 


class neural_subm_from_argmax(nn.Module):
    def __init__(self, neural_net, num_feature, in_d,
                 hidden_layers, nb_steps=50, dev="cpu",mode = 'softmax'):
        super(neural_subm_from_argmax, self).__init__()
        self.device = dev
        self.in_d = in_d
        if neural_net == deep_subm_baseline or neural_net == mixture_submodular:
            self.neural_net =  neural_net(num_feature, in_d, hidden_layers, nb_steps=50, dev = self.device, task="summary")
        else:
            self.neural_net =  neural_net(num_feature, in_d, hidden_layers, nb_steps=50, dev = self.device)

        self.mode = mode
        self.LogSoftmax = nn.LogSoftmax(dim=1)
        
    def preprocess_input(self, UniversalSet, ArgMaxSet_Index, Not_ArgMax_Index):
        
        dim_element = self.neural_net.num_feature
        size_ArgMax = ArgMaxSet_Index.shape[0]
        size_Universe = UniversalSet.shape[0]

        Universe_Index = torch.tensor(list(range(size_Universe)))
#         Not_ArgMax_Index = find_not_in_set(Universe_Index, ArgMaxSet_Index)

        self.Upper_tri = torch.triu(torch.ones(size_ArgMax, size_Universe)).to(self.device)
        self.minus_large_upper_tri = -1000.0*(1-self.Upper_tri).to(self.device)
        
        self.minus_large_upper_tri_tensor = \
                    self.minus_large_upper_tri.unsqueeze(2).expand(size_ArgMax, size_Universe, dim_element)
        self.Upper_tri_tensor = self.Upper_tri.unsqueeze(2).expand(size_ArgMax, size_Universe, dim_element)
        
        self.ArgMaxSet_feat =  UniversalSet[ArgMaxSet_Index,:]
        self.Not_ArgMaxSet_feat = UniversalSet[Not_ArgMax_Index,:]

        self.ArgMaxSet_Index = ArgMaxSet_Index
        self.Universe_Index =  Universe_Index
        
        self.size_ArgMax = size_ArgMax
        self.size_Universe = size_Universe
        self.dim_element = dim_element
        
    def process_input(self,P):
        
        Permuted_ArgMaxSet_feat = torch.matmul(P, self.ArgMaxSet_feat)
        feat_mat = torch.cat((Permuted_ArgMaxSet_feat, self.Not_ArgMaxSet_feat),0)
#         print(Permuted_ArgMaxSet_feat.shape,  self.Not_ArgMaxSet_feat.shape, feat_mat.shape)
        self.feat_tensor = feat_mat.unsqueeze(0).expand(self.size_ArgMax, self.size_Universe, self.dim_element)

        ArgMaxSet_tensor = Permuted_ArgMaxSet_feat.unsqueeze(1)
        ArgMaxS_shifted_by_one_element = torch.cat([0*ArgMaxSet_tensor[0:1,:,:],ArgMaxSet_tensor])[:-1,:,:] 
        
        ArgMaxS_expanded =  ArgMaxS_shifted_by_one_element.expand(self.size_ArgMax, self.size_Universe, self.dim_element)
        ArgMaxS_cumsum =  torch.cumsum(ArgMaxS_expanded, dim =0)
        Arg_of_Softmax  = ArgMaxS_cumsum  + self.feat_tensor

        self.Arg_of_Softmax_padded =  Arg_of_Softmax
        
        
    def forward(self,P):
        
        self.process_input(P)
        n,m = self.Arg_of_Softmax_padded.shape[0], self.Arg_of_Softmax_padded.shape[1]
        x = self.Arg_of_Softmax_padded.view(-1,self.dim_element) 
        h = generate_h(self.Arg_of_Softmax_padded.view(-1,self.dim_element), self.in_d-1)
        
        out_ = self.neural_net(x,h)
#         print(out_.shape,x.shape)
        out_ = out_.view(n,m)
        if self.mode == 'softmax':
            out =  self.Upper_tri * out_ +   self.minus_large_upper_tri
            self.out_padded = out
            self.out_mat = self.LogSoftmax(out)
            self.log_like = torch.sum(torch.diagonal(self.out_mat))
            
            return self.log_like, out_
        
    def maximize(self,U, budget):
        
        UU = U.to(self.device)
        seq = []
        for i in range(budget):
            h = generate_h(UU, self.in_d-1)
            out = self.neural_net(UU,h)
            out[seq]=torch.min(out)-100
            ii =  torch.argmax(out)
            UU = UU + UU[ii]
            seq.append(ii.item())
            
        return seq
    
class mixture_submodular(nn.Module):
    def __init__(self, num_feature, in_d, 
                 hidden_layers, nb_steps=50, dev="cpu",num_rnn_steps=2,task=None):
        super(mixture_submodular, self).__init__()
        
        self.task = task
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d = in_d
        self.device = dev
        self.relu = nn.ReLU()
        self.mixnet = nn.Linear(in_d,1)
 
        
        
    def reset_parameters(self):
        
        self.log_weight.data = 0*self.log_weight.data+1

        
            
    def compute_modular_function(self, x_, h_ , n):
        

        x = x_
        
        if len(list(x.shape))==3:
            x = torch.sum(x,dim = [1,2]).unsqueeze(1)
        else:
            x = torch.sum(x,dim = 1).unsqueeze(1)
            
            
        x = x.to(self.device)
        h = h_.to(self.device)
        
        return x, h

    def forward(self, x_, h_):
        
        
        x, h = self.compute_modular_function( x_, h_ , 0)

        comp = []
        for i in range(self.in_d):
            comp.append(x)
            if self.task != 'summary':
                x = torch.log(x)

            if self.task == 'summary':
                x = torch.log(torch.abs(x)+1)
            
        OuterIntegralOut = self.mixnet(torch.cat(tuple(comp),1)) 

        return  OuterIntegralOut
    
    
            
class deep_subm_baseline(nn.Module):
    def __init__(self, num_feature, in_d, 
                 hidden_layers, nb_steps=50, dev="cpu",num_rnn_steps=2,task=None):
        super(deep_subm_baseline, self).__init__()
        self.task=task
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature

        self.device = dev
        self.in_d = in_d
        self.splus =  nn.Softplus()

        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))

        self.thrs =  nn.Parameter(torch.Tensor(1))   
        self.relu =  nn.ReLU()
        self.reset_parameters()
        

        
    def reset_parameters(self):
        
        self.log_weight.data = 0*self.log_weight.data+1
        self.thrs.data = 0*self.thrs.data + 1
        
            
    def compute_modular_function(self, x_, h_ , n):
        
        wt =   self.log_weight#self.splus(self.log_weight)

        x = nn.functional.linear(x_, wt,bias = None)

        
        if len(list(x.shape))==3:
            x = torch.sum(x,dim = [1,2]).unsqueeze(1)
        else:
            x = torch.sum(x,dim = 1).unsqueeze(1)
            
            
        x = x.to(self.device)
        h = h_.to(self.device)
        
        return x, h

    def forward(self, x_, h_):
        
        for n in range(self.num_rnn_steps):

            x, h = self.compute_modular_function( x_, h_ , n)
            
            if self.task!="summary":
                x = torch.log(x+self.thrs**2)
                ##this often can create numerical error if x somehow < -self.thrs**2; in that case
                ## use torch.log(x**2+self.thrs**2): we found that this often gives better performance;
                ## another option is torch.sigmoid(x+1)+ self.thrs**2
           
            if self.task=="summary":
                x = torch.log(torch.abs(x)+1) ## another option is torch.sigmoid(torch.abs(x)+1)
        return  x
                
        
class subnet_flex_recurrent(nn.Module):
    def __init__(self, num_feature, in_d, 
                 hidden_layers, nb_steps=50, dev="cpu",num_rnn_steps=2):
        super(subnet_flex_recurrent, self).__init__()
        
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d =in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand = InnerUMNN(num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device)

        
#         self.splus =  nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
#         self.log_weight_t = nn.Parameter(torch.Tensor(1, self.num_rnn_steps))    
        self.sigmoid_param =  nn.Parameter(torch.Tensor(1))   
        self.reset_parameters()
        

        
    def reset_parameters(self):
        
        self.log_weight.data = 0*self.log_weight.data+1
#         self.log_weight_t.data =  0*self.log_weight_t.data+1

            
    def compute_modular_function(self, x_, h_ , n):
        
#         log_weight_t =   self.log_weight_t
#         one_hot_t = F.one_hot(torch.tensor(n), num_classes =  self.num_rnn_steps)
#         one_hot_t = one_hot_t.float() 
#         t_vec = nn.functional.linear(one_hot_t.to(self.device), log_weight_t.to(self.device),bias=None)

        
        x = nn.functional.linear(x_, self.log_weight,bias = None)

#         x = x  + 0* t_vec
        
        if len(list(x.shape))==3:
            x = torch.sum(x,dim = [1,2]).unsqueeze(1)
        else:
            x = torch.sum(x,dim = 1).unsqueeze(1)
            
            
        x = x.to(self.device)
        h = h_.to(self.device)
        
        return x, h

    def forward(self, x_, h_):
        
        
        OuterIntegralOut = 0
        for n in range(self.num_rnn_steps):

            x, h = self.compute_modular_function( x_, h_ , n)

            x0 = torch.zeros(x.shape).to(self.device) 

            lam =  self.sigmoid(self.sigmoid_param)
            x = (1-lam) * x +  lam * OuterIntegralOut
            
#             x = x
            OuterIntegralOut = ParallelNeuralIntegral.apply(x0, x, self.OuterIntegrand,
                                                            _flatten(self.OuterIntegrand.parameters()), h, self.nb_steps) 
            
            

        return  OuterIntegralOut
        
        
class modular_func(nn.Module):
    def __init__(self, num_feature, in_d,
                 hidden_layers, nb_steps=50, dev="cpu"):
        super(modular_func, self).__init__()
        self.num_feature = num_feature
        self.in_d =in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand = InnerUMNN(num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device)
        self.offset = torch.nn.Linear(1, 1, bias=True).to(self.device)
        self.splus =  nn.Softplus()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.log_weight.data = 0*self.log_weight.data+1

    def forward(self, x, h):
        
            
        # wt =   self.splus(self.log_weight)
        x = nn.functional.linear(x, self.log_weight,bias=None)

        if len(list(x.shape))==3:
            x = torch.sum(x,dim = [1,2]).unsqueeze(1)
        else:
            x = torch.sum(x,dim = 1).unsqueeze(1)
            
            
        x0 = torch.zeros(x.shape).to(self.device) 
        x = x.to(self.device)
        return  x

    
class neural_subm_one_level(nn.Module):
    def __init__(self, num_feature, in_d,
                 hidden_layers, nb_steps=50, dev="cpu"):
        super(neural_subm_one_level, self).__init__()
        self.num_feature = num_feature
        self.in_d =in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand = InnerUMNN(num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device)
        self.offset = torch.nn.Linear(1, 1, bias=True).to(self.device)
        self.splus =  nn.Softplus()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
        self.reset_parameters()
        
    def reset_parameters(self):
        self.log_weight.data = 0*self.log_weight.data+1

    def forward(self, x, h):
        
            
        # wt =   self.splus(self.log_weight)
        x = nn.functional.linear(x, self.log_weight,bias=None)

        if len(list(x.shape))==3:
            x = torch.sum(x,dim = [1,2]).unsqueeze(1)
        else:
            x = torch.sum(x,dim = 1).unsqueeze(1)
            
            
        x0 = torch.zeros(x.shape).to(self.device) 
        x = x.to(self.device)
        h = h.to(self.device)
#         offset = self.offset(torch.zeros(1).to(self.device))
#         offset = offset.to(self.device) + x0
        
        OuterIntegralOut = ParallelNeuralIntegral.apply(x0, x, self.OuterIntegrand,
                                                        _flatten(self.OuterIntegrand.parameters()), h, self.nb_steps) 

        return  OuterIntegralOut
    


class InnerUMNN(nn.Module):
    def __init__(self, num_feature, in_d,
                 hidden_layers, nb_steps=50, dev="cpu"):
        super(InnerUMNN, self).__init__()
        
        self.device = dev
        self.nb_steps = nb_steps
        self.Integrand = IntegrandNN(in_d, hidden_layers)
        # self.xmax_up = xmax_up
        
    def forward(self, x, h):
        xmax = torch.max(x)+10
        xfinal = torch.zeros(x.shape).to(self.device)+xmax
        x = x.to(self.device)
        h = h.to(self.device)
        OuterIntegralOut = ParallelNeuralIntegral.apply(x, xfinal, self.Integrand,
                                                        _flatten(self.Integrand.parameters()), h, self.nb_steps)
        
#         if torch.norm(OuterIntegralOut) > 10:
#             OuterIntegralOut = 10*OuterIntegralOut/(0.01+OuterIntegralOut)
#         OuterIntegralOut[OuterIntegralOut>10]=10
#         OuterIntegralOut=torch.clamp(OuterIntegralOut,min=0, max=2)
#         print(OuterIntegralOut)
        return  OuterIntegralOut 

class InnerUMNN_neg(nn.Module):
    def __init__(self, num_feature, in_d,
                 hidden_layers, nb_steps=50, dev="cpu"):
        super(InnerUMNN_neg, self).__init__()
        
        self.device = dev
        self.nb_steps = nb_steps
        self.Integrand = IntegrandNN_zero_stable(in_d, hidden_layers)

        
    def forward(self, x, h):
        x = x.to(self.device)
        h = h.to(self.device)
        x0 = torch.zeros(x.shape).to(self.device) 

        OuterIntegralOut =  ParallelNeuralIntegral.apply(x0, x, self.Integrand,
                                                        _flatten(self.Integrand.parameters()), h, self.nb_steps)
        
        # if torch.norm(OuterIntegralOut) > 10:
        #     OuterIntegralOut = 10*OuterIntegralOut/(0.01+OuterIntegralOut)
#         OuterIntegralOut[OuterIntegralOut>10]=10
#         OuterIntegralOut=torch.clamp(OuterIntegralOut,min=0, max=2)
#         print(OuterIntegralOut)
        return  OuterIntegralOut 
        
class subnet_flex_dual_alpha(nn.Module):
    def __init__(self, num_feature, in_d, 
                 hidden_layers, nb_steps=50, dev="cpu",num_rnn_steps=2,alpha_val=0.9):
        super(subnet_flex_dual_alpha, self).__init__()
        self.method_name = "subnet_flex_dual_alpha"
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d =in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand_First_out = IntegrandNN_outer_level(in_d, hidden_layers)
        self.OuterIntegrand_Second_Out = IntegrandNN(in_d, hidden_layers)
        self.alpha_val = alpha_val
#         self.splus =  nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
#         self.log_weight_t = nn.Parameter(torch.Tensor(1, self.num_rnn_steps))    
        self.sigmoid_param =  nn.Parameter(torch.Tensor(1))   
        self.reset_parameters()
        

        
    def reset_parameters(self):
        
        self.log_weight.data = 0*self.log_weight.data+1
#         self.log_weight_t.data =  0*self.log_weight_t.data+1

            
    def compute_modular_function(self, x_, h_ , n):
        
#         log_weight_t =   self.log_weight_t
#         one_hot_t = F.one_hot(torch.tensor(n), num_classes =  self.num_rnn_steps)
#         one_hot_t = one_hot_t.float() 
#         t_vec = nn.functional.linear(one_hot_t.to(self.device), log_weight_t.to(self.device),bias=None)

        
        x = nn.functional.linear(x_, self.log_weight,bias = None)

#         x = x  + 0* t_vec
        
        if len(list(x.shape))==3:
            x = torch.sum(x,dim = [1,2]).unsqueeze(1)
        else:
            x = torch.sum(x,dim = 1).unsqueeze(1)
            
            
        x = x.to(self.device)
        h = h_.to(self.device)
        
        return x, h

    def forward(self, x_, h_):
        
        '''
        dphi/dx = gamma(x)
        dgamma/dx = F(x)
        '''
        
        OuterIntegral_First_out = 0
        OuterIntegral_Second_Out = 0
        for n in range(self.num_rnn_steps):

            x, h = self.compute_modular_function( x_, h_ , n)
            x_input_second_integral = x
            x_input_to_first_integrand = x
            x0 = torch.zeros(x.shape).to(self.device) 
            
            lam =  self.sigmoid(self.sigmoid_param)
            x = (1-lam) * x +  lam * OuterIntegral_First_out
            
            x_input_second_integral = (1-lam) * x_input_second_integral +  lam * OuterIntegral_Second_Out
            
            # x_input_to_first_integrand = (1-lam)* x_input_to_first_integrand + lam * self.OuterIntegrand_First_out(x_input_to_first_integrand,h)
            output_from_first_integrand =  self.OuterIntegrand_First_out(x_input_second_integral,h)
            
#             x = x
            OuterIntegral_First_out = ParallelNeuralIntegral.apply(x0, x, self.OuterIntegrand_First_out,
                                                            _flatten(self.OuterIntegrand_First_out.parameters()), h, self.nb_steps) 
            xmax = torch.max(x_input_second_integral)+10
            xfinal = torch.zeros(x.shape).to(self.device)+xmax
            OuterIntegral_Second_Out = (1/self.alpha_val)*ParallelNeuralIntegral.apply(x_input_second_integral, xfinal, self.OuterIntegrand_Second_Out,
                                                            _flatten(self.OuterIntegrand_Second_Out.parameters()), h, self.nb_steps) 

        # return  OuterIntegral_First_out, x_input_to_first_integrand, OuterIntegral_Second_Out
        return  OuterIntegral_First_out, output_from_first_integrand, OuterIntegral_Second_Out

class subnet_flex_dual(nn.Module):
    def __init__(self, num_feature, in_d, 
                 hidden_layers, nb_steps=50, dev="cpu",num_rnn_steps=2):
        super(subnet_flex_dual, self).__init__()
        self.method_name = "subnet_flex_dual"
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d =in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.OuterIntegrand_First_out = IntegrandNN_outer_level(in_d, hidden_layers)
        self.OuterIntegrand_Second_Out = IntegrandNN(in_d, hidden_layers)
        
#         self.splus =  nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.log_weight = nn.Parameter(torch.Tensor(1, self.num_feature))
#         self.log_weight_t = nn.Parameter(torch.Tensor(1, self.num_rnn_steps))    
        self.sigmoid_param =  nn.Parameter(torch.Tensor(1))   
        self.reset_parameters()
        

        
    def reset_parameters(self):
        
        self.log_weight.data = 0*self.log_weight.data+1
#         self.log_weight_t.data =  0*self.log_weight_t.data+1

            
    def compute_modular_function(self, x_, h_ , n):
        
#         log_weight_t =   self.log_weight_t
#         one_hot_t = F.one_hot(torch.tensor(n), num_classes =  self.num_rnn_steps)
#         one_hot_t = one_hot_t.float() 
#         t_vec = nn.functional.linear(one_hot_t.to(self.device), log_weight_t.to(self.device),bias=None)

        
        x = nn.functional.linear(x_, self.log_weight,bias = None)

#         x = x  + 0* t_vec
        
        if len(list(x.shape))==3:
            x = torch.sum(x,dim = [1,2]).unsqueeze(1)
        else:
            x = torch.sum(x,dim = 1).unsqueeze(1)
            
            
        x = x.to(self.device)
        h = h_.to(self.device)
        
        return x, h

    def forward(self, x_, h_):
        
        '''
        dphi/dx = gamma(x)
        dgamma/dx = F(x)
        '''
        
        OuterIntegral_First_out = 0
        OuterIntegral_Second_Out = 0
        for n in range(self.num_rnn_steps):

            x, h = self.compute_modular_function( x_, h_ , n)
            x_input_second_integral = x
            x_input_to_first_integrand = x
            x0 = torch.zeros(x.shape).to(self.device) 
            
            lam =  self.sigmoid(self.sigmoid_param)
            x = (1-lam) * x +  lam * OuterIntegral_First_out
            
            x_input_second_integral = (1-lam) * x_input_second_integral +  lam * OuterIntegral_Second_Out
            
            # x_input_to_first_integrand = (1-lam)* x_input_to_first_integrand + lam * self.OuterIntegrand_First_out(x_input_to_first_integrand,h)
            output_from_first_integrand =  self.OuterIntegrand_First_out(x_input_second_integral,h)
            
#             x = x
            OuterIntegral_First_out = ParallelNeuralIntegral.apply(x0, x, self.OuterIntegrand_First_out,
                                                            _flatten(self.OuterIntegrand_First_out.parameters()), h, self.nb_steps) 
            xmax = torch.max(x_input_second_integral)+10
            xfinal = torch.zeros(x.shape).to(self.device)+xmax
            OuterIntegral_Second_Out = ParallelNeuralIntegral.apply(x_input_second_integral, xfinal, self.OuterIntegrand_Second_Out,
                                                            _flatten(self.OuterIntegrand_Second_Out.parameters()), h, self.nb_steps) 

        # return  OuterIntegral_First_out, x_input_to_first_integrand, OuterIntegral_Second_Out
        return  OuterIntegral_First_out, output_from_first_integrand, OuterIntegral_Second_Out


 
class subnet_flex_dual_non_monotone(nn.Module):
    def __init__(self, num_feature, in_d, 
                 hidden_layers, nb_steps=50, dev="cpu",num_rnn_steps=1,thrs=45000,two_integrals=True):
        super(subnet_flex_dual_non_monotone, self).__init__()
        self.method_name = "subnet_flex_dual_non_monotone"
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d =in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.first_subnet = subnet_flex_dual(num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device,num_rnn_steps=1)
        self.second_subnet = subnet_flex_dual(num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device,num_rnn_steps=1)
        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()
        self.thrs=thrs
        self.two_integrals = two_integrals
        self.offset = 0.8
        
    def reset_parameters(self):
        
        
        self.bias.data = 0*self.bias.data+6.0

    def rebatch(self,x,h):
        x_sum = torch.sum(x,dim=[1,2])
        true = x_sum>self.thrs

        a = torch.nonzero(true.int()).squeeze(1)
        b = torch.nonzero(1-true.int()).squeeze(1)

        x_less_ths = x[b]
        x_higher_ths = x[a]
        return x_less_ths, x_higher_ths, h[b],h[a],b,a


    def forward(self, x_, h_):
        
        '''
        dphi/dx = gamma(x)
        dgamma/dx = F(x)
        '''
        x_less_ths,x_higher_ths,h_1, h_2,b,a = self.rebatch(x_,h_)

        y_pred = torch.empty(x_.shape[0],1).to(self.device)
        gamma_x = torch.empty(x_.shape[0],1).to(self.device)
        F_x =  torch.empty(x_.shape[0],1).to(self.device)

        if x_less_ths.shape[0]>0:
            y_pred_1, gamma_x_1, F_x_1 =  self.first_subnet(x_less_ths,h_1)
            y_pred[b] = y_pred_1
            gamma_x[b] = gamma_x_1
            F_x[b] = F_x_1

        if x_higher_ths.shape[0]>0:

            if self.two_integrals== True:
                y_pred_2, gamma_x_2, F_x_2 =  self.second_subnet(0.8-x_higher_ths,h_2) 
            else:
                y_pred_2, gamma_x_2, F_x_2 =  self.first_subnet(0.8-x_higher_ths,h_2) 

            y_pred[a] = y_pred_2
            gamma_x[a] = gamma_x_2
            F_x[a] = F_x_2

        return  y_pred, gamma_x, F_x

class subnet_flex_dual_non_monotone_switch(nn.Module):
    def __init__(self, num_feature, in_d, 
                 hidden_layers, nb_steps=50, dev="cpu",num_rnn_steps=1):
        super(subnet_flex_dual_non_monotone_switch, self).__init__()
        self.method_name = "subnet_flex_dual_non_monotone"
        self.num_rnn_steps = num_rnn_steps
        self.num_feature = num_feature
        self.in_d =in_d
        self.device = dev
        self.nb_steps = nb_steps
        self.first_subnet = subnet_flex_dual(num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device,num_rnn_steps=1)
        self.second_subnet = subnet_flex_dual(num_feature, in_d, hidden_layers, nb_steps=50, dev=self.device,num_rnn_steps=1)
        self.relu = nn.ReLU()
        self.shift_2c = nn.Parameter(torch.Tensor(1))
        self.param_1= nn.Parameter(torch.Tensor(1))
        self.param_2= nn.Parameter(torch.Tensor(1))

        self.reset_parameters()
        

        
    def reset_parameters(self):
        
        
        self.shift_2c.data = 0*self.shift_2c.data+0.8
        self.param_1.data = 0*self.param_1.data+1
        self.param_2.data = 0*self.param_2.data+0.7

    def rebatch(self,x,h):
        x_sum = torch.sum(x,dim=[1,2])
        true = x_sum>45000

        a = torch.nonzero(true.int()).squeeze(1)
        b = torch.nonzero(1-true.int()).squeeze(1)

        x_less_ths = x[b]
        x_higher_ths = x[a]
        return x_less_ths, x_higher_ths, h[b],h[a],b,a


    def forward(self, x_, h_):
        
        '''
        dphi/dx = gamma(x)
        dgamma/dx = F(x)
        '''
        x_sum = torch.sum(x_,dim=[1,2])
        switch = torch.zeros(x_sum.shape[0],1).to(self.device)
        switch[45000-x_sum<=0]=0
        switch[45000-x_sum>0]=1
        y_pred, gamma_x, F_x =  tuple(self.first_subnet(x_,h_)[i]*switch + (1-switch)*self.relu(self.second_subnet(self.shift_2c-x_,h_)[i]) for i in range(3))
        return  y_pred, gamma_x, F_x
 
