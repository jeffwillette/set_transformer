import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from modules import PermEquiMax, PermEquiMax2, PermEquiMean, PermEquiMean2


def list2string(l):
    delim = '_'
    return delim.join(map(str, l))


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        
    def forward(self, x):
        return self.linear(x)

    def flops(self):
        #NOTE: We ignore activation funcitons.
        MAC = self.out_features * self.in_features
        ADD = 0
        if self.bias:
            ADD = self.out_features
        flops = 2 * MAC + ADD
        return flops

class BMM(nn.Module):
    def __init__(self):
        super(BMM, self).__init__()
        self.A_m = 0
        self.A_n = 0
        self.B_p = 0

    def forward(self, A, B):
        if not self.training:
            if self.A_m == 0:
                _, self.A_m, self.A_n = A.size()
                _, _, self.B_p = B.size()
        
        return torch.bmm(A, B)

    def flops(self):
        return 2 * self.A_m * self.A_n * self.B_p


class SlotSetEncoder(nn.Module):
    def __init__(self, K, h, d, d_hat, eps=1e-8, _slots='Random'):
        super(SlotSetEncoder, self).__init__()
        self.K = K                                              #Number of Slots
        self.h = h                                              #Slot Size
        self.d = d                                              #Input Dimension
        self.d_hat = d_hat                                      #Linear Projection Dimension
        self.eps = eps                                          #Additive epsilon for stability
        self._slots = _slots                                    #Use Random or Learned Slots
        self.sqrt_d_hat = 1.0 / math.sqrt(d_hat)                #Normalization Term
        
        if _slots == 'Random':
            self.sigma = nn.Parameter(torch.rand(1, 1, h))
            self.mu    = nn.Parameter(torch.rand(1, 1 ,h))
        elif _slots == 'Learned':
            self.S = nn.Parameter(torch.rand(1, K, h))
        else:
            raise ValueError('{} not implemented for slots'.format(_slots))
        
        self.gru = nn.GRUCell(d_hat, d_hat)

        self.k = Linear(in_features=d, out_features=d_hat, bias=False)
        self.q = Linear(in_features=h, out_features=d_hat, bias=False)
        self.v = Linear(in_features=d, out_features=d_hat, bias=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_hat, d_hat),
            nn.ReLU(inplace = True),
            nn.Linear(d_hat, d_hat)
        )

        self.norm_slots = nn.LayerNorm(normalized_shape=h)
        self.bmm_attn = BMM()
        self.bmm_upda = BMM()

    def get_attn_loss(self):
        return self.attn_loss

    def forward(self, X, S=None):
        N, n, d, device = *(X.size()), X.device
        
        #Sample Slots Based on N
        if S is None:   #S \in R^{N x K xh}
            if self._slots == 'Random':
                S = torch.normal(self.mu.repeat(N, self.K, 1).to(device), torch.exp(self.sigma.repeat(N, self.K, 1).to(device)))
            else:
                S = self.S.repeat(N, 1, 1)
            
        # S = self.norm_slots(S)

        #Linear Projections
        k = self.k(X)   #k \in R^{N x n x d_hat}
        v = self.v(X)   #v \in R^{N x n x d_hat}
        q = self.q(S)   #q \in R^{N x K x d_hat}
        
        #Compute M
        M = self.sqrt_d_hat * self.bmm_attn(k, q.transpose(1, 2)) #M \in R^{N x n x K}
        
        #Compute sigmoid attention
        attn = torch.sigmoid(M) + self.eps         #attn \in R^{N x n x K}
        # self.attn_loss = attn.mean()
        self.attn_loss = torch.tensor(0)
        
        #Compute attention weights
        W = attn / attn.sum(dim=2, keepdims=True)   #W \in R^{N x n x K}
        
        #Compute S_hat
        S_hat = self.bmm_upda(W.transpose(1, 2), v)     #S_hat \in R^{N x K x d_hat}
        return S_hat 

    def check_minibatch_consistency_random_slots(self, X, split_size):
        N, n, d, device = *(X.size()), X.device
        
        #Sample Slots for Current S
        S = torch.normal(self.sigma.repeat(N, self.K, 1).to(device), torch.log(self.mu.repeat(N, self.K, 1).to(device)))
        
        #Encode full set
        S_hat_X = self.forward(X=X, S=S)
        
        #Split X each with split_size elements.
        X = torch.split(X, split_size_or_sections=split_size, dim=1)

        #Encode splits
        S_hat_split = torch.zeros(N, self.K, self.d_hat).to(device)
        for split_i in X:
            S_hat_split_i = self.forward(X=split_i, S=S)
            S_hat_split = S_hat_split + S_hat_split_i 
        
        consistency = torch.all(torch.isclose(S_hat_X, S_hat_split, rtol=1e-4))
        print('Random  Slot Encoder is MiniBatch Consistent               : ', consistency)
    
    def check_minibatch_consistency_learned_slots(self, X, split_size):
        N, n, d, device = *(X.size()), X.device
        
        S = self.S.repeat(N, 1, 1)

        #Encode full set
        S_hat_X = self.forward(X=X, S=S)
        
        #Split X each with split_size elements.
        X = torch.split(X, split_size_or_sections=split_size, dim=1)

        #Encode splits
        S_hat_split = torch.zeros(N, self.K, self.d_hat).to(device)
        for split_i in X:
            S_hat_split_i = self.forward(X=split_i, S=S)
            S_hat_split = S_hat_split + S_hat_split_i 
        consistency = torch.all(torch.isclose(S_hat_X, S_hat_split, rtol=1e-4))     #NOTE: Learned Slots are consistent with low precision.
        print('Learned Slot Encoder is MiniBatch Consistent               : ', consistency)
    
    def check_input_invariance_random_slots(self, X):
        N, n, d, device = *(X.size()), X.device
        
        #Sample Slots for Current S
        S = torch.normal(self.sigma.repeat(N, self.K, 1).to(device), torch.log(self.mu.repeat(N, self.K, 1).to(device)))
        
        #Encode full set
        S_hat = self.forward(X=X, S=S)
        
        #Random permutations on X
        permutations = torch.randperm(n)
        X = X[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)
        
        consistency = torch.all(torch.isclose(S_hat, S_hat_perm, rtol=1e-4))
        print('Random  Slot Encoder is Permutation Invariant w.r.t Input  : ', consistency)
    
    def check_input_invariance_learned_slots(self, X):
        N, n, d, device = *(X.size()), X.device
        
        S = self.S.repeat(N, 1, 1)
        
        #Encode full set
        S_hat = self.forward(X=X, S=S)
        
        #Random permutations on X
        permutations = torch.randperm(n)
        X = X[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)
        
        consistency = torch.all(torch.isclose(S_hat, S_hat_perm))
        print('Learned Slot Encoder is Permutation Invariant w.r.t Input  : ', consistency)
 
    def check_slot_equivariance_random_slots(self, X):
        N, n, d, device = *(X.size()), X.device
        
        #Sample Slots for Current S
        S = torch.normal(self.sigma.repeat(N, self.K, 1).to(device), torch.log(self.mu.repeat(N, self.K, 1).to(device)))
        
        #Encode full set
        S_hat = self.forward(X=X, S=S)
        
        #Random permutations on S
        permutations = torch.randperm(self.K)
        S = S[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)

        #Apply sampe permutation on S_hat
        S_hat = S_hat[:, permutations, :]
        
        consistency = torch.all(torch.isclose(S_hat, S_hat_perm))
        print('Random  Slot Encoder is Permutation Equivariant w.r.t Slots: ', consistency)
    
    def check_slot_equivariance_learned_slots(self, X):
        N, n, d, device = *(X.size()), X.device
        
        S = self.S.repeat(N, 1, 1)
        
        #Encode full set
        S_hat = self.forward(X=X, S=S)
        
        #Random permutations on S
        permutations = torch.randperm(self.K)
        S = S[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)

        #Apply sampe permutation on S_hat
        S_hat = S_hat[:, permutations, :]
        
        consistency = torch.all(torch.isclose(S_hat, S_hat_perm))
        print('Learned Slot Encoder is Permutation Equivariant w.r.t Slots: ', consistency)


class ConsistentAggregator(nn.Module):
    def __init__(self, K, h, d, d_hat, g='sum', ln=True, _slots='Random'):
        super(ConsistentAggregator, self).__init__()
        self.K = K                  #Number of slots in each stage
        self.h = h                  #The dimension of each slot
        self.d = d                  #Input dimension to each stage
        self.g = g                  #Choice of aggregation function g: sum, mean, max, min
        self.d_hat = d_hat          #Projection dimension in each stage
        self.ln = ln                #Use LayerNorm or Not

        self.name = 'Consistent/{}/{}/{}/{}/{}'.format(_slots, list2string(K), list2string(h), list2string(d), list2string(d_hat))

        self.enc = []
        for i in range(len(K)):
            self.enc.extend([SlotSetEncoder(K=K[i], h=h[i], d=d[i], d_hat=d_hat[i])]) 
        self.enc = nn.Sequential(*self.enc)

        if self.ln:
            self.norm = nn.LayerNorm(normalized_shape=d_hat[-1])
            self.name = '{}/{}'.format(self.name, 'LN')
        else:
            self.name = '{}/{}'.format(self.name, 'NO_LN')

    def attn_loss(self):
        loss = 0
        for sse in self.enc:
            loss += sse.get_attn_loss()
        return loss

    def forward(self, x, split_size=None):
        if split_size is None:
            enc = self.enc(x)
            if enc.size(1) > 1:
                #This is for the modelnet models where we dont necessarily have to directly go to 1.
                if self.g == 'mean':
                    enc = enc.mean(dim=1, keepdims=True)
                elif self.g == 'sum':
                    enc = enc.sum(dim=1, keepdims=True)
                elif self.g == 'max':
                    enc, _ = enc.max(dim=1, keepdims=True)
                elif self.g == 'min':
                    enc, _ = enc.min(dim=1, keepdims=True)
                elif self.g is None:
                    pass
                else:
                    raise NotImplementedError
            return self.norm(enc.squeeze(1)) if self.ln else enc.squeeze(1)
        else:
            B, _, _, device = *x.size(), x.device
            x = torch.split(x, split_size_or_sections=split_size, dim=1)
            
            enc = []
            for split in x:
                enc.append(self.enc(split))
            enc = torch.cat(enc, dim=1)
            #NOTE:This Does not affect training!
            if self.g == 'mean':
                enc = enc.mean(dim=1, keepdims=True)
            elif self.g == 'sum':
                enc = enc.sum(dim=1, keepdims=True)
            elif self.g == 'max':
                enc, _ = enc.max(dim=1, keepdims=True)
            elif self.g == 'min':
                enc, _ = enc.min(dim=1, keepdims=True)
            elif self.g is None:
                pass
            else:
                raise NotImplementedError
            return self.norm(enc.squeeze(1)) if self.ln else enc.squeeze(1)

    #TODO: Check permutation invariance/equivariance -- In theory it should be since SlotSetEncoder is.
    def check_minibatch_consistency_learned(self, x, split_size):
        enc_full = self.forward(x)
        enc_split = self.forward(x, split_size=split_size)
        consistency = torch.all(torch.isclose(enc_full, enc_split, rtol=1e-2))
        print('Heirarchical Aggregator is MiniBatch Consistent? : ', consistency)


types = {'max': PermEquiMax, 'max2': PermEquiMax2, 'mean': PermEquiMean, 'mean2': PermEquiMean2, 'linear': nn.Linear}


class ConsistentModel(nn.Module):
    def __init__(self, in_dim=3, out_dim=40, num_outputs=1, hidden_dim=128, num_layers=4, extractor='max2', K=[16], h=[512], d=[256], d_hat=[256], g='max', ln=True, _slots='Random'):
        super(ConsistentModel, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.extractor = extractor
        self.num_outputs = num_outputs
        self.K = K
        self.h = h
        self.d = d
        self.d_hat = d_hat
        self.g = g
        self.ln = ln
        self._slots = _slots
        self.name = '{}'.format(extractor)

        activation = nn.ReLU
        layer = types[extractor]

        self.features = []
        for i in range(num_layers):
            if i == 0:
                in_features = in_dim
            else:
                in_features = hidden_dim
            self.features.append(layer(in_features, hidden_dim))
            if i != num_layers - 1:
                self.features.append(activation())
        self.features = nn.Sequential(*self.features)

        self.pool = ConsistentAggregator(K=K, h=h, d=d, d_hat=d_hat, g=g, ln=ln, _slots=_slots)

        self.dec = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, out_dim * self.num_outputs)
            # nn.Linear(self.hidden_dim, out_dim)
        )

        self.name = '{}/{}'.format(self.pool.name, self.name)
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def attn_loss(self):
        return self.pool.attn_loss()

    def forward(self, x):
        # print(x.size())
        features = self.features(x)
        # print(features.size())
        pool = self.pool(features)
        # print(pool.size())
        # pred = self.dec(pool)
        pred = self.dec(pool).reshape(-1, self.num_outputs, self.out_dim)
        # print(pred.size())

        return pred       


def check_consistent_modelnet():
    m = ConsistentModel(extractor='PermEquiMax')
    x = torch.rand(3, 500, 3)
    y = m(x)
    print(y.size())


def check_minibatch_consistency_learned():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K     = [256, 128, 32, 16, 1]           #Heirarchy with reducing number of slots.
    h     = [256, 256, 128, 128, 128]       #The dimension of slots in each heirarchy.
    d_hat = [128, 256, 128, 64, 128]        #Projection dimension in each heirarchy
    d     = [64, 128, 256, 128, 64]         #Input dimension to each heirarchy

    x = torch.rand(256, 200, 64)

    model = ConsistentAggregator(K, h, d, d_hat, _slots='Learned')
    model.check_minibatch_consistency_learned(x=x, split_size=20)

if __name__ == '__main__':
    #check_minibatch_consistency_learned()
    check_consistent_modelnet()
