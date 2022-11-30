# %%
import torch
import torch.nn as nn
from torch.distributions import bernoulli, normal
import torch.nn.functional as F
# %%
# class naming convention: p_A_BC -> p(A|B,C)

####### Generative model / Decoder / Model network #######

class p_x_z(nn.Module):
    
    def __init__(self, nh, d, h, binfeats, contfeats):
        super().__init__()
        torch.manual_seed(1)
        self.nh = nh
        self.d = d
        self.h = h
        self.binfeats = binfeats
        self.contfeats = contfeats
        self.hx1 = nn.Linear(self.d, self.h)
        self.hx2 = nn.ModuleList(nn.Linear(self.h, self.h) for _ in range(self.nh))
        self.logits = nn.Linear(self.h, len(self.binfeats))
        self.mu = nn.Linear(self.h, len(self.contfeats))
        self.sigma = nn.Linear(self.h, len(self.contfeats))
        self.softplus = nn.Softplus()
        nn.init.xavier_uniform_(self.hx1.weight)
        nn.init.xavier_uniform_(self.logits.weight)
        nn.init.xavier_uniform_(self.mu.weight)
        nn.init.xavier_uniform_(self.sigma.weight)

    def forward(self, z):
        hx = F.elu(self.hx1(z)) # x1_post_eval,x2_post_eval: z_ph1=qz.mean()
        for j in range (self.nh):
            hx = F.elu(self.hx2[j](hx))
        logits = F.sigmoid(self.logits(hx))
        x_bin = bernoulli.Bernoulli(logits=logits)
        mu, sigma = F.elu(self.mu(hx)), self.softplus(self.sigma(hx))
        x_con = normal.Normal(loc=mu, scale=sigma)
        return x_bin, x_con

# %%
class p_t_z(nn.Module):

    def __init__(self, d, h):
        super().__init__()
        torch.manual_seed(1)
        self.d = d
        self.h = h
        self.logitst1 = nn.Linear(self.d, self.h)
        self.logitst2 = nn.Linear(self.h, 1)
        nn.init.xavier_uniform_(self.logitst1.weight)
        nn.init.xavier_uniform_(self.logitst2.weight)
    
    def forward(self, z):
        logitst = F.elu(self.logitst1(z))
        logitst = F.sigmoid(self.logitst2(logitst))
        t = bernoulli.Bernoulli(logits=logitst)
        return t

# %%
class p_y_zt(nn.Module):

    def __init__(self, d, nh, h):
        super().__init__()
        torch.manual_seed(1)
        self.d = d
        self.nh = nh
        self.h = h
        self.hy1_0 = nn.Linear(self.d, self.h)
        self.hy1_1 = nn.Linear(self.d, self.h)
        self.hy2_0 = nn.ModuleList(nn.Linear(self.h, self.h)for _ in range(self.nh))
        self.hy2_1 = nn.ModuleList(nn.Linear(self.h, self.h)for _ in range(self.nh))
        self.mu2_t0 = nn.Linear(self.h, 1)
        self.mu2_t1 = nn.Linear(self.h, 1)
        nn.init.xavier_uniform_(self.hy1_0.weight)
        nn.init.xavier_uniform_(self.hy1_1.weight)
        nn.init.xavier_uniform_(self.mu2_t0.weight)
        nn.init.xavier_uniform_(self.mu2_t1.weight)
 
    def forward(self, z, t):
        hy_0 = F.elu(self.hy1_0(z))
        for j in range(self.nh):
            hy_0 = F.elu(self.hy2_0[j](hy_0))
        mu2_t0 = F.elu(self.mu2_t0(hy_0))
        
        hy_1 = F.elu(self.hy1_1(z))
        for j in range(self.nh):
            hy_1 = F.elu(self.hy2_1[j](hy_1))
        mu2_t1 = F.elu(self.mu2_t1(hy_1))
        y = normal.Normal(loc=t * mu2_t1 + (1. - t) * mu2_t0, scale=torch.ones_like(mu2_t0)) # y_post,y_post_eval: t_ph1=t_ph
        return y

# %%
####### Inference model / Encoder #######

class q_t_x(nn.Module):

    def __init__(self, h, x_dim):
        super().__init__()
        torch.manual_seed(1)
        self.h = h
        self.x_dim = x_dim
        self.logits_t1 = nn.Linear(self.x_dim, self.h)
        self.logits_t2 = nn.Linear(self.h, 1)
        nn.init.xavier_uniform_(self.logits_t1.weight)
        nn.init.xavier_uniform_(self.logits_t2.weight)
    
    def forward(self, x):
        logits_t = F.elu(self.logits_t1(x))
        logits_t = F.sigmoid(self.logits_t2(logits_t))
        qt = bernoulli.Bernoulli(logits=logits_t)
        return qt

# %%
class q_y_xt(nn.Module):

    def __init__(self, nh, h, x_dim):
        super().__init__()
        torch.manual_seed(1)
        self.nh = nh
        self.h = h
        self.x_dim = x_dim
        self.hqy_1 = nn.Linear(self.x_dim, self.h)
        self.hqy_2 = nn.ModuleList(nn.Linear(self.h, self.h)for _ in range(self.nh))
        self.mu_qy_t0 = nn.Linear(self.h, 1)
        self.mu_qy_t1 = nn.Linear(self.h, 1)
        nn.init.xavier_uniform_(self.hqy_1.weight)
        nn.init.xavier_uniform_(self.mu_qy_t0.weight)
        nn.init.xavier_uniform_(self.mu_qy_t1.weight)
    
    def forward(self, x, qt):
        hqy = F.elu(self.hqy_1(x))
        for j in range(self.nh):
            hqy = F.elu(self.hqy_2[j](hqy))
        mu_qy_t0 = F.elu(self.mu_qy_t0(hqy))
        mu_qy_t1 = F.elu(self.mu_qy_t1(hqy))
        qy = normal.Normal(loc=qt * mu_qy_t1 + (1. - qt) * mu_qy_t0, scale=torch.ones_like(mu_qy_t0))
        return qy

# %%
class q_z_tyx(nn.Module):

    def __init__(self, nh, h, d, x_dim):
        super().__init__()
        torch.manual_seed(1)
        self.nh = nh
        self.h = h
        self.d = d
        self.x_dim = x_dim
        self.hqz1 = nn.Linear(self.x_dim+1, self.h)
        self.hqz2 = nn.ModuleList(nn.Linear(self.h, self.h)for _ in range(self.nh))
        self.muq_t0 = nn.Linear(self.h, self.d)
        self.sigmaq_t0 = nn.Linear(self.h, self.d)
        self.muq_t1 = nn.Linear(self.h, self.d)
        self.sigmaq_t1 = nn.Linear(self.h, self.d)
        self.softplus0 = nn.Softplus()
        self.softplus1 = nn.Softplus()
        nn.init.xavier_uniform_(self.hqz1.weight)
        nn.init.xavier_uniform_(self.muq_t0.weight)
        nn.init.xavier_uniform_(self.sigmaq_t0.weight)
        nn.init.xavier_uniform_(self.muq_t1.weight)
        nn.init.xavier_uniform_(self.sigmaq_t1.weight)

    def forward(self, x, qt, qy):
        inpt2 = torch.concat([x, qy], 1)
        hqz = F.elu(self.hqz1(inpt2))
        for j in range(self.nh):
            hqz = F.elu(self.hqz2[j](hqz))
        muq_t0 = F.elu(self.muq_t0(hqz))
        sigmaq_t0 = self.softplus0(self.sigmaq_t0(hqz))
        muq_t1 = F.elu(self.muq_t1(hqz))
        sigmaq_t1 = self.softplus1(self.sigmaq_t1(hqz))
        qz = normal.Normal(loc=qt * muq_t1 + (1. - qt) * muq_t0, scale=qt * sigmaq_t1 + (1. - qt) * sigmaq_t0)
        return qz