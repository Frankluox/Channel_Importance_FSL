import torch
import torch.nn as nn
from qpth.qp import QPFunction
import torch.nn.functional as F
import numpy as np
import os


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))





def simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1),beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1),beta)
    return x_pos+x_neg

def MetaOptNetHead_SVM_CS(query, support, n_way, n_shot, C_reg=0.1, double_precision=True, maxIter=3):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).
    This model is the classification head that we use for the final version.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      use_simple: whether use the simple transformation
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    support_labels = torch.eye(n_way).unsqueeze(0).repeat(n_shot,1,1).reshape(n_shot*n_way, -1).unsqueeze(0).repeat(support.size(0),1,1).to(support.device)

    # support_labels = torch.arange(n_way).unsqueeze(0).repeat(n_shot,1).reshape(-1).unsqueeze(0).repeat(support.size(0),1)
    # print(support_labels.shape)
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    #and C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    #This borrows the notation of liblinear.
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)


    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).to(support.device)

    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
    #This seems to help avoid PSD error from the QP solver.
    block_kernel_matrix += 1.0 * torch.eye(n_way*n_support).expand(tasks_per_batch, n_way*n_support, n_way*n_support).to(support.device)

    support_labels_one_hot = support_labels.reshape(tasks_per_batch, n_support * n_way)
    
    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot

    #This part is for the inequality constraints:
    #\alpha^m_i <= C^m_i \forall m,i
    #where C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    C = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support).to(support.device)
    h = C_reg * support_labels_one_hot

    #This part is for the equality constraints:
    #\sum_m \alpha^m_i=0 \forall i
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).to(support.device)

    A = batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).to(support.device))
    b = torch.zeros(tasks_per_batch, n_support).to(support.device)

    if double_precision:
        G, e, C, h, A, b = [x.double().to(support.device) for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float().to(support.device) for x in [G, e, C, h, A, b]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    # print(G.device)
    # print(e.detach().device)
    # print(C.detach().device)
    # print(h.detach().device)
    # print(A.detach().device)
    # print(b.detach().device)
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits

class MetaoptHead(nn.Module):
    def __init__(self, use_Oracle, statistics_root,dataset_name):
        super(MetaoptHead, self).__init__()
        self.head = MetaOptNetHead_SVM_CS
        self.use_Oracle = use_Oracle
        if self.use_Oracle:
            self.mean = torch.from_numpy(np.load(os.path.join(statistics_root, "meanof"+dataset_name+".npy")))
            self.std = torch.from_numpy(np.load(os.path.join(statistics_root, "stdof"+dataset_name+".npy")))
            self.num = np.load(os.path.join(statistics_root, "numof"+dataset_name+".npy")).tolist()
            self.abs_mean = torch.from_numpy(np.load(os.path.join(statistics_root, "abs_meanof"+dataset_name+".npy")))
        
    def forward(self, features_test, features_train, way, shot, use_simple, use_Oracle: bool, all_labels: list = None, **kwargs):
        assert not (use_simple and use_Oracle)
        if use_Oracle:
            assert features_train.size(0) == features_test.size(0) == 1
            assert way == 2

        if features_train.dim() == 5:
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)
        
        assert features_train.dim() == features_test.dim() == 3

        if use_Oracle:
            mean_1 = self.mean[all_labels[0]].to(features_train.device)
            mean_2 = self.mean[all_labels[1]].to(features_train.device)

            abs_mean_1 = self.abs_mean[all_labels[0]].to(features_train.device)
            abs_mean_2 = self.abs_mean[all_labels[1]].to(features_train.device)
            
            std_1 = self.std[all_labels[0]].to(features_train.device)
            std_2 = self.std[all_labels[1]].to(features_train.device)

            num_1 = self.num[all_labels[0]]
            num_2 = self.num[all_labels[1]]

            all_mean = (num_1*abs_mean_1+num_2*abs_mean_2)/(num_1+num_2)
            mean_difference = torch.abs(mean_1-mean_2)
            Oracle_importance = 2/((std_1+1e-12)/(mean_difference+1e-12)+(std_2+1e-12)/(mean_difference+1e-12))
            Oracle_importance = F.normalize(Oracle_importance, p=2, dim=0, eps=1e-12)
            
            # Equivalent to Oracle_importance/all_mean; avoid numerical issues
            proportion = all_mean*Oracle_importance/torch.pow(all_mean+1e-12,2)

            # see appendix F, sometimes this helps
            # single = torch.randn_like(proportion).fill_(1.)
            # proportion = torch.where(proportion>50.,single,proportion)#others

        
            proportion_train = proportion.unsqueeze(0).unsqueeze(0).repeat(features_train.size(0),features_train.size(1),1)
            proportion_test = proportion.unsqueeze(0).unsqueeze(0).repeat(features_test.size(0),features_test.size(1),1)

            features_train = features_train*proportion_train
            features_test = features_test*proportion_test

        if use_simple:
            features_test = simple_transform(features_test,1.3)
            features_train = simple_transform(features_train,1.3)
        
        features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)

        return self.head(features_test, features_train, way, shot, **kwargs)

def create_model(**kwargs):
    return MetaoptHead(**kwargs)


