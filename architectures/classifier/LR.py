from torch import nn
from torch.nn import functional as F
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor
import numpy as np
import os



def simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1),beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1),beta)
    return x_pos+x_neg

class Logistic_Regression(nn.Module):
    def __init__(self, use_Oracle, statistics_root=None,dataset_name=None):
        super().__init__()
        self.use_Oracle = use_Oracle
        if self.use_Oracle:
            self.mean = torch.from_numpy(np.load(os.path.join(statistics_root, "meanof"+dataset_name+".npy")))
            self.std = torch.from_numpy(np.load(os.path.join(statistics_root, "stdof"+dataset_name+".npy")))
            self.num = np.load(os.path.join(statistics_root, "numof"+dataset_name+".npy")).tolist()
            self.abs_mean = torch.from_numpy(np.load(os.path.join(statistics_root, "abs_meanof"+dataset_name+".npy")))


    def forward(self, features_test: Tensor, features_train: Tensor, 
                way: int, shot: int, use_simple: bool, use_Oracle: bool, all_labels: list = None):
        assert features_train.size(0) == features_test.size(0) == 1
        assert not (use_simple and use_Oracle)

        if features_train.dim() == 5:
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)
        if use_Oracle:

            assert way == 2
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

            # See appendix F, sometimes this helps a little.
            # single = torch.randn_like(proportion).fill_(1.)
            # proportion = torch.where(proportion>50.,single,proportion)#others

        
            proportion_train = proportion.unsqueeze(0).unsqueeze(0).repeat(features_train.size(0),features_train.size(1),1)
            proportion_test = proportion.unsqueeze(0).unsqueeze(0).repeat(features_test.size(0),features_test.size(1),1)

            features_train = features_train*proportion_train
            features_test = features_test*proportion_test

        
        features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)

        features_train = torch.squeeze(features_train,0)
        features_test = torch.squeeze(features_test,0)

        

        

        assert features_train.dim() == features_test.dim() == 2

        if use_simple:
            features_train = simple_transform(features_train,1.3)
            features_test = simple_transform(features_test,1.3)
    
        X_sup = features_train.cpu().detach().numpy()
        X_query = features_test.cpu().detach().numpy()
        label = torch.arange(way, dtype=torch.int8).repeat(shot).numpy()

        classifier = LogisticRegression(random_state=0, max_iter=1000).fit(X=X_sup, y=label)
        classification_scores = torch.from_numpy(classifier.predict_proba(X_query)).to(features_test.device)
        return classification_scores

def create_model(**kwargs):
    return Logistic_Regression(**kwargs)
