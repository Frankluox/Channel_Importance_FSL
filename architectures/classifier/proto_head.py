import torch
import torch.nn as nn
import torch.nn.functional as F
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



class PN_head(nn.Module):
    r"""The metric-based protypical classifier from ``Prototypical Networks for Few-shot Learning''.
    """
    def __init__(self, use_Oracle, statistics_root,dataset_name) -> None:
        super().__init__()
        self.use_Oracle = use_Oracle
        if self.use_Oracle:
            self.mean = torch.from_numpy(np.load(os.path.join(statistics_root, "meanof"+dataset_name+".npy")))
            self.std = torch.from_numpy(np.load(os.path.join(statistics_root, "stdof"+dataset_name+".npy")))
            self.num = np.load(os.path.join(statistics_root, "numof"+dataset_name+".npy")).tolist()
            self.abs_mean = torch.from_numpy(np.load(os.path.join(statistics_root, "abs_meanof"+dataset_name+".npy")))

    def forward(self, features_test: Tensor, features_train: Tensor, 
                way: int, shot: int, use_simple: bool, use_Oracle: bool, all_labels: list = None) -> Tensor:
        r"""Take batches of few-shot training examples and testing examples as input,
            output the logits of each testing examples.

        Args:
            features_test: Testing examples. size: [batch_size, num_query, c, h, w]
            features_train: Training examples which has labels like:[abcdabcdabcd].
                            size: [batch_size, way*shot, c, h, w]
            way: The number of classes of each few-shot classification task.
            shot: The number of training images per class in each few-shot classification
                  task.
            use_simple: whether use the simple transformation
        Output:
            classification_scores: The calculated logits of testing examples.
                                   size: [batch_size, num_query, way]
        """
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
            proportion = all_mean*Oracle_importance/torch.pow(all_mean+1e-12,2)

            # see appendix F, sometimes this helps
            # single = torch.randn_like(proportion).fill_(1.)
            # proportion = torch.where(proportion>50.,single,proportion)#others

        
            proportion_train = proportion.unsqueeze(0).unsqueeze(0).repeat(features_train.size(0),features_train.size(1),1)
            proportion_test = proportion.unsqueeze(0).unsqueeze(0).repeat(features_test.size(0),features_test.size(1),1)

            features_train = features_train*proportion_train
            features_test = features_test*proportion_test


        batch_size = features_train.size(0)
            
        if use_simple:
            features_train = simple_transform(features_train,1.3)
            features_test = simple_transform(features_test,1.3)
        

        features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)

        #prototypes: [batch_size, way, c]
        prototypes = torch.mean(features_train.reshape(batch_size, shot, way, -1),dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=2, eps=1e-12)
        classification_scores = torch.bmm(features_test, prototypes.transpose(1, 2))

        return classification_scores

def create_model(**kwargs):
    return PN_head(**kwargs)
