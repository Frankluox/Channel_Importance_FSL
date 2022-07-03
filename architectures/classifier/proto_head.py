import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
    def __init__(self) -> None:
        super().__init__()

    def forward(self, features_test: Tensor, features_train: Tensor, 
                way: int, shot: int, use_simple: bool) -> Tensor:
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
        if features_train.dim() == 5:
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)
        assert features_train.dim() == features_test.dim() == 3

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

def create_model():
    return PN_head()