from torch import nn
from torch.nn import functional as F
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor


def simple_transform(x, beta):
    x = 1/torch.pow(torch.log(1/x+1),beta)
    return x

def extended_simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1),beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1),beta)
    return x_pos+x_neg

class Logistic_Regression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_test: Tensor, features_train: Tensor, 
                way: int, shot: int, use_simple: bool):
        assert features_train.size(0) == features_test.size(0) == 1

        features_train = torch.squeeze(features_train,0)
        features_test = torch.squeeze(features_test,0)

        

        if features_train.dim() == 4:
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)

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

def create_model():
    return Logistic_Regression()
