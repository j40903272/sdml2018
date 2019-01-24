import sys
sys.path = ["/home/dada/.local/lib/python3.5/site-packages"] + sys.path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel():
    def __init__(self):
        df = pd.read_csv("train_rating.csv")
        df = df.drop("date", axis=1).drop_duplicates()
        self.food_rank = df.foodid.value_counts().index.tolist()
    
    def predict(self, userid):
        pred = []
        already_have = eaten_dict[userid]
        for i in self.food_rank:
            if i not in already_have:
                pred.append(i)
            if len(pred) >= 20:
                break
        return pred

    

class NeuMF(nn.Module):
    def __init__(self, user_size, food_size, hidden=200):
        super(NeuMF, self).__init__()
        self.user_embed_mf = nn.Embedding(user_size, hidden, padding_idx=0)
        self.food_embed_mf = nn.Embedding(food_size, hidden, padding_idx=0)
        self.user_embed_mlp = nn.Embedding(user_size, hidden, padding_idx=0)
        self.food_embed_mlp = nn.Embedding(food_size, hidden, padding_idx=0)
        
        self.fc = nn.Linear(hidden+hidden//4, 1)
        self.fc1 = nn.Linear(hidden*2, hidden)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.fc3 = nn.Linear(hidden//2, hidden//4)

    def forward(self, user, food, test=False):
        u_mf = self.user_embed_mf(user)
        f_mf = self.food_embed_mf(food)
        u_mlp = self.user_embed_mlp(user)
        f_mlp = self.food_embed_mlp(food)
        ### mf
        product = torch.mul(u_mf, f_mf)
        mf = F.dropout(product, 0.25)
        ### mlp
        concat = torch.cat([u_mlp, f_mlp], dim=-1)
        x = F.relu(self.fc1(concat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mlp = F.dropout(x, 0.25)
        ### concat
        vec = torch.cat([mf, mlp], dim=-1)
        logits = self.fc(vec)
        output = logits.squeeze()
        return output
    
    
class GMF(nn.Module):
    def __init__(self, user_size, food_size, hidden=10):
        super(GMF, self).__init__()
        self.user_embed = nn.Embedding(user_size, hidden, padding_idx=0)
        self.food_embed = nn.Embedding(food_size, hidden, padding_idx=0)
        self.fc1 = nn.Linear(hidden, 1)

    def forward(self, user, food, test=False):
        u = self.user_embed(user)
        f = self.food_embed(food)
        product = F.dropout(u*f, 0.3)
        x = self.fc1(product)
        output = x.squeeze()
        return output
    
    
class MLP(nn.Module):
    def __init__(self, user_size, food_size, hidden=100, global_mean=0):
        super(MLP, self).__init__()
        self.user_embed = nn.Embedding(user_size, hidden, padding_idx=0)
        self.food_embed = nn.Embedding(food_size, hidden, padding_idx=0)
    
        self.fc1 = nn.Linear(hidden*2, hidden)
        self.fc2 = nn.Linear(hidden, hidden//2)
        self.fc3 = nn.Linear(hidden//2, hidden//4)
        self.fc4 = nn.Linear(hidden//4, 1)
        
    def forward(self, user, food, test=False):
        u = self.user_embed(user)
        b = self.food_embed(food)
        concat = torch.cat([u, b], dim=-1)
        x = F.dropout(concat, 0.3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.3)
        x = self.fc4(x)
        output = x.squeeze()
        return output
    
    
class MF(nn.Module):
    def __init__(self, user_size, food_size, hidden=200, global_mean=1.):
        super(MF, self).__init__()
        self.user_embed = nn.Embedding(user_size, hidden, padding_idx=0)
        self.food_embed = nn.Embedding(food_size, hidden, padding_idx=0)
        self.user_bias = nn.Embedding(user_size, 1, padding_idx=0)
        self.food_bias = nn.Embedding(food_size, 1, padding_idx=0)
        self.global_mean = global_mean
        self.user_bias.weight.data.fill_(0)
        self.food_bias.weight.data.fill_(0)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.food_embed.weight)
    
    def forward(self, user, food, test=False):
        u = self.user_embed(user)
        u_b = self.user_bias(user).squeeze()
        b = self.food_embed(food)
        b_b = self.food_bias(food).squeeze()
        output = (u*b).sum(1) + u_b + b_b + self.global_mean
        return output
    