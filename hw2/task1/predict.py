import sys
sys.path = ["/home/dada/.local/lib/python3.5/site-packages"] + sys.path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models import MF


stored = torch.load('pytorch_model.pt')
df = stored['train_csv']
global_mean = stored['global_mean']
state = stored['state_dict']

all_food_list = np.array(df.foodid.unique())
eaten_dict = {uid:set(df_.foodid.unique()) for uid, df_ in df.groupby("userid")}
query = {uid:(Variable(torch.from_numpy(np.array([uid]*len(all_food_list)))).cuda(), 
          Variable(torch.from_numpy(all_food_list)).cuda()) for uid in df.userid.unique()}

model = MF(
        user_size=int(df.userid.max()+1),
        food_size=int(df.foodid.max()+1),
        hidden=200,
        global_mean=global_mean
        ).cuda()
model.load_state_dict(state)


def predict(M, uid):
    y_pred = []
    query_u, query_f = query[uid]
    rank = M(query_u, query_f).detach().cpu().numpy()
    rank = np.argsort(rank)[::-1]
    rank = all_food_list[rank]
    for j in rank:
        if j not in eaten_dict[uid]:
            y_pred.append(j)
        if len(y_pred) >= 20:
            break
    assert not len(set(y_pred) & eaten_dict[uid])
    return y_pred


submit = pd.DataFrame()
submit["userid"] = df.userid.unique()

preds = []
for uid in df.userid.unique():
    pred = predict(model, uid)
    pred = " ".join([str(j) for j in pred])
    preds.append(pred)
submit["foodid"] = preds
submit.to_csv("submit.csv", index=False)
