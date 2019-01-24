
# coding: utf-8

# In[1]:


import sys
sys.path = ["/home/dada/.local/lib/python3.5/site-packages"] + sys.path
sys.path


# In[2]:


import pickle
import pandas as pd
import numpy as np
from mapk import mapk, apk
from tqdm import tqdm
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
torch.__version__


# In[4]:


from subprocess import call
import os
import time
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LNGUAGE"] = "en_US.UTF-8"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["KAGGLE_USERNAME"] = "b04902103"


# In[ ]:


df_rate = pd.read_csv("data/rating_train.csv")
df = df_rate.drop("date", axis=1)
df = df.drop_duplicates()
df["label"] = np.ones((len(df)))


eaten_dict = dict()
for i in df.groupby("userid"):
    uid = i[0]
    df_ = i[1]
    eaten_dict[uid] = set(df_.foodid.unique())
#get_ipython().run_line_magic('store', 'eaten_dict')
# with open("val_dict.pkl", "rb") as f:
#     val_dict = pickle.load(f)
# get_ipython().run_line_magic('store', '-r eaten_dict')
# get_ipython().run_line_magic('store', '-r val_dict')


# # negative sample

# In[ ]:


#from sklearn.preprocessing import scale
food_count = np.zeros((df.foodid.max()+1))
for i, j in zip(df.foodid.value_counts().index.values, df.foodid.value_counts().values):
    food_count[i] = j
#food_count = np.log(food_count+1)

user_count = np.zeros((df_rate.userid.max()+1))
for i, j in zip(df_rate.userid.value_counts().index.values, df_rate.userid.value_counts().values):
    user_count[i] = j
#user_count = np.log(user_count+1)


# In[ ]:


all_food_list = np.array(df.foodid.unique())
all_food_set = set(df.foodid.unique())
all_user_list = np.array(df.userid.unique())

not_eat_dict = dict()
food_prob = dict()
for i in eaten_dict:
    not_eat_dict[i] = all_food_set - eaten_dict[i]# - val_dict[i]
    tmp = list(not_eat_dict[i])
    prob = np.array([food_count[i] for i in tmp])
    food_prob[i] = prob/np.sum(prob)


# In[ ]:


class neg_sampler2():
    def __init__(self):
        self.neg_sample = {}
        self.neg_ptr = {}
        
        for i in df.userid.unique():
            self.neg_sample[i] = self.gen_neg_sample(i)
            
        for i in df.userid.unique():
            self.neg_ptr[i] = 0
    
    def get_neg_samples(self, uid_array):
        f = np.vectorize(lambda x:self.neg_sample[x][self.neg_ptr[x]])
        neg_samples = f(uid_array)
        
        for uid in uid_array:
            self.neg_ptr[uid] += 1
            if self.neg_ptr[uid] >= len(self.neg_sample[uid]):
                self.neg_ptr[uid] = 0
                self.neg_sample[uid] = self.gen_neg_sample(uid)
        return neg_samples
    
    def gen_neg_sample(self, uid):
        return np.random.choice(list(not_eat_dict[uid]), size=1024, p=food_prob[uid])#


# In[ ]:


# get_ipython().run_cell_magic('time', '', "\nfrom scipy import sparse\nN = neg_sampler2()\ninteractions = sparse.coo_matrix((df['label'].values,(df.userid.values,df.foodid.values)), \n                                 shape=(df.userid.max()+1, df.foodid.max()+1))")


# # evaluation

# In[ ]:


user_query = dict()
for i in df.userid.unique():
    userid = i
    user_query[userid] = (Variable(torch.from_numpy(np.array([userid]*len(all_food_list)))).cuda(), 
                          Variable(torch.from_numpy(np.array(all_food_list))).cuda())


# In[ ]:


# def cal(M, flag=False):
#     M.eval()
#     score = []
#     for i in user_query:
#         userid = i
#         already_have = eaten_dict[userid]
#         y_true = val_dict[userid]-eaten_dict[userid]
#         y_pred = []
        
#         query_u, query_f = user_query[userid]
#         rank = M(query_u, query_f, True).detach().cpu().data.numpy()
#         rank = np.argsort(rank)[::-1]
#         rank = all_food_list[rank]
        
#         for j in rank:
#             if j not in already_have or flag:
#                 y_pred.append(j)
#             if len(y_pred) >= 20:
#                 break
#         if not flag:
#             assert not len(set(y_pred) & set(already_have))
#         s = apk(y_true, list(y_pred), 20)
#         score.append(s)
        
#     return np.mean(score)


# # submit

# In[24]:


def predict(M, uid):
    userid = uid
    already_have = eaten_dict[userid]
    y_pred = []
    query_u, query_f = user_query[userid]
    rank = M(query_u, query_f, True).detach().cpu().numpy()
    rank = np.argsort(rank)[::-1]
    rank = all_food_list[rank]
    for j in rank:
        if j not in already_have:
            y_pred.append(j)
        if len(y_pred) >= 20:
            break
    assert not len(set(y_pred) & set(already_have))
    return y_pred


# In[23]:


def write_csv():
    submit = pd.DataFrame()
    submit["userid"] = df.userid.unique()

    preds = []
    for uid in df.userid.unique():
        pred = predict(model, uid)
        pred = " ".join([str(j) for j in pred])
        preds.append(pred)
    submit["foodid"] = preds
    submit.to_csv("submit.csv", index=False)


# In[ ]:


train_loader = DataLoader(
        dataset=TensorDataset(
            torch.from_numpy(df.userid.values),
            torch.from_numpy(df.foodid.values),
            torch.from_numpy(df.label.values),
        ),
        batch_size=1024,
        shuffle=True,
        num_workers=8)


# In[ ]:

from models import NeuMF, GMF, MLP, MF
model = MF(
        user_size=int(df.userid.max()+1),
        food_size=int(df.foodid.max()+1),
        hidden=200
        ).cuda()

best = 0.1
opt = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
N = neg_sampler2()




# In[26]:
# warp
MAX_SAMPLE = 50
for epoch in range(50):
    print('Epoch:', epoch)
    model.train()
    total_loss = []
    regu = []
    regr = []
    for user, food, rate in tqdm(train_loader):
        user, food, rate = Variable(user.cuda()), Variable(food.cuda()), rate.cuda()
        pos_pred = model(user, food)
    
        ############
        loss = torch.zeros((user.size()[0])).cuda()
        for _ in range(MAX_SAMPLE):
            neg = N.get_neg_samples(user.detach().cpu().data.numpy())
            neg = Variable(torch.from_numpy(neg).cuda())
            neg_pred = model(user, neg, True)
            
            tmp = torch.where((neg_pred > pos_pred-1.0), 
                              (1.0-pos_pred+neg_pred),
                              torch.zeros((user.size()[0])).cuda())
            
            tmp = tmp * torch.log(torch.floor(  (torch.full((user.size()[0],), 5531).cuda()  )/(_+1)  ))
            tmp = torch.clamp(tmp, max=10.)
            #mask = (loss == 0).float()
            loss += tmp#*mask
            
            
        loss = torch.mean(loss)
        regr.append(loss.item())
            
        lamb = 0.0001
        reg1 = lamb * torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), 1)
        reg2 = lamb * torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), 2)
        regu_loss = reg1 + reg2
        regu.append(regu_loss.item())
        
        loss += regu_loss
        total_loss.append(loss.item())
        
        opt.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm(model.parameters(), 1.)
        opt.step()
        
    print('train_loss:{:.5f}, regu:{:.5f}, regr:{:.5f}'
          .format(np.mean(total_loss), np.mean(regu), np.mean(regr)))
    
    if best > best:
        write_csv()
        best = ll
        torch.save({
                'state_dict': model.state_dict(),
                'train_csv':df,
                'global_mean': model.global_mean},
            'pytorch_model.pt')


# In[28]:

# a = torch.load('pytorch_model.pt')
# model_saved = MF(
#         user_size=int(df.userid.max()+1),
#         food_size=int(df.foodid.max()+1),
#         hidden=200
#         ).cuda()
# model_saved.load_state_dict(torch.load('pytorch_model.pt')['state_dict'])



# In[ ]:

# bpr
# for epoch in range(5):
#     print('Epoch:', epoch)
#     total_loss = []
#     regu = []
#     regr = []
#     for user, food, rate in tqdm(train_loader):
#         user, food, rate = user.cuda(), food.cuda(), rate.cuda().float()
#         pos_pred = model(user, food)
    
#         ############
#         neg = N.get_neg_samples(user.detach().cpu().numpy())
#         neg = torch.from_numpy(neg).cuda()
#         neg_pred = model(user, neg, True)
#         target = torch.ones(user.size()).cuda()
#         #loss = F.margin_ranking_loss(pos_pred, neg_pred, target, margin=1.)
#         #loss = torch.mean(1-pos_pred-neg_pred)
#         #loss = torch.mean(1-F.sigmoid(pos_pred-neg_pred))
#         loss = torch.mean(torch.log(1+torch.exp(-(pos_pred-neg_pred))))
#         regr.append(loss.item())
            
#         lamb = 0.0001
#         reg1 = lamb * torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), 1)
#         reg2 = lamb * torch.norm(torch.cat([x.view(-1) for x in model.parameters()]), 2)
#         regu_loss = reg1 + reg2
#         regu.append(regu_loss.item())
        
#         loss += regu_loss
#         total_loss.append(loss.item())
        
#         opt.zero_grad()
#         loss.backward()
#         nn.utils.clip_grad_norm(model.parameters(), 1.)
#         opt.step()
        
#     print('train_loss:{:.5f}, regu:{:.5f}, regr:{:.5f}'
#           .format(np.mean(total_loss), np.mean(regu), np.mean(regr)))





# In[25]:


write_csv()
# get_ipython().system("kaggle competitions submit -c ntucsie-sdml2018-2-1 -f submit.csv -m oh~~")
# time.sleep(5)
# get_ipython().system('kaggle competitions submissions ntucsie-sdml2018-2-1 | head -n 10')

