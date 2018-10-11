import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pickle
import numpy as np


def preprocess(graph):
    
    nodeCount = int(graph.max()) + 1
    out_degrees = np.zeros(nodeCount)
    in_degrees = np.zeros(nodeCount)
    for node_i, node_j in graph:
        out_degrees[node_i] += 1
        in_degrees[node_j] += 1
    # avoid divied zero
    out_degrees[out_degrees == 0] = 1
    in_degrees[in_degrees == 0] = 1
    
    
    PMI_dict = {}
    PMI = np.zeros((len(graph))).astype('float32')
    for idx, edge in enumerate(graph):
        fromId, toId = edge
        pmi = len(graph) / 5.0 / out_degrees[fromId] / in_degrees[toId]
        PMI[idx] = np.log(pmi)
        PMI_dict[(fromId, toId)] = np.log(pmi)
    PMI[PMI < 0] = 0
    
    head_node = graph[:, 0]
    tail_node = graph[:, 1]

    train_loader = DataLoader(
        dataset=TensorDataset(
            torch.from_numpy(head_node.astype('int64')),
            torch.from_numpy(tail_node.astype('int64')),
            torch.from_numpy(PMI.astype('float32')),
            torch.from_numpy(in_degrees[head_node].astype('float32')),
            torch.from_numpy(out_degrees[tail_node].astype('float32'))
        ),
        batch_size=1024,
        shuffle=True,
        num_workers=8)
    
    return train_loader, PMI_dict



class PRUNE_Model(nn.Module):
    def __init__(self):
        super(PRUNE_Model, self).__init__()
        nodeCount = 37501
        self.node_emb = nn.Embedding(nodeCount, 128) # emb size 128
        
        w_init = np.identity(64) + abs(np.random.randn(64, 64) / 1000.0)
        self.w_shared = Variable(torch.from_numpy(w_init)).float().cuda()
        
        self.rank = nn.Sequential(
            self.node_emb,
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        self.prox = nn.Sequential(
            self.node_emb,
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        torch.nn.init.xavier_normal_(self.node_emb.weight)
        for layer in self.rank:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
        for layer in self.prox:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
        
        
    def forward(self, head, tail, pmi, indeg, outdeg):
        
        head_r = self.rank(head)
        head_p = self.prox(head)
        
        tail_r = self.rank(tail)
        tail_p = self.prox(tail)
        
        w = nn.ReLU()(self.w_shared)
        zWz = (head_p * torch.matmul(tail_p, w)).sum(1)
        prox_loss = ((zWz - pmi)**2).mean()
        
        rank_loss = indeg * (-tail_r / indeg + head_r / outdeg).pow(2)
        rank_loss = rank_loss.mean()

        lamb = 0.01
        total_loss = prox_loss + lamb * rank_loss
        return total_loss


    
def train(model, epochs, lr, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        print('Epoch:', epoch)

        total_loss = []
        for e , (head, tail, pmi, indegr, outdegr) in enumerate(train_loader):
            head, tail, pmi, indegr, outdegr = head.cuda(), tail.cuda(), pmi.cuda(), indegr.cuda(), outdegr.cuda()
            loss = model(head, tail, pmi, indegr, outdegr)
            total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('train_loss:{:.4f}'.format(np.mean(total_loss)))
        torch.save({'state_dict': model.state_dict()}, 'prune.pt')

        
        
if __name__ == '__main__':
    prune_data_loader, PMI_dict = preprocess(graph)
    model = PRUNE_Model().cuda()
    train(model, prune_epoch, prune_lr)
    emb_weight = model.node_emb.weight.data.cpu().numpy()
    with open('prune_weight.pkl', 'wb') as f:
        pickle.dump(emb_weight, f)
    with open('pmi.pkl', 'wb') as f:
        pickle.dump(PMI_dict, f)
    