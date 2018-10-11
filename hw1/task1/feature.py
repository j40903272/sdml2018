import pickle
import networkx as nx
with open('prune_weight.pkl', 'rb') as f:
    embedding = pickle.load(f)
with open('PMI_dict.pkl', 'rb') as f:
    PMI_dict = pickle.load(f)

def extract_node_feature(G, unG, node):
    '''
    0. embedding (128)
    1. degree
    2. indegree
    3. outdegree
    '''
    feat = list(embedding[node])
    feat.append(G.degree(node))
    feat.append(G.in_degree(node))
    feat.append(G.out_degree(node))
    '''
    feat.append(pagerank[node])
    feat.append(hit[node])
    feat.append(auth[node])
    feat.append(nx.clustering(unG, [node]))
    feat.append(nx.average_clustering(unG, [node]))
    feat.append(nx.square_clustering(unG, [node]))
    
    feat.append(deg_cent[node])
    feat.append(in_cent[node])
    feat.append(out_cent[node])
    feat.append(b_cent[node])
    feat.append(l_cent[node])
    
    feat.append(nx.node_clique_number(unG, [node]))
    feat.append(nx.triangles(unG, [node]))
    '''
    return feat



def extract_edge_feature(G, unG, head, tail, node_feat):
    '''
    featrue
    1. head node feature
    2. tail node featrue
    3. pmi
    4. num common successors
    5. num common predecessors
    6. num pred(head) & succ(tail)
    7. num common neighbor
    8. jaccard
    9. resource
    10. adamic
    11. has path
    '''
    head_feat = node_feat[head] if head in node_feat else [0]*131
    tail_feat = node_feat[tail] if tail in node_feat else [0]*131
    all_feat = head_feat + tail_feat
    if head not in G or tail not in G:
        return all_feat + [0]*9
    
    all_feat.append(0 if (head, tail not in PMI_dict) else PMI_dict[(head, tail)]) #263
    all_feat.append(len(set(G.successors(head)) & set(G.successors(tail))))
    all_feat.append(len(set(G.predecessors(head)) & set(G.predecessors(tail))))
    all_feat.append(len(set(G.predecessors(head)) & set(G.successors(tail))))
    all_feat.append(len(set(nx.common_neighbors(unG, head, tail))))
    all_feat.append(list(nx.jaccard_coefficient(unG, [(head, tail)]))[0][2])
    all_feat.append(list(nx.resource_allocation_index(unG, [(head, tail)]))[0][2])
    all_feat.append(list(nx.adamic_adar_index(unG, [(head, tail)]))[0][2])
    all_feat.append(list(nx.preferential_attachment(unG, [(head, tail)]))[0][2]) #271
    
    
    
    #all_feat.append(eb_cent[(head, tail)])
    #all_feat.append(nx.has_path(G, head, tail))
        
    return all_feat