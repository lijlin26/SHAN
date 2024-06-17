
import numpy as np 

import torch
import random
import logging

from itertools import combinations
import logging

import dgl

import networkx as nx
import numpy as np
from itertools import combinations






def set_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def  preprocess_graph(graph, target_ntype_range, threshold, K, sample_times):
    other_nodes = list(range(graph.num_nodes()))
    other_nodes = list(set(other_nodes) ^ set(target_ntype_range))
    graph_homos = []
    num_simplex = {}
    degree_total = []
    for k in range(1, K+1):
        num_simplex[str(k)]=0
    for k in range(1, K+1):
        if k == 0:
            u = target_ntype_range
            v = target_ntype_range
        else:
            u = []
            v = []

            target_nodes_candidates = graph.out_edges(other_nodes)[1]
            target_nodes_candidates = target_nodes_candidates.numpy()
            

            for i in range(sample_times):
                temp = np.random.permutation(target_nodes_candidates)
                target_nodes_candidates = np.concatenate((target_nodes_candidates, temp))

            if target_nodes_candidates.shape[0] % (k+1) != 0 :
                pad = target_nodes_candidates.shape[0] % (k+1)
                target_nodes_candidates = target_nodes_candidates[:-pad]
                target_nodes_candidates = target_nodes_candidates.reshape(-1, (k+1))
            else:
                target_nodes_candidates = target_nodes_candidates.reshape(-1, (k+1))

            for candidate in target_nodes_candidates:
                neigh_overlap = graph.out_edges(candidate)[1]
                neigh_overlap = np.unique(neigh_overlap, return_counts=True)[1]
                neigh_overlap = (neigh_overlap >= (k+1)).sum()
                if neigh_overlap >= int(threshold[k-1]):
                    num_simplex[str(k)] += 1
                    for node_pair in combinations(candidate, 2):
                        u.append(node_pair[0].item())
                        v.append(node_pair[1].item())
        
        u = torch.tensor(u)
        v = torch.tensor(v)
        
        
        graph_homo = dgl.graph((u, v), num_nodes=graph.num_nodes())

        graph_homo = dgl.to_bidirected(graph_homo)

        graph_homo = dgl.remove_self_loop(graph_homo)
        graph_homo = dgl.add_self_loop(graph_homo)
        
        degrees = graph_homo.in_degrees().tolist()

        degree_total += degrees


        graph_homos.append(graph_homo)
        




    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)



    logging.info(num_simplex)
    for g in graph_homos:
        logging.info(g)
    logging.info(graph)


    return graph, graph_homos