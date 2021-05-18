import torch
import torch.nn as nn
import torch.nn.functional as F
from loss_utils import get_positive_expectation, get_negative_expectation


def local_loss(node_features, global_prop, adj, assign, batch_num_nodes, measure, demask):
    '''
    Args:
        node_features: node-level features map
        measure:Loss measure. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    batch_size = node_features.shape[0]
    # num_nodes = node_features.shape[1]
    num_clusters = assign.shape[2]
    feature_dim = node_features.shape[2]

    e_pos = []
    e_neg = []

    if demask == True:
        node_features_final = []
        assign_final = []
        global_prop_final = []
        adj_final = []
        total_num_nodes = 0
        total_neg_num = 0
        total_pos_num = 0
        for m, nodes in enumerate(batch_num_nodes):
            node_features_final.append(node_features[m, :nodes, :])
            assign_final.append(assign[m, :nodes, :])
            global_prop_final.append(global_prop[m, :nodes, :])
            adj_final.append(adj[m, :nodes, :nodes])

            total_num_nodes = total_num_nodes + nodes
            neg_num = nodes * nodes

            pos_mask = torch.zeros((nodes, nodes)).cuda()
            neg_mask = torch.ones((nodes, nodes)).cuda()
            # neg_mask = adj_final[m].clone().cuda()

            for i in range(nodes):
                for j in range(num_clusters):
                    if assign_final[m][i][j] == 1:
                        for k in range(nodes):
                            if assign_final[m][k][j] == 1:
                                pos_mask[i][k] = 1.
                                neg_mask[i][k] = 0.
                                # pos_mask[i][k] = adj_final[m][i][k]
                                # neg_mask[i][k] = 0.
                                total_pos_num += 1
                                neg_num -= 1
                        break
            total_neg_num += neg_num

            r1 = node_features_final[m].repeat(1, num_clusters)
            r2 = global_prop_final[m].repeat(1, feature_dim)
            res_m = torch.matmul(r1, r2.t())
            # res_m = torch.matmul(node_features_final[m], node_features_final[m].t())
            e_pos_m = get_positive_expectation(res_m * pos_mask, measure, average=False).sum()
            e_pos.append(e_pos_m)
            e_neg_m = get_negative_expectation(res_m * neg_mask, measure, average=False).sum()
            e_neg.append(e_neg_m)

    if demask == False:
        node_features_final = node_features
        assign_final = assign
        global_prop_final = global_prop
        adj_final = adj

        num_nodes = node_features.shape[1]

        total_pos_num = 0
        total_neg_num = batch_size * (num_nodes * num_nodes)

        for m in range(batch_size):
            pos_mask = torch.zeros((num_nodes, num_nodes)).cuda()
            neg_mask = torch.ones((num_nodes, num_nodes)).cuda()
            # neg_mask = adj_final[m].clone().cuda()

            for i in range(num_nodes):
                for j in range(num_clusters):
                    if assign_final[m][i][j] == 1:
                        for k in range(num_nodes):
                            if assign_final[m][k][j] == 1:
                                pos_mask[i][k] = 1.
                                neg_mask[i][k] = 0.
                                # pos_mask[i][k] = adj_final[m][i][k]
                                # neg_mask[i][k] = 0.
                                total_pos_num += 1
                                total_neg_num -= 1
                        break

        r1 = node_features_final[m].repeat(1, num_clusters)
        r2 = global_prop_final[m].repeat(1, feature_dim)
        res_m = torch.matmul(r1, r2.t())
        # res_m = torch.matmul(node_features_final[m], node_features_final[m].t())
        e_pos_m = get_positive_expectation(res_m * pos_mask, measure, average=False).sum()
        e_pos.append(e_pos_m)
        e_neg_m = get_negative_expectation(res_m * neg_mask, measure, average=False).sum()
        e_neg.append(e_neg_m)

    e_pos_all = sum(e_pos)/total_pos_num
    e_neg_all = sum(e_neg)/(total_neg_num + (1e-10))

    return e_neg_all - e_pos_all


def global_loss(node_features, cluster_features, assign, batch_num_nodes, measure, demask):
    '''
    Args:
        node_features: node-level features map
        cluster_features: cluster-level features map
        measure:Loss measure. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    batch_size = node_features.shape[0]
    num_clusters = cluster_features.shape[1]

    total_pos_num = 0
    total_neg_num = 0
    e_pos = []
    e_neg = []

    if demask == True:
        node_features_final = []
        assign_final = []
        total_num_nodes = 0
        for m, nodes in enumerate(batch_num_nodes):
            node_features_final.append(node_features[m, :nodes, :])
            assign_final.append(assign[m, :nodes, :])
            total_num_nodes = total_num_nodes + nodes

            pos_mask = assign_final[m].cuda()
            neg_mask = (torch.ones_like(assign_final[m]) - assign_final[m]).cuda()

            res_m = torch.matmul(node_features_final[m], cluster_features[m].t())

            e_pos_m = get_positive_expectation(res_m * pos_mask, measure, average=False).sum()
            e_pos.append(e_pos_m)
            e_neg_m = get_negative_expectation(res_m * neg_mask, measure, average=False).sum()
            e_neg.append(e_neg_m)
        total_pos_num = total_num_nodes
        total_neg_num = total_num_nodes * num_clusters - total_pos_num
    if demask == False:
        node_features_final = node_features
        assign_final = assign
        num_nodes = node_features.shape[1]

        for m in range(batch_size):
            pos_mask = assign_final[m].cuda()
            neg_mask = (torch.ones_like(assign_final[m]) - assign_final[m]).cuda()

            res_m = torch.matmul(node_features_final[m], cluster_features[m].t())

            e_pos_m = get_positive_expectation(res_m * pos_mask, measure, average=False).sum()
            e_pos.append(e_pos_m)
            e_neg_m = get_negative_expectation(res_m * neg_mask, measure, average=False).sum()
            e_neg.append(e_neg_m)

        total_pos_num = batch_size * num_nodes
        total_neg_num = batch_size * num_nodes * num_clusters - total_pos_num

    e_pos_all = sum(e_pos) / total_pos_num
    e_neg_all = sum(e_neg) / (total_neg_num + (1e-10))

    return e_neg_all - e_pos_all

