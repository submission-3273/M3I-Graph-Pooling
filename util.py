from __future__ import print_function
import numpy as np
import networkx as nx
import argparse
import torch
from logreg import LogReg
import torch.nn as nn
import random
from graph_sampler import GraphSampler


cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-cuda', type=str, default='0')
cmd_opt.add_argument('-dataset', default='MUTAG', help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-input_dim', type=int, default=10, help='input dimension for constant node feature')
cmd_opt.add_argument('-num_epochs', type=int, default=100, help='number of epochs')
cmd_opt.add_argument('-num_class', type=int, default=2, help='#classes')
cmd_opt.add_argument('-num_pool', type=int, default=1, help='hierarchical pooling')
cmd_opt.add_argument('-num_gc_layers', type=int, default=2, help='number of GCN layers between pooling')
cmd_opt.add_argument('-assign_ratio', type=float, default=0.5, help='the coarsen ratio for each pooling layer')
cmd_opt.add_argument('-epoch_flag', type=int, default=20, help='early stopping (default:20)')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-latent_dim', type=int, default=64, help='dimension(s) of latent layers')
cmd_opt.add_argument('-out_dim', type=int, default=32, help='s2v output size')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-bn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
cmd_opt.add_argument('-dropout', type=float, default=0.0, help='add dropout after dense layer')
cmd_opt.add_argument('-concat', type=bool, default=True, help='concat or not for patch embeddings and graph embeddings')
cmd_opt.add_argument('-feature_type', type=str, default='default', help='default/id/deg-num/deg/constant/struct')
cmd_opt.add_argument('-feat', type=str, default='node-label', help='useing node features or node labels for default')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0, help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')

cmd_args, _ = cmd_opt.parse_known_args()


class GNNGraph(object):
    def __init__(self, g, label, adj, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.adj = adj
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(edge_features.values()[0]) == np.ndarray)
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)


def load_data():

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('data/%s/%s.txt' % (cmd_args.dataset, cmd_args.dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if attr is not None:
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            #assert len(g.edges()) * 2 == n_edges  (some graphs in COLLAB have self-loops, ignored here)
            assert len(g) == n
            adj = np.array(nx.adjacency_matrix(g).todense())
            g_list.append(GNNGraph(g, l, adj, node_tags, node_features))
    for g in g_list:
        g.label = label_dict[g.label]
    cmd_args.num_class = len(label_dict)
    cmd_args.feat_dim = len(feat_dict) # maximum node label (tag)
    cmd_args.edge_feat_dim = 0
    if node_feature_flag == True:
        cmd_args.attr_dim = node_features.shape[1] # dim of node features (attributes)
    else:
        cmd_args.attr_dim = 0

    max_nodes = max([G.num_nodes for G in g_list])
    min_nodes = min([G.num_nodes for G in g_list])
    mean_nodes = int(np.mean([G.num_nodes for G in g_list]))

    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)
    print('#max nodes: %d' % max_nodes)
    print('#min nodes: %d' % min_nodes)
    print('#mean nodes: %d' % mean_nodes)

    if cmd_args.test_number == 0:
        train_idxes = np.loadtxt('data/%s/10fold_idx/train_idx-%d.txt' % (cmd_args.dataset, cmd_args.fold), dtype=np.int32).tolist()
        test_idxes = np.loadtxt('data/%s/10fold_idx/test_idx-%d.txt' % (cmd_args.dataset, cmd_args.fold), dtype=np.int32).tolist()
        return [g_list[i] for i in train_idxes], [g_list[i] for i in test_idxes], max_nodes, mean_nodes
    else:
        return g_list[: n_g - cmd_args.test_number], g_list[n_g - cmd_args.test_number :], max_nodes, mean_nodes




def prepare_graphs(graphs, val_idx):
    random.shuffle(graphs)
    val_size = len(graphs) // 10
    train_graphs = graphs[:val_idx * val_size]
    if val_idx < 9:
        train_graphs = train_graphs + graphs[(val_idx + 1) * val_size:]
    val_graphs = graphs[val_idx * val_size: (val_idx + 1) * val_size]
    print('Num training graphs: ', len(train_graphs),
          '; Num validation graphs: ', len(val_graphs))

    print('Number of graphs: ', len(graphs))
    print('Number of edges: ', sum([G.number_of_edges() for G in graphs]))
    print('Max, avg, std of graph size: ',
          max([G.number_of_nodes() for G in graphs]), ', '
                                                      "{0:.2f}".format(np.mean([G.number_of_nodes() for G in graphs])),
          ', '
          "{0:.2f}".format(np.std([G.number_of_nodes() for G in graphs])))

    max_nodes = max([G.number_of_nodes() for G in graphs])
    mean_nodes = int(np.mean([G.number_of_nodes() for G in graphs]))

    return train_graphs, val_graphs, max_nodes, mean_nodes


def prepare_data(train_graphs, val_graphs, train_batch_size, max_nodes=0):

    # minibatch
    dataset_sampler = GraphSampler(train_graphs, normalize=False, max_num_nodes=max_nodes,
            features=cmd_args.feature_type)
    train_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=1)

    dataset_sampler = GraphSampler(val_graphs, normalize=False, max_num_nodes=max_nodes,
            features=cmd_args.feature_type)
    val_dataset_loader = torch.utils.data.DataLoader(
            dataset_sampler,
            batch_size=len(val_graphs),
            shuffle=False,
            num_workers=1)

    return train_dataset_loader, val_dataset_loader, \
            dataset_sampler.max_num_nodes, dataset_sampler.feat_dim


def PrepareFeatureLabel(max_nodes, batch_graph):
    labels = torch.LongTensor(len(batch_graph))

    batch_num_nodes = []
    concat_adjs = []
    concat_feats = []

    if batch_graph[0].node_tags is not None:
        node_tag_flag = True
        # concat_tag = []
    else:
        node_tag_flag = False

    if batch_graph[0].node_features is not None:
        node_feat_flag = True
        # concat_feat = []
    else:
        node_feat_flag = False

    for i in range(len(batch_graph)):
        labels[i] = batch_graph[i].label
        nodes = batch_graph[i].num_nodes
        batch_num_nodes.append(batch_graph[i].num_nodes)

        tag = batch_graph[i].node_tags
        tag = torch.LongTensor(tag).view(-1, 1)
        node_tag = torch.zeros(nodes, cmd_args.feat_dim)
        node_tag.scatter_(1, tag, 1)
        node_tag = np.array(node_tag)

        node_feat = batch_graph[i].node_features

        f = np.zeros((max_nodes, cmd_args.feat_dim), dtype=float)

        if node_tag_flag == True and node_feat_flag == False:
            # concat_tag += batch_graph[i].node_tags
            # concat_tag = batch_graph[i].node_tags
            for j in range(nodes):
                f[j, :] = node_tag[j]
            concat_feats.append(f)
        if node_feat_flag == True and node_tag_flag == False:
            # tmp = torch.from_numpy(batch_graph[i].node_features).type('torch.FloatTensor')
            # concat_feat.append(tmp)
            for j in range(nodes):
                f[j, :] = node_feat[j]
            concat_feats.append(f)
        if node_feat_flag == True and node_tag_flag == True:
            node_feat = torch.cat([node_tag.type_as(node_feat), node_feat], 1)
            for j in range(nodes):
                f[j, :] = node_feat[j]
            concat_feats.append(f)
        if node_feat_flag == False and node_tag_flag == False:
            node_feat = torch.ones(nodes, 1)  # use all-one vector as node features
            for j in range(nodes):
                f[j, :] = node_feat[j]
            concat_feats.append(f)

        adj = batch_graph[i].adj
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((max_nodes, max_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj
        concat_adjs.append(adj_padded)



    batch_num_nodes = np.array(batch_num_nodes)
    concat_adjs = torch.Tensor(concat_adjs).cuda()
    # node_feat = node_feat.cuda()
    concat_feats = torch.Tensor(concat_feats).cuda()
    labels = labels.cuda()

    return concat_feats, concat_adjs, labels, batch_num_nodes


def find_epoch(out_dim, nb_classes, train_embs, train_labels, test_embs, test_labels):
    log = LogReg(out_dim, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.001, weight_decay=0.00001)
    xent = nn.CrossEntropyLoss()
    log.cuda()

    epoch_flag = 0
    epoch_win = 0
    best_acc = torch.zeros(1).cuda()

    for e in range(20000):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_labels)

        loss.backward()
        opt.step()

        if (e + 1) % 100 == 0:
            log.eval()
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == test_labels).float() / test_labels.shape[0]
            if acc >= best_acc:
                epoch_flag = e + 1
                best_acc = acc
                epoch_win = 0
            else:
                epoch_win += 1
            if epoch_win == 10:
                break
    return epoch_flag


# ---- NetworkX compatibility
def node_iter(G):
    if float(nx.__version__)<2.0:
        return G.nodes()
    else:
        return G.nodes

def node_dict(G):
    if float(nx.__version__)>2.1:
        node_dict = G.nodes
    else:
        node_dict = G.node
    return node_dict
# ---------------------------


def get_root_path():
    from os.path import dirname, abspath
    return dirname(dirname(abspath(__file__)))


def get_data_path():
    return get_root_path() + '/MIP/data'


def get_save_path():
    return get_root_path() + '/MIP/save'
