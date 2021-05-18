import os

import numpy as np
import torch
import random
from train import train
from encoder import MipGcnEncoder
from util import cmd_args, load_data

def task(writer=None):
    train_graphs, test_graphs, max_nodes, mean_nodes = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

    model = MipGcnEncoder(mean_nodes, cmd_args.feat_dim, cmd_args.latent_dim, cmd_args.out_dim, cmd_args.num_gc_layers, assign_ratio=cmd_args.assign_ratio, num_pooling=cmd_args.num_pool, bn=cmd_args.bn, concat=cmd_args.concat, dropout=cmd_args.dropout).cuda()

    train(train_graphs, model, train_graphs, test_graphs, max_nodes, val_dataset=test_graphs, writer=writer)


if __name__ == '__main__':
    print(cmd_args)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    writer = None
    os.environ['CUDA_VISIBLE_DEVICES'] = cmd_args.cuda
    print('CUDA', cmd_args.cuda)
    task(writer=writer)
