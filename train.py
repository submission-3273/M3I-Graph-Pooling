import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from util import cmd_args, get_save_path, PrepareFeatureLabel
from eval import evaluate
import random
from tqdm import tqdm
import numpy as np


def train(dataset, model, train_graphs, val_graphs, max_nodes, val_dataset=None, writer=None,
          mask_nodes=True):

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_train_loss = 1e9
    best_train_epoch = 0
    cnt_wait = 0
    save_dir = '{}/best_mip.pkl'.format(get_save_path())
    best_acc = torch.zeros(1).cuda()

    accs = []

    for epoch in range(cmd_args.num_epochs):
        epoch_time = 0
        loss_epoch = 0
        avg_loss = 0.0
        model.train()
        print('Epoch: ', epoch)
        random.shuffle(train_idxes)
        total_iters = (len(train_idxes) + (cmd_args.batch_size - 1) * (optimizer is None)) // cmd_args.batch_size
        pbar = tqdm(range(total_iters), unit='batch')
        all_targets = []

        for pos in pbar:
            selected_idx = train_idxes[pos * cmd_args.batch_size: (pos + 1) * cmd_args.batch_size]

            batch_graph = [train_graphs[idx] for idx in selected_idx]
            targets = [train_graphs[idx].label for idx in selected_idx]
            all_targets += targets

            node_feats, adjs, labels, batch_num_nodes = PrepareFeatureLabel(max_nodes, batch_graph)

            begin_time = time.time()
            model.zero_grad()

            graph_avg_loss = model(node_feats, adjs, batch_num_nodes)
            loss_epoch += graph_avg_loss
            graph_avg_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            elapsed = time.time() - begin_time
            epoch_time += elapsed

        avg_graph_loss = loss_epoch/len(dataset)
        print('Epoch average loss: ', avg_graph_loss, '; epoch time: ', epoch_time)

        test_acc = evaluate(model, train_graphs, val_graphs, max_nodes, cmd_args.num_class, mask_nodes=True)
        if test_acc >= best_acc:
            best_acc = test_acc
        print('best accuracy:', best_acc)

        if avg_graph_loss < best_train_loss:
            best_train_loss = avg_graph_loss
            cnt_wait = 0

            torch.save(model.state_dict(), save_dir)
        else:
            cnt_wait += 1

        if cnt_wait == cmd_args.epoch_flag:
            print('Early stopping!')
            break
        accs.append(best_acc)

    with open('acc_result.txt', 'a+') as f:
        # f.write(str(test_loss[1]) + '\n')
        f.write(str(best_acc) + '\n')
