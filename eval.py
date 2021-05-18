from torch.autograd import Variable
from logreg import LogReg
import torch
import torch.nn as nn
from util import cmd_args, PrepareFeatureLabel

def find_epoch(out_dim, nb_classes, train_embs, train_labels, val_embs, val_labels, concat=True):
    if concat:
        graph_dim = out_dim * cmd_args.num_pool
    else:
        graph_dim = out_dim
    log = LogReg(graph_dim, nb_classes)
    opt = torch.optim.Adam(log.parameters(), lr=0.0001, weight_decay=0.0001)
    xent = nn.CrossEntropyLoss()
    log.cuda()

    epoch_flag = 0
    epoch_win = 0
    test_acc = torch.zeros(1).cuda()

    for e in range(20000):
        log.train()
        opt.zero_grad()

        logits = log(train_embs)
        loss = xent(logits, train_labels)

        loss.backward(retain_graph=True)
        opt.step()

        if (e + 1) % 100 == 0:
            log.eval()
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(preds == val_labels).float() / val_labels.shape[0]
            if acc >= test_acc:
                epoch_flag = e + 1
                test_acc = acc
                epoch_win = 0
            else:
                epoch_win += 1
            if epoch_win == 20:
                break
    print('test accuracy:', test_acc)

    return epoch_flag, test_acc


def evaluate(model, train_graphs, val_graphs, max_nodes, num_classes, mask_nodes=True):
    train_h0, train_adj, train_labels, train_batch_num_nodes = PrepareFeatureLabel(max_nodes, train_graphs)
    train_embs = model.embed(train_h0, train_adj, train_batch_num_nodes)

    val_h0, val_adj, val_labels, val_batch_num_nodes = PrepareFeatureLabel(max_nodes, val_graphs)
    val_embs = model.embed(val_h0, val_adj, val_batch_num_nodes)

    iter_num, test_acc = find_epoch(model.assign_input_dim, num_classes, train_embs, train_labels, val_embs, val_labels, concat=cmd_args.concat)

    return test_acc




