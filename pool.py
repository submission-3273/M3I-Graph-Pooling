import torch
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from torch.autograd import Variable

class Pool(nn.Module):

    def __init__(self, input_dim, coarsen_node_num, act, dropout=0.0, bias=True):
        super(Pool, self).__init__()

        self.input_dim = input_dim
        self.coarsen_node_num = coarsen_node_num
        self.act = act
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, coarsen_node_num).cuda())
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        self.a = nn.Parameter(torch.FloatTensor(2*coarsen_node_num, 1).cuda())
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(coarsen_node_num).cuda())
            nn.init.constant_(self.bias.data, 0.0)
        else:
            self.bias = None

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        z = torch.matmul(x, self.weight)
        batch_size = z.size()[0]
        ori_node_num = z.size()[1]

        c1 = z.repeat(1, 1, self.coarsen_node_num).view(batch_size, ori_node_num * self.coarsen_node_num, -1)
        c2 = z.repeat(1, self.coarsen_node_num, 1)
        # concat = torch.cat([c1, c2], 1).view(batch_size, ori_node_num, -1, 2 * self.coarsen_node_num)
        concat = torch.cat([c1, c2], 2)

        a_input = torch.matmul(concat.view(batch_size, ori_node_num*self.coarsen_node_num, 2*self.coarsen_node_num), self.a).squeeze(2)

        e = self.act(a_input.view(batch_size, ori_node_num, self.coarsen_node_num))

        # zero_vec = -9e15 * torch.ones_like(e)
        assign = F.softmax(e, dim=2)

        if self.dropout > 0.001:
            assign = self.dropout_layer(assign)

        if self.bias is not None:
            assign = assign + self.bias

        assign_hard = gumbel_softmax(assign)

        new_x = torch.matmul(torch.transpose(self.act(assign_hard), 1, 2), x)
        new_adj = torch.transpose(self.act(assign_hard), 1, 2) @ adj @ assign_hard

        return new_x, new_adj, assign_hard, z


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=0.01):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.01, hard=True):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard
