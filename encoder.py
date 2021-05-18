import torch
import torch.nn as nn
import torch.nn.functional as F
from pool import Pool
from loss import local_loss, global_loss


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding

        # self.act = nn.PReLU() if act =='prelu' else act
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        # nn.init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu'))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
            # nn.init.constant_(self.bias.data, 0.0)
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            # print(y[0][0])
        return y

class GcnEncoderGraph(nn.Module):
    def __init__(self, num_layers, bn=True, concat=True, bias=True):
        super(GcnEncoderGraph, self).__init__()
        self.num_layers = num_layers
        self.bn = bn
        self.concat = concat
        self.add_self = not concat
        self.bias = bias
        self.act = nn.ReLU()

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=self.add_self,  normalize_embedding=normalize, dropout=dropout, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=self.add_self, normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(self.num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim,add_self=self.add_self, normalize_embedding=normalize, dropout=dropout,bias=self.bias)
        return conv_first, conv_block, conv_last

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

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]

        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        if self.concat:
            x_tensor = torch.cat(x_all, dim=2)
        else:
            x_tensor = x_all[-1]
        if embedding_mask is not None:  #mask node feature
            x_tensor = x_tensor * embedding_mask
        return x_tensor


class MipGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, num_layers,
                 assign_ratio=0.5, num_pooling=2, bn=True, concat=True, dropout=0.0):

        super(MipGcnEncoder, self).__init__(num_layers, concat=concat, bn=bn)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = embedding_dim
        self.num_pooling = num_pooling

        if concat:
            self.assign_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.assign_input_dim = embedding_dim

        # GC
        self.conv_first_before_pool = nn.ModuleList()
        self.conv_block_before_pool = nn.ModuleList()
        self.conv_last_before_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first, conv_block, conv_last = self.build_conv_layers(input_dim, hidden_dim, embedding_dim, normalize=bn, dropout=dropout)
            self.conv_first_before_pool.append(conv_first)
            self.conv_block_before_pool.append(conv_block)
            self.conv_last_before_pool.append(conv_last)
            #prepare for next GCN block after pooling
            input_dim = self.assign_input_dim

        # assignment
        assign_dims = []
        self.assign_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_module = Pool(self.assign_input_dim, assign_dim, act=nn.ReLU())
            self.assign_modules.append(assign_module)

            # next pooling layer
            assign_dim = int(assign_dim * assign_ratio)
            if concat:
                self.assign_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
            else:
                self.assign_input_dim = embedding_dim

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if batch_num_nodes is not None:
            batch_size = adj.size()[0]
            # mask
            max_num_nodes = adj.size()[1]
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            batch_size = 1
            embedding_mask = None

        batch_loss = 0
        for i in range(self.num_pooling):
            if i == 0:
                embedding_tensor = self.gcn_forward(x, adj, self.conv_first_before_pool[i], self.conv_block_before_pool[i], self.conv_last_before_pool[i], embedding_mask)
                x_new, adj_new, assign_hard, global_prop = self.assign_modules[i].forward(embedding_tensor, adj)
                global_l = global_loss(embedding_tensor, x_new, assign_hard, batch_num_nodes, measure='JSD', demask=True)
                local_l = local_loss(embedding_tensor, global_prop, adj, assign_hard, batch_num_nodes, measure='JSD', demask=True)

            else:
                embedding_tensor = self.gcn_forward(x, adj, self.conv_first_before_pool[i], self.conv_block_before_pool[i], self.conv_last_before_pool[i], embedding_mask=None)
                x_new, adj_new, assign_hard, global_prop = self.assign_modules[i].forward(embedding_tensor, adj)
                global_l = global_loss(embedding_tensor, x_new, assign_hard, batch_num_nodes, measure='JSD', demask=False)
                local_l = local_loss(embedding_tensor, global_prop, adj, assign_hard, batch_num_nodes, measure='JSD', demask=False)

            batch_loss = batch_loss+local_l+global_l
            # batch_loss += local_l

            graph_avg_loss = batch_loss/batch_size

            x = x_new
            adj = adj_new
        return graph_avg_loss

    def embed(self, x, adj, batch_num_nodes):
        if batch_num_nodes is not None:
            batch_size = adj.size()[0]
            # mask
            max_num_nodes = adj.size()[1]
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            batch_size = 1
            embedding_mask = None

        graph_level_all = []

        for i in range(self.num_pooling):
            if i == 0:
                embedding_tensor = self.gcn_forward(x, adj, self.conv_first_before_pool[i], self.conv_block_before_pool[i], self.conv_last_before_pool[i], embedding_mask)
            else:
                embedding_tensor = self.gcn_forward(x, adj, self.conv_first_before_pool[i], self.conv_block_before_pool[i], self.conv_last_before_pool[i], embedding_mask=None)

            x_new, adj_new, assign_hard, _ = self.assign_modules[i].forward(embedding_tensor, adj)
            out = torch.sum(x_new, dim=1)
            graph_level_all.append(out)
            x = x_new
            adj = adj_new

        if self.concat:
            graph_level_embd = torch.cat(graph_level_all, dim=1)
        else:
            graph_level_embd = graph_level_all[-1]
        return graph_level_embd
