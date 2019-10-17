#coding:utf-8
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var

# import sys
# import logging

# #logger
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)


class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        #self.index = index
        self.idx = idx
        self.word = []
        self.index = []
        #self.height = 1
        #self.size = 1
        #self.num_leaves = 1
        self.parent = None
        #self.label = None

################################# generate tree structure ##############################
def gen_nn_inputs(root_node, max_degree=None, only_leaves_have_vals=True, with_labels=False):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    _clear_indices(root_node)
    #x, leaf_labels = _get_leaf_vals(root_node)
    X_word, X_index = _get_leaf_vals(root_node)
    tree, internal_word, internal_index = _get_tree_traversal(root_node, len(X_word), max_degree)
    #assert all(v is not None for v in x)
    #if not only_leaves_have_vals:
    #    assert all(v is not None for v in internal_x)
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    if max_degree is not None:
        assert all(len(t) == max_degree + 1 for t in tree)
    '''if with_labels:
        labels = leaf_labels + internal_labels
        labels_exist = [l is not None for l in labels]
        labels = [l or 0 for l in labels]
        return (np.array(x, dtype='int32'),
                np.array(tree, dtype='int32'),
                np.array(labels, dtype=theano.config.floatX),
                np.array(labels_exist, dtype=theano.config.floatX))'''   
    ##### debug here #####
    '''ls = []
    for x in X_word:
        l = len(x)
        if not l in ls: ls.append(l)
    print ls'''        
    #print type(X_word)    
    return (np.array(X_word, dtype='float64'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'))
    #return (np.array(X_word),
    #        np.array(X_index),
    #        np.array(tree))        


def _clear_indices(root_node):
    root_node.idx = None
    [_clear_indices(child) for child in root_node.children if child]


def _get_leaf_vals(root_node):
    """Get leaf values in deep-to-shallow, left-to-right order."""
    all_leaves = []
    layer = [root_node]
    while layer:
        next_layer = []
        for node in layer:
            if not node.children:
                all_leaves.append(node)
            else:
                next_layer.extend([child for child in node.children[::-1] if child])
        layer = next_layer

    X_word = []
    X_index = []
    for idx, leaf in enumerate(reversed(all_leaves)):
        leaf.idx = idx
        X_word.append(leaf.word)
        X_index.append(leaf.index)
        #print idx, leaf
        #print leaf.word
    return X_word, X_index


def _get_tree_traversal(root_node, start_idx=0, max_degree=None):
    """Get computation order of leaves -> root."""
    if not root_node.children:
        return [], [], []
    layers = []
    layer = [root_node]
    while layer:
        layers.append(layer[:])
        next_layer = []
        [next_layer.extend([child for child in node.children if child])
         for node in layer]
        layer = next_layer

    tree = []
    internal_word = []
    internal_index = []
    idx = start_idx
    for layer in reversed(layers):  #reversed()反转函数，如输入"[1,2,3,4,5]"，返回"[5,4,3,2,1]"
        for node in layer:
            if node.idx is not None:
                # must be leaf
                assert all(child is None for child in node.children)
                continue

            child_idxs = [(child.idx if child else -1)
                          for child in node.children]  ## idx of child node
            if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_word.append(node.word if node.word is not None else -1)
            internal_index.append(node.index if node.index is not None else -1)
            idx += 1

    return tree, internal_word, internal_index

class RvNN(nn.Module):
    def __init__(self, device, word_dim, hidden_dim = 5, Nclass=4, degree=2, irregular_tree=True):
        super(RvNN, self).__init__()
        assert word_dim > 1 and hidden_dim > 1
        self.cudaFlag = device
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.degree = degree
        self.irregular_tree = irregular_tree
        self.Nclass = Nclass

        self.embedding = nn.Embedding(self.word_dim, self.hidden_dim)
        nn.init.normal(self.embedding.weight, mean=0, std=0.1)
        self.W_z = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.U_z = self.init_matrix([self.hidden_dim,self.hidden_dim])
        self.b_z = self.init_vector(self.hidden_dim)

        self.W_r = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.U_r = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.b_r = self.init_vector(self.hidden_dim)

        self.W_h = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.U_h = self.init_matrix([self.hidden_dim, self.hidden_dim])
        self.b_h = self.init_vector(self.hidden_dim)
        
        self.W_out = self.init_matrix([self.hidden_dim, self.Nclass])
        self.b_out = self.init_vector(self.Nclass)
        if(self.cudaFlag):
            self.W_z = self.W_z.cuda()
            self.U_z = self.U_z.cuda()
            self.b_z = self.b_z.cuda()

            self.W_r = self.W_r.cuda()
            self.U_r = self.U_r.cuda()
            self.b_r = self.b_r.cuda()

            self.W_h = self.W_h.cuda()
            self.U_h = self.U_h.cuda()
            self.b_h = self.b_h.cuda()
        
            self.W_out = self.W_out.cuda()
            self.b_out = self.b_out.cuda()	
        # self.ix = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.ih = nn.Linear(self.hidden_dim, self.hidden_dim)
         
        # self.fx = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.fh = nn.Linear(self.hidden_dim, self.hidden_dim)

        # self.ux = nn.Linear(self.hidden_dim, self.hidden_dim)
        # self.uh = nn.Linear(self.hidden_dim, self.hidden_dim)

        # self.out = nn.Linear(self.hidden_dim, self.Nclass)
        # nn.init.normal(self.ix.weight, mean=0, std=0.1)
        # nn.init.constant(self.ix.bias, 0)
        # nn.init.normal(self.ih.weight, mean=0, std=0.1)
        # nn.init.constant(self.ih.bias, 0)
        # nn.init.normal(self.fx.weight, mean=0, std=0.1)
        # nn.init.constant(self.fx.bias, 0)
        # nn.init.normal(self.fh.weight, mean=0, std=0.1)
        # nn.init.constant(self.fh.bias, 0)
        # nn.init.normal(self.ux.weight, mean=0, std=0.1)
        # nn.init.constant(self.ux.bias, 0)
        # nn.init.normal(self.uh.weight, mean=0, std=0.1)
        # nn.init.constant(self.uh.bias, 0)
        # nn.init.normal(self.out.weight, mean=0, std=0.1)
        # nn.init.constant(self.out.bias, 0)

         # self.criterion = criterion
    def init_matrix(self, shape):
        std = 0.1*torch.ones(shape)
        return Var(torch.normal(mean=0.0, std=std),requires_grad=True)
    
    def init_vector(self, shape):
        return Var(torch.zeros(shape), requires_grad=True)

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.ix, self.ih, self.ox, self.oh, self.ux, self.uh]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        if self.cudaFlag:
        	params = params.cuda()
        return params
    def hard_sigmoid(self, x):
        """
        Computes element-wise hard sigmoid of x.
        See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
        """
        x = (0.2 * x) + 0.5
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)
        return x

    def node_forward(self, parent_word, parent_index, child_h, child_exists):
        h_tilde = torch.sum(child_h, 0)

        parent_embedding = self.embedding(parent_index)
        # logger.info('the size of parent_embedding:%s, the size of parent_word: %s'%(parent_embedding.size(), parent_word.size()))
        parent_xe = torch.squeeze(torch.mm(parent_embedding.transpose(1, 0), torch.unsqueeze(parent_word.float(), 1)))

        z = self.hard_sigmoid(torch.squeeze(self.W_z.mm(torch.unsqueeze(parent_xe,1)) + self.U_z.mm(torch.unsqueeze(h_tilde,1))) + self.b_z)
        r = self.hard_sigmoid(torch.squeeze(self.W_r.mm(torch.unsqueeze(parent_xe,1)) + self.U_r.mm(torch.unsqueeze(h_tilde,1))) + self.b_r)
        c = F.tanh(torch.squeeze(self.W_h.mm(torch.unsqueeze(parent_xe,1)) + self.U_h.mm(torch.unsqueeze(h_tilde*r,1))) + self.b_h)
        h = z*h_tilde + (1 - z)*c
        
        return h

    def init_node_child(self, x_word, x_index, num_leaves):
        dummy = 0*torch.zeros(self.degree, self.hidden_dim)
        if self.cudaFlag:
            dummy = dummy.cuda()
        # leaf_h = []
        for i in range(num_leaves):
            if i == 0:
                leaf_h = torch.unsqueeze(self.node_forward(x_word[i], x_index[i], dummy, torch.sum(dummy, 1)), 0)
            else:
                leaf_h = torch.cat((leaf_h, torch.unsqueeze(self.node_forward(x_word[i], x_index[i], dummy, torch.sum(dummy, 1)), 0)), 0)
        # print('the shape of leaf_h:%s'%str(leaf_h.size()))
        # logger.info('leaf_h:%s'%str(leaf_h))
        # leaf_h = torch.Tensor(leaf_h)
        # print('the size of leaf_h:%s'%str(leaf_h.size()))
        if self.irregular_tree:
            init_node_h = torch.cat([leaf_h, leaf_h, leaf_h], 0)
        else:
            init_node_h = leaf_h
        # print('the size of init_node_h:%s'%str(init_node_h.size()))

        return leaf_h, init_node_h

    def recurrence(self, x_word, x_index, node_info, t, node_h, last_h, num_leaves):
        child_exists = (node_info[:-1] > -1).int()
        # print('node_info:%s'%str(node_info))
        offset = torch.ones_like(child_exists)*2*num_leaves*int(self.irregular_tree) - child_exists * t ### offset???
        # offset = child_exists * (-t)
        # print('offset:%s'%str(offset))
        index = torch.add(node_info[:-1], offset).long()
        # index = index[:,:-1]
        # index = torch.add(node_info, offset)
        # print('the index:%s'%(str(index)))
        # child_h = node_h[index] * torch.unsqueeze(child_exists, 1)
        child_h = node_h[index]*(child_exists.view(len(child_exists), 1).float())
        
        parent_h = self.node_forward(x_word, x_index, child_h, child_exists)
        # print(parent_h.size())
        node_h = torch.cat([node_h, parent_h.view(1, self.hidden_dim)], 0)
        return node_h[1:], parent_h

    def forward(self, x_word, x_index, tree):
        self.num_nodes = x_word.shape[0]
        num_parents = tree.shape[0]

        num_leaves = self.num_nodes - num_parents
        leaf_h, init_node_h = self.init_node_child(x_word, x_index, num_leaves)
        # print('the tree:%s'%(str(tree)))
        # print('num_leaves:%s'%str(num_leaves))
        # print('num_parents:%s'%str(num_parents))
        # print('the size of init_node_h:%s'%str(init_node_h.size()))
        dummy = torch.zeros(self.hidden_dim)
        for i in range(num_parents):
            init_node_h, dummy = self.recurrence(x_word[num_leaves+i], x_index[num_leaves+i], tree[i], i, init_node_h, dummy, num_leaves)
            if(i == 0):
                parent_h = torch.unsqueeze(dummy, 0)
            else:
                parent_h = torch.cat([parent_h, torch.unsqueeze(dummy, 0)], 0)
        # output = torch.squeeze((torch.cat([leaf_h, parent_h], 0)).mm(self.W_out)[-1])+self.b_out
        # return F.softmax(output)
        return torch.squeeze(parent_h[-1:])

class RvNN_Co_GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RvNN_Co_GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.linear(x)
        # x = F.sigmoid(x)
        # return F.softmax(x)
        return x