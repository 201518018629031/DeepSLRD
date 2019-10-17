#coding:utf-8
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var


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
#def gen_nn_inputs(root_node, ini_word, ini_index):
def gen_nn_inputs(root_node, ini_word):
    """Given a root node, returns the appropriate inputs to NN.

    The NN takes in
        x: the values at the leaves (e.g. word indices)
        tree: a (n x degree) matrix that provides the computation order.
            Namely, a row tree[i] = [a, b, c] in tree signifies that a
            and b are children of c, and that the computation
            f(a, b) -> c should happen on step i.

    """
    #_clear_indices(root_node)
    #x, leaf_labels = _get_leaf_vals(root_node)
    #X_word, X_index = _get_leaf_vals(root_node)
    tree = [[0, root_node.idx]] 
    #X_word, X_index = [ini_word], [ini_index]
    X_word, X_index = [root_node.word], [root_node.index]
    #print X_index
    #print X_word
    #exit(0)
    internal_tree, internal_word, internal_index  = _get_tree_path(root_node)
    #print internal_tree
    #assert all(v is not None for v in x)
    #if not only_leaves_have_vals:
    #    assert all(v is not None for v in internal_x)
    tree.extend(internal_tree)    
    X_word.extend(internal_word)
    X_index.extend(internal_index)
    X_word.append(ini_word)
    #if max_degree is not None:
    #    assert all(len(t) == max_degree + 1 for t in tree)
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
    #print X_word    
    #print type(X_word)    
    return (np.array(X_word, dtype='float32'),
            np.array(X_index, dtype='int32'),
            np.array(tree, dtype='int32'))
    #return (np.array(X_word),
    #        np.array(X_index),
    #        np.array(tree))        

def _get_tree_path(root_node):
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
    #print 'layer:', layers
    tree = []
    word = []
    index = []
    for layer in layers:
        for node in layer:
            if not node.children:
               continue 
            #child_idxs = [child.idx for child in ]  ## idx of child node
            for child in node.children:
                tree.append([node.idx, child.idx])
                word.append(child.word if child.word is not None else -1)
                index.append(child.index if child.index is not None else -1)
            '''if max_degree is not None:
                child_idxs.extend([-1] * (max_degree - len(child_idxs)))
            assert not any(idx is None for idx in child_idxs)

            node.idx = idx
            tree.append(child_idxs + [node.idx])
            internal_word.append(node.word if node.word is not None else -1)
            internal_index.append(node.index if node.index is not None else -1)
            idx += 1'''

    return tree, word, index

################################ tree rnn class ######################################
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
    def init_matrix(self, shape):
        std = 0.1*torch.ones(shape)
        return Var(torch.normal(mean=0.0, std=std),requires_grad=True)
    
    def init_vector(self, shape):
        return Var(torch.zeros(shape), requires_grad=True)

    def hard_sigmoid(self, x):
        """
        Computes element-wise hard sigmoid of x.
        See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
        """
        x = (0.2 * x) + 0.5
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)
        return x

    def node_forward(self, word, index, parent_h):
        #h_tilde = torch.sum(child_h, 0)

        child_embedding = self.embedding(index)
        # logger.info('the size of parent_embedding:%s, the size of parent_word: %s'%(parent_embedding.size(), parent_word.size()))
        child_xe = torch.squeeze(torch.mm(child_embedding.transpose(1, 0), torch.unsqueeze(word.float(), 1)))

        z = self.hard_sigmoid(torch.squeeze(self.W_z.mm(torch.unsqueeze(child_xe,1)) + self.U_z.mm(torch.unsqueeze(parent_h,1))) + self.b_z)
        r = self.hard_sigmoid(torch.squeeze(self.W_r.mm(torch.unsqueeze(child_xe,1)) + self.U_r.mm(torch.unsqueeze(parent_h,1))) + self.b_r)
        c = F.tanh(torch.squeeze(self.W_h.mm(torch.unsqueeze(child_xe,1)) + self.U_h.mm(torch.unsqueeze(parent_h*r,1))) + self.b_h)
        h = z*parent_h + (1 - z)*c
        
        return h

    # def init_node_child(self, x):
    #     return self.init_vector(self.hidden_dim)

    def recurrence(self, x_word, x_index, node_info, node_h, last_h):
        parent_h = node_h[node_info[0]]

        child_h = self.node_forward(x_word, x_index, parent_h)

        node_h = torch.cat([node_h[:node_info[1]], child_h.view(1, self.hidden_dim), node_h[node_info[1]+1:]], 0)

        return node_h, child_h

    def forward(self, x_word, x_index, num_parent, tree):
        for i in range(len(x_word)):
            if i == 0:
                init_node_h = torch.unsqueeze(self.init_vector(self.hidden_dim), 0)
            else:
                init_node_h = torch.cat([init_node_h, torch.unsqueeze(self.init_vector(self.hidden_dim), 0)], 0)
        dummy = self.init_vector(self.hidden_dim)
        if(self.cudaFlag):
            init_node_h = init_node_h.cuda()
            dummy = dummy.cuda()
        for i in range(len(x_word)-1):
            init_node_h, dummy = self.recurrence(x_word[i], x_index[i], tree[i], init_node_h, dummy)
            if(i==0):
                child_hs = torch.unsqueeze(dummy, 0)
            else:
                child_hs = torch.cat([child_hs, torch.unsqueeze(dummy, 0)], 0)
        # print(num_parent)
        # output = torch.squeeze(torch.unsqueeze(torch.max(child_hs[num_parent-1:], 0)[0], 0).mm(self.W_out)+self.b_out)
        output = torch.squeeze(torch.max(child_hs[num_parent-1:], 0)[0])
        # return F.softmax(output)
        return output

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