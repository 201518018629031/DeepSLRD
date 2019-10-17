# -*- coding: utf-8 -*-
import sys
import logging
#logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import TD_RvNN_torch
from gcn_models import GCN
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.testing import assert_array_almost_equal
import time
import datetime
import random
from evaluate import *
import networkx as nx
import scipy.sparse as sp
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_str',type=str, default='Twitter15',help='choose dataset, you can choose either "Twitter15" or "Twitter16"')
parser.add_argument('--fold_index',type=str, default='2',help='fold index, choose from 0-4')
parser.add_argument('--gcn_hidden', type=int, default=64, help='Number of gcn hidden units')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN dropout rate (1 - keep probability)')
parser.add_argument('--gcn_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--gcn_output',type=int, default=4, help='dimension of the user')
parser.add_argument('--cuda_id', type=int, default=0, help='Set the gpu id')

args = parser.parse_args()
torch.cuda.set_device(args.cuda_id)

obj = args.dataset_str # choose dataset, you can choose either "Twitter15" or "Twitter16"
fold = args.fold_index # fold index, choose from 0-4
tag = "_u2b"
vocabulary_size = 5000
hidden_dim = 100
Nclass = 4
Nepoch = 600
lr = 0.005

unit="TD_RvNN-"+obj+str(fold)+'-vol.'+str(vocabulary_size)+tag

treePath = './resource/data.TD_RvNN.vol_'+str(vocabulary_size)+'.txt'

trainPath = "./nfold/RNNtrainSet_"+obj+str(fold)+"_tree.txt"
testPath = "./nfold/RNNtestSet_"+obj+str(fold)+"_tree.txt"
labelPath = "./resource/"+obj+"_label_All.txt"

#the path graph imformation
graphPath = './ind_'+args.dataset_str+'.graph'
featuresPath = './ind_'+args.dataset_str+'.features'
tweetid2useridx = './ind_'+args.dataset_str+'.poster'

#floss = open(lossPath, 'a+')

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

################################### tools #####################################
def str2matrix(Str, MaxL): # str = index:wordfreq index:wordfreq
    wordFreq, wordIndex = [], []
    l = 0
    for pair in Str.split(' '):
        wordFreq.append(float(pair.split(':')[1]))
        wordIndex.append(int(pair.split(':')[0]))
        l += 1
    ladd = [ 0 for i in range( MaxL-l ) ]
    wordFreq += ladd
    wordIndex += ladd
    #print MaxL, l, len(Str.split(' ')), len(wordFreq)
    #print Str.split(' ')
    return wordFreq, wordIndex

def loadLabel(label, l1, l2, l3, l4):
    labelset_nonR, labelset_f, labelset_t, labelset_u = ['news', 'non-rumor'], ['false'], ['true'], ['unverified']
    if label in labelset_nonR:
       y_train = [1,0,0,0]
       l1 += 1
    if label in labelset_f:
       y_train = [0,1,0,0]
       l2 += 1
    if label in labelset_t:
       y_train = [0,0,1,0]
       l3 += 1
    if label in labelset_u:
       y_train = [0,0,0,1]
       l4 += 1
    return y_train, l1,l2,l3,l4

def constructTree(tree):
    ## tree: {index1:{'parent':, 'maxL':, 'vec':}
    ## 1. ini tree node
    index2node = {}
    for i in tree:
        node = TD_RvNN_torch.Node_tweet(idx=i)
        index2node[i] = node
    ## 2. construct tree
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]
        wordFreq, wordIndex = str2matrix( tree[j]['vec'], tree[j]['maxL'] )
        #print tree[j]['maxL']
        nodeC.index = wordIndex
        nodeC.word = wordFreq
        #nodeC.time = tree[j]['post_t']
        ## not root node ##
        if not indexP == 'None':
           nodeP = index2node[int(indexP)]
           nodeC.parent = nodeP
           nodeP.children.append(nodeC)
        ## root node ##
        else:
           root = nodeC
    ## 3. convert tree to DNN input
    parent_num = tree[j]['parent_num']
    ini_x, ini_index = str2matrix( "0:0", tree[j]['maxL'] )
    #x_word, x_index, tree = tree_gru_u2b.gen_nn_inputs(root, ini_x, ini_index)
    x_word, x_index, tree = TD_RvNN_torch.gen_nn_inputs(root, ini_x)
    return x_word, x_index, tree, parent_num

################################# loas data ###################################
def loadData():
    logger.info("loading tree label")
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        label, eid = line.split('\t')[0], line.split('\t')[2]
        labelDic[eid] = label.lower()
    logger.info(len(labelDic))

    logger.info("reading tree") ## X
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
        parent_num, maxL = int(line.split('\t')[3]), int(line.split('\t')[4])
        Vec =  line.split('\t')[5]
        if eid not in treeDic:
           treeDic[eid] = {}
        treeDic[eid][indexC] = {'parent':indexP, 'parent_num':parent_num, 'maxL':maxL, 'vec':Vec}
    logger.info('tree no:%d'%len(treeDic))

    logger.info("reading graph")
    with open(graphPath, 'rb') as f:
        if sys.version_info > (3, 0):
            graph = pkl.load(f, encoding='latin1')
        else:
            graph = pkl.load(f)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    logger.info("reading the features of graph node")
    with open(featuresPath, 'rb') as f:
        if sys.version_info > (3, 0):
            features = pkl.load(f, encoding='latin1')
        else:
            features = pkl.load(f)

    # features = features.tolil()

    logger.info("reading the map of tweete to user in graph")
    with open(tweetid2useridx, 'rb') as f:
        if sys.version_info > (3, 0):
            tw2useridx = pkl.load(f, encoding='latin1')
        else:
            tw2useridx = pkl.load(f)

    graph_train = []
    graph_test = []

    logger.info("loading train set")
    tree_train, word_train, index_train, y_train, parent_num_train, c = [], [], [], [], [], 0
    l1,l2,l3,l4 = 0,0,0,0
    for eid in open(trainPath):
        #if c > 8: break
        eid = eid.rstrip()
        if eid not in labelDic: continue
        if eid not in treeDic: continue
        if len(treeDic[eid]) <= 0:
           #print labelDic[eid]
           continue
        ## 1. load label
        graph_train.append(tw2useridx[eid])
        label = labelDic[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_train.append(y)
        ## 2. construct tree
        #print eid
        x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
        tree_train.append(tree)
        word_train.append(x_word)
        index_train.append(x_index)
        parent_num_train.append(parent_num)
        #print treeDic[eid]
        #print tree, child_num
        #exit(0)
        c += 1
    logger.info('%d, %d, %d, %d'%(l1,l2,l3,l4))

    logger.info("loading test set")
    tree_test, word_test, index_test, parent_num_test, y_test, c = [], [], [], [], [], 0
    l1,l2,l3,l4 = 0,0,0,0
    for eid in open(testPath):
        #if c > 4: break
        eid = eid.rstrip()
        if eid not in labelDic: continue
        if eid not in treeDic: continue
        if len(treeDic[eid]) <= 0:
           #print labelDic[eid]
           continue
        ## 1. load label
        graph_test.append(tw2useridx[eid])
        label = labelDic[eid]
        y, l1,l2,l3,l4 = loadLabel(label, l1, l2, l3, l4)
        y_test.append(y)
        ## 2. construct tree
        x_word, x_index, tree, parent_num = constructTree(treeDic[eid])
        tree_test.append(tree)
        word_test.append(x_word)
        index_test.append(x_index)
        parent_num_test.append(parent_num)
        c += 1

    features = clear_data(features)
    features = normalize(features)
    features = torch.FloatTensor(features)
    adj = preprocess_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    graph_train = torch.LongTensor(graph_train)
    graph_test = torch.LongTensor(graph_test)

    # logger.info('the graph_train:%s'%str(graph_train))
    logger.info('the legth of graph_train:%s'%str(len(graph_train)))
    logger.info('the legth of graph_test:%s'%str(len(graph_test)))

    logger.info('%d, %d, %d, %d'%(l1,l2,l3,l4))
    logger.info("train no:%d, %d, %d, %d, %d"%(len(tree_train), len(word_train), len(index_train),len(parent_num_train), len(y_train)))
    logger.info("test no:%d, %d, %d, %d, %d"%(len(tree_test), len(word_test), len(index_test), len(parent_num_test), len(y_test)))
    logger.info("dim1 for 0:%d, %d, %d"%(len(tree_train[0]), len(word_train[0]), len(index_train[0])))
    logger.info("case 0:%s, %s, %s, %s"%(str(tree_train[0][0]), str(word_train[0][0]), str(index_train[0][0]), str(parent_num_train[0])))
    #print index_train[0]
    #print word_train[0]
    #print tree_train[0]
    return adj, features, graph_train, graph_test, tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test

def clear_data(mx):
    mx[np.isnan(mx)] = 0.
    colmax = np.max(mx, 0)
    colmin = np.min(mx, 0)
    res = (mx - colmin)/(colmax - colmin)
    res[np.isnan(res)] = 0.

    return res

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

##################################### MAIN ####################################
## 1. load tree & word & index & label
adj, features, graph_train, graph_test, tree_train, word_train, index_train, parent_num_train, y_train, tree_test, word_test, index_test, parent_num_test, y_test = loadData()

## 2. ini RNN model
t0 = time.time()
device = torch.cuda.is_available()
model = TD_RvNN_torch.RvNN(device, vocabulary_size, hidden_dim, args.gcn_hidden)
gcn_model = GCN(nfeat=features.shape[1],
            nhid=args.gcn_hidden,
            nclass=args.gcn_output,
            dropout=args.gcn_dropout)
co_model = TD_RvNN_torch.RvNN_Co_GCN(hidden_dim+args.gcn_output, Nclass)
if device:
    model.cuda()
    gcn_model.cuda()
    co_model.cuda()
t1 = time.time()
logger.info('Recursive model established, %s'%str((t1-t0)/60))

## 3. looping SGD
# criterion = nn.MSELoss(reduce = True, size_average=False)
criterion = nn.CrossEntropyLoss()
params = list(model.parameters())+list(gcn_model.parameters())+list(co_model.parameters())
optimizer = torch.optim.Adagrad(params, lr=lr, weight_decay=args.weight_decay)
losses_5, losses = [], []
num_examples_seen = 0
if device:
    features = features.cuda()
    adj = adj.cuda()
    graph_train = graph_train.cuda()
    graph_test = graph_test.cuda()

# logger.info(str(parent_num_train))
for epoch in range(Nepoch):
    ## one SGD
    model.train()
    gcn_model.train()
    co_model.train()
    optimizer.zero_grad()
    indexs = [i for i in range(len(y_train))]
    #random.shuffle(indexs)
    for i in indexs:
        if device:
            word_train_instance = torch.from_numpy(word_train[i]).cuda()
            index_train_instance = torch.from_numpy(index_train[i]).long().cuda()
            parent_num_train_instance = torch.IntTensor([parent_num_train[i]]).cuda()
            tree_train_instance = torch.from_numpy(tree_train[i]).cuda()
            y_train_instance = torch.Tensor(y_train[i]).cuda()
        else:
            word_train_instance = torch.from_numpy(word_train[i])
            index_train_instance = torch.from_numpy(index_train[i]).long()
            parent_num_train_instance = torch.IntTensor([parent_num_train[i]])
            tree_train_instance = torch.from_numpy(tree_train[i])
            y_train_instance = torch.Tensor(y_train[i])
        rvnn_out = model(word_train_instance, index_train_instance, parent_num_train_instance, tree_train_instance)
        gcn_out = gcn_model(features, adj)
        pred_y = co_model(torch.cat([rvnn_out,gcn_out[graph_train[i]]], 0))

        pred_y = torch.unsqueeze(pred_y, 0)
        y_train_instance = torch.unsqueeze(y_train_instance, 0)

        loss = criterion(pred_y, y_train_instance.max(1)[1])
        loss.backward()
        optimizer.step()
        #print loss, pred_y
        losses.append(round(float(loss.detach().cpu().numpy()),2))

        num_examples_seen += 1
    logger.info("epoch=%d: loss=%f" % ( epoch, np.mean(losses) ))

    ## cal loss & evaluate
    if epoch % 5 == 0:
        model.eval()
        gcn_model.eval()
        co_model.eval()
        losses_5.append((num_examples_seen, np.mean(losses)))
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, np.mean(losses)))

        prediction = []
        for j in range(len(y_test)):
            #print j
            if device:
                word_test_instance = torch.from_numpy(word_test[j]).cuda()
                index_test_instance = torch.from_numpy(index_test[j]).long().cuda()
                # logger.info(parent_num_train[i])
                parent_num_test_instance = torch.IntTensor([parent_num_test[j]]).cuda()
                tree_test_instance = torch.from_numpy(tree_test[j]).cuda()
            else:
                word_test_instance = torch.from_numpy(word_test[j])
                index_test_instance = torch.from_numpy(index_test[j]).long()
                parent_num_test_instance = torch.IntTensor([parent_num_test[j]])
                tree_test_instance = torch.from_numpy(tree_test[j])
            rvnn_out = model(word_test_instance, index_test_instance, parent_num_test_instance, tree_test_instance)
            gcn_out = gcn_model(features, adj)
            predict = co_model(torch.cat([rvnn_out,gcn_out[graph_test[j]]], 0))

            predict = F.softmax(predict)

            prediction.append(predict.detach().cpu().numpy())
        res = evaluation_4class(prediction, y_test)
        logger.info('results:%s'%str(res))

        ## Adjust the learning rate if loss increases
        if len(losses_5) > 1 and losses_5[-1][1] > losses_5[-2][1]:
            lr = lr * 0.5
            adjust_learning_rate(optimizer, lr)
            logger.info("Setting learning rate to %f"%lr)

    losses = []
model_path = './td_model/'+args.dataset_str+'_'+'f'+args.fold_index+'_'+'train_'+str(epoch)+'_model_params.pkl'
gcn_path = './td_model/'+args.dataset_str+'_'+'f'+args.fold_index+'_'+'train_'+str(epoch)+'_gcn_params.pkl'
co_path = './td_model/'+args.dataset_str+'_'+'f'+args.fold_index+'_'+'train_'+str(epoch)+'_co_params.pkl'
torch.save(model.state_dict(), model_path)
torch.save(gcn_model.state_dict(), gcn_path)
torch.save(co_model.state_dict(), co_path)
