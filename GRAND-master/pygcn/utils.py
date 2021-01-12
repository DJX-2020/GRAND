import sys

import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import random as rm

import torch


# 这个函数看来是构建标签的 one-hot 矩阵
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def label_vector_generator(adj, label_vector, index = [], order=1, shuffle=False, style=0):
    label_vectors = []
    lv = torch.zeros(label_vector.shape)
    lv[index, :] = label_vector[index, :]
    for i in range(order):
        label_vectors.append(lv)
        if i!= (order-1):
            lv = torch.spmm(adj, lv)
    if style == 0:
        return sum(label_vectors)*1.0/order
    return torch.cat(label_vectors, 1)


def feature_generator(adj, features, order=0):
    n = features.shape[0]
    index = np.random.permutation(n)
    index_1 = index[: n//2]
    index_2 = index[n//2 :]
    mask_1 = torch.zeros(n,1)
    mask_2 = torch.zeros(n,1)
    mask_1[index_1] = 1
    mask_2[index_2] = 1

    features_1 = [mask_1.cuda() * features]
    features_2 = [mask_2.cuda() * features]

    alpha = 1
    for i in range(order):
        features_1.append(alpha*torch.spmm(adj, features_1[-1]) + (1-alpha)*features_1[0])
    for i in range(order):
        features_2.append(alpha*torch.spmm(adj, features_2[-1]) + (1-alpha)*features_2[0])

    return sum(features_1)*1./(order+1), sum(features_2)*1./(order+1)


def MMD(x, y, alpha=0.5):
    n_x, n_y = x.size(0), y.size(0)

    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    rrx = (xx.diag().unsqueeze(1).expand(n_x, n_y))
    rry = (yy.diag().unsqueeze(0).expand(n_x, n_y))

    K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(- alpha * (rrx + rry - 2*zz))

    loss = 1./(n_x*(n_x-1)) * torch.sum(K) + 1./(n_y*(n_y-1)) * torch.sum(L) - (2./(n_x*n_y)) * torch.sum(P)

    return 0.5*loss


def MMD_same(x, y, alpha=0.5):
    n_x, n_y = x.size(0), y.size(0)

    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2*zz))

    loss = 1./(n_x*(n_x-1)) * torch.sum(K) + 1./(n_y*(n_y-1)) * torch.sum(L) - (2./(n_x*n_y)) * torch.sum(P)

    return loss


# 准确率函数，和我写的其实一样
def accuracy(output, labels):
    preds = output.max(dim=1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# 加载数据，这个函数应该是最重要的了
def load_data(dataset_str='cora'):
    # 一共有三个数据集：citeseer, cora, pubmed
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        # 一个 with open 语句就把图数据文件打开了？也太容易了吧
        # with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
        with open("../data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    # test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # adj = adj + sp.eye(adj.shape[0])  # A + In，sp.eye(size) 生成大小为 size 的单位对角矩阵
    # D1_ = np.array(adj.sum(axis=1)) ** (-0.5)
    # D2_ = np.array(adj.sum(axis=0)) ** (-0.5)
    # D1_ = sp.diags(D1_[:, 0], format='csr')
    # D2_ = sp.diags(D2_[0, :], format='csr')
    # A_ = adj.dot(D1_)
    # A_ = D2_.dot(A_)
    #
    # D1 = np.array(adj.sum(axis=1)) ** (-0.5)
    # D2 = np.array(adj.sum(axis=0)) ** (-0.5)
    # D1 = sp.diags(D1[:, 0], format='csr')
    # D2 = sp.diags(D2[0, :], format='csr')
    #
    # A = adj.dot(D1)
    # A = D2.dot(A)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]  # onehot

    size = labels.shape[0]
    print("Size of dataset {}: {}".format(dataset_str, size))

    all_indices = list(range(size))
    rm.shuffle(all_indices)

    train_size = int(size * 0.8)
    val_size = int(size * 0.1)
    print("Train size: {}, Val/Test size: {}".format(train_size, val_size))

    idx_train = all_indices[: train_size]
    idx_val = all_indices[train_size: train_size + val_size]
    idx_test = all_indices[train_size + val_size:]
    # idx_val = range(len(y), len(y) + 500)
    # idx_test = test_idx_range.tolist()

    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y) + 500)
    # idx_test = test_idx_range.tolist()

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    # A = sparse_mx_to_torch_sparse_tensor(A)
    A = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, features, labels, idx_train, idx_val, idx_test


if __name__ == '__main__':
    A, features, labels, idx_train, idx_val, idx_test = load_data()
    print("A: ", A)  # 邻接矩阵是一个二维稀疏矩阵就
    # 特征矩阵形状为 [2708, 1433]，表示一共有 2708 篇文章，特征用 1433 个词汇是否在文中出现表示
    # 1433 个词汇中的每个单词在文章中是存在(1)不存在(0)
    print("features: ", features)
    print(features.shape)
    # 相应的标签也有 2708 个，应该是有 7 类
    print("labels: ", labels)
    print(labels.shape)
    # 训练集数据的下标
    print("idx_train: ", idx_train.size()[0])
    # 验证集数据的下标
    print("idx_val: ", idx_val.size()[0])
    # 测试集数据的下标
    print("idx_test: ", idx_test.size()[0])
