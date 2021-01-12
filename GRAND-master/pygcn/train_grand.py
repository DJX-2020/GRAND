from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# 他 utils 里写了那么多函数，都没用到，就用了一个 load_data，一个 accuracy
from pygcn.utils import load_data, accuracy
from pygcn.models import GCN, MLP

# 这是个啥我就不知道了
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

data_list = ['cora', 'citeseer', 'pubmed']

# Training settings
# 如果我没猜错的话这里应该是超参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--input-droprate', type=float, default=0.5, help='Dropout rate of the input layer (1 - keep probability).')
parser.add_argument('--hidden-droprate', type=float, default=0.5, help='Dropout rate of the hidden layer (1 - keep probability).')
parser.add_argument('--dropnode-rate', type=float, default=0.5, help='Dropnode rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--order', type=int, default=5, help='Propagation step')
parser.add_argument('--sample', type=int, default=4, help='Sampling times of dropnode')
parser.add_argument('--tem', type=float, default=0.5, help='Sharpening temperature')
parser.add_argument('--lam', type=float, default=1., help='Lamda')
parser.add_argument('--dataset', type=str, default=data_list, help='Data set')
# parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device')
parser.add_argument('--use_bn', action='store_true', default=False, help='Using Batch Normalization')
# torch.cuda.set_device(args.cuda_device)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# print(args.cuda_device)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 设置训练所用数据集
dataset = args.dataset[2]

# Load data
# 这个加载数据集没有什么高级的技术，就是自己写了一个加载数据集的方法以及自己划分了训练集、验证集、测试集，没有用到 DataLoader
# 完全可以不管数据集加载的过程，就把它当成和 MNIST、Caffer 一样是别人提供好的就行
# 最后返回的结果是邻接矩阵、特征矩阵以及标签矩阵（一维向量）
A, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
idx_unlabel = torch.arange(idx_train.shape[0], labels.shape[0] - 1, dtype=torch.int)

# Model and optimizer
model = MLP(nfeat=features.shape[1],  # 每一个样本的特征维数
            nhid=args.hidden,  # MLP 隐层单元数
            nclass=labels.max().item() + 1,  # 总类别数，找到 labels 中最大的数值，加一即可
            input_droprate=args.input_droprate,  # 输入的 dropout 率
            hidden_droprate=args.hidden_droprate,  # 隐层的 dropout 率
            use_bn=args.use_bn)  # 是否使用批归一化

# 优化器，使用了学习率衰减
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    A = A.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_unlabel = idx_unlabel.cuda()


def propagate(feature, adj, order):
    # order 是超参数，默认为 5，在论文中就是混级传播时 A 的次数 k
    x = feature
    xs = feature
    for i in range(order):
        # torch.sparse.mm(m1, m2) 是做稀疏矩阵 m1 和稠密矩阵 m2 的矩阵乘法
        x = torch.sparse.mm(adj, x).detach_()
        xs.add_(x)

    # return xs.div_(order + 1.0).detach_()
    return (xs / (order + 1.0)).detach_()


# key1: 随机传播
def rand_prop(features, training=True):
    n = features.shape[0]  # 样本个数
    drop_rate = args.dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    if training:
        # 第一步，为每一个节点获取一个二进制掩模矩阵，torch.bernoulli 就是伯努利分布
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        # 第二步，做节点特征与掩模矩阵的元素乘法
        features = masks.cuda() * features
    else:
        features = features * (1. - drop_rate)
    features = propagate(features, A, args.order)
    return features


# key2: 一致性损失
def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.

    for p in ps:
        sum_p = sum_p + p
    # 这一项是 Z_
    avg_p = sum_p / len(ps)
    # p2 = torch.exp(logp2)

    # 这一项是 Z_'（公式2）
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return loss


def train(epoch):
    t = time.time()

    x = features
    S = args.sample  # 数据增广的次数 S

    model.train()  # 训练标志

    # 一共做了 S 次图像增广
    x_list = []
    for s in range(S):
        x_list.append(rand_prop(x, training=True))
    # 对应一个增广就有一个输出（logits）
    output_list = []
    for s in range(S):
        output_list.append(model(x_list[s]))
    # 损失是每一个增广的输出与真实标签的损失之和
    loss_train = 0.
    for s in range(S):
        loss_train += F.nll_loss(output_list[s][idx_train], labels[idx_train])

    loss_train = loss_train / S  # 对损失做了一个平均
    loss_consis = consis_loss(output_list)  # 一致性损失
    loss = loss_train + args.lam * loss_consis

    # 这个地方就有问题了，做准确率验证的时候到底以哪个增广的输出为标准呢
    acc_train = accuracy(output_list[0][idx_train], labels[idx_train])

    # 优化三部曲
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 验证
    model.eval()
    _x = rand_prop(x, training=False)
    output = model(_x)
    output = torch.log_softmax(output, dim=-1)

    # nll_loss：negative log likelihood loss，负对数似然损失
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.item(), acc_val.item()


def test():
    model.eval()
    x = features
    x = rand_prop(x, training=False)
    output = model(x)
    output = torch.log_softmax(output, dim=-1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


if __name__ == '__main__':
    t_total = time.time()  # 总训练时间

    loss_values = []  # 损失记录
    acc_values = []  # 准确度记录
    bad_counter = 0  # 这个应该是用来记录损失没有改变的迭代轮数

    loss_best, loss_mn = np.inf, np.inf
    acc_best, acc_mx = 0.0, 0.0

    best_epoch = 0

    for epoch in range(args.epochs):
        # 训练
        l, a = train(epoch)
        loss_values.append(l)
        acc_values.append(a)

        print('Current bad_counter: {}'.format(bad_counter))

        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:  # or epoch < 400:
            if loss_values[-1] <= loss_best:  # and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                torch.save(model.state_dict(), dataset +'.pkl')

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        if bad_counter == args.patience:
            print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load(dataset + '.pkl'))

    test()
