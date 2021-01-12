import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# 他 utils 里写了那么多函数，都没用到，就用了一个 load_data，一个 accuracy
from pygcn.utils import load_data, accuracy
from pygcn.models import GCN, MLP

import matplotlib.pyplot as plt

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
def rand_prop(features, adj, training=True):
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
    features = propagate(features, adj, args.order)
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


def train(num_epochs, dataset='cora'):
    # 加载数据
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)

    # 模型
    model = MLP(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1,
                input_droprate=args.input_droprate, hidden_droprate=args.hidden_droprate, use_bn=args.use_bn)

    # 优化器，使用了学习率衰减
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    t_start = time.time()  # 总训练时间
    losses = []  # 损失记录
    accs = []  # 准确度记录
    bad_counter = 0  # 这个应该是用来记录损失没有改变的迭代轮数

    loss_best, loss_mn = np.inf, np.inf
    acc_best, acc_mx = 0.0, 0.0

    best_epoch = 0

    for epoch in range(num_epochs):
        t = time.time()

        S = args.sample  # 数据增广的次数 S

        model.train()  # 训练标志

        # 一共做了 S 次图像增广
        x_list = []
        for s in range(S):
            x_list.append(rand_prop(features, adj, training=True))
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
        loss_train = loss_train + args.lam * loss_consis

        # 这个地方就有问题了，做准确率验证的时候到底以哪个增广的输出为标准呢
        acc_train = accuracy(output_list[0][idx_train], labels[idx_train])

        # 优化三部曲
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # 验证
        model.eval()
        _x = rand_prop(features, adj, training=False)
        output = model(_x)
        output = torch.log_softmax(output, dim=-1)
        # 验证结果
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        # 测试结果
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.4f}'.format(loss_train.item()),
        #       'acc_train: {:.4f}'.format(acc_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'acc_val: {:.4f}'.format(acc_val.item()),
        #       'time: {:.4f}s'.format(time.time() - t))
        #
        # print("\tTest set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))

        losses.append((loss_train.item(), loss_val.item()))
        accs.append((acc_train.item(), acc_val.item(), acc_test.item()))

        # print('Current bad_counter: {}'.format(bad_counter))

        if losses[-1][-1] <= loss_mn or accs[-1][-1] >= acc_mx:  # or epoch < 400:
            if losses[-1][-1] <= loss_best:  # and acc_values[-1] >= acc_best:
                loss_best = losses[-1][-1]
                acc_best = accs[-1][-1]
                best_epoch = epoch
                torch.save(model.state_dict(), dataset + '.pkl')

            loss_mn = np.min((losses[-1][-1], loss_mn))
            acc_mx = np.max((accs[-1][-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        # print(bad_counter, loss_mn, acc_mx, loss_best, acc_best, best_epoch)
        if bad_counter == args.patience:
            # print('Early stop! Min loss: ', loss_mn, ', Max accuracy: ', acc_mx)
            # print('Early stop model validation loss: ', loss_best, ', accuracy: ', acc_best)
            break

    t_total = time.time() - t_start
    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(t_total))
    # print("Best epoch is {}".format(best_epoch))

    # Restore best model
    # print('Loading {} th epoch'.format(best_epoch))
    # model.load_state_dict(torch.load(dataset + '.pkl'))

    return losses, accs, 100 * acc_best, t_total


if __name__ == '__main__':
    dataset = ['cora', 'citeseer', 'pubmed']
    # dataset = ['cora']
    results = {'cora': [[], []], 'citeseer': [[], []], 'pubmed': [[], []]}

    for i in range(100):
        for dset in data_list:
            losses, accs, acc_best, t = train(args.epochs, dataset=dset)
            results[dset][0].append(acc_best)
            results[dset][1].append(t)
            print("Dataset: {}, time {:03d}, acc: {:.2f}%".format(dset, i + 1, acc_best))
        print()

    for key, val in results.items():
        print("\nFinal results of {}: ".format(key))
        print('Accuracy: average {:.2f}%, max {:.2f}%, min {:.2f}%'.format(
            np.mean(val[0]), np.max(val[0]), np.min(val[0])
        ))
        print('Average time: {:.4f}s'.format(np.mean(val[1])))

