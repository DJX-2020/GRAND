import torch
from utils import load_data, accuracy
from models import GCNC
import numpy as np
import torch.optim as optim
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

# 超参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# 损失函数
criterion = torch.nn.CrossEntropyLoss()


def train(dataset='cora'):
    # 加载数据集
    print("loading graph {}...".format(dataset))
    # 加载出的 labels 是一个 [data_size] 的向量
    adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset_str=dataset)
    # print(labels.size())
    if args.cuda:
        features, adj, labels = features.cuda(), adj.cuda(), labels.cuda()
        idx_train, idx_val, idx_test = idx_train.cuda(), idx_val.cuda(), idx_test.cuda()

    # 加载模型
    model = GCNC(in_feat=features.size()[1], n_class=labels.max().item() + 1)

    if args.cuda:
        model = model.cuda()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 记录器、开始时间记录
    train_rec, val_rec = [], []
    acc_test = None  # 在这里先声明一下，免得编译器显示异常
    t_start = time.time()

    for epoch in range(args.epochs):  # 记录训练的开始时间
        epoch_start = time.time()

        # 训练标志，执行了这句话以后模型中的 self.training 应该就变成 True
        model.train()
        output, _ = model(features, adj)

        # 记录训练中的损失和准确度
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])

        # 优化三部曲
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # 验证、测试
        model.eval()
        output, _ = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        acc_test = accuracy(output[idx_test], labels[idx_test])

        # print('Epoch: {:04d}'.format(epoch + 1),
        #       'loss_train: {:.6f}'.format(loss_train.item()),
        #       'loss_val: {:.6f}'.format(loss_val.item()),
        #       'acc_train: {:.2f}%'.format(100. * acc_train.item()),
        #       'acc_val: {:.2f}%'.format(100. * acc_val.item()),
        #       'train_time: {:.4f}s'.format(time.time() - epoch_start))
        #
        # print("\tTest set results: accuracy= {:.2f}%".format(100. * acc_test.item()))

        train_r, val_r = (loss_train, acc_train), (loss_val, acc_val)
        train_rec.append(train_r)
        val_rec.append(val_r)

    t_span = time.time() - t_start
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(t_span))

    # return train_rec, val_rec, '{:.2f}%'.format(100 * acc_test.item()), '{:.4f}s'.format(t_span)
    return train_rec, val_rec, 100 * acc_test.item(), t_span


if __name__ == '__main__':
    dataset = ['cora', 'citeseer', 'pubmed']
    # dataset = ['cora']
    results = {'cora': [[], []], 'citeseer': [[], []], 'pubmed': [[], []]}

    for dset in dataset:
        for i in range(100):
            # Train model
            train_rec, val_rec, test_acc, t = train(dset)
            results[dset][0].append(test_acc)
            results[dset][1].append(t)
            print("Dataset: {}, time {:03d}, acc: {:.2f}%".format(dset, i + 1, test_acc))
        print()

    for key, val in results.items():
        print("\nFinal results of {}: ".format(key))
        print('Accuracy: average {:.2f}%, max {:.2f}%, min {:.2f}%'.format(
            np.mean(val[0]), np.max(val[0]), np.min(val[0])
        ))
        print('Average time: {:.4f}s'.format(np.mean(val[1])))

        plt.figure()
        plt.plot(range(100), val[0])
        plt.xlabel('Times')
        plt.ylabel('Accuracy')
        plt.title('Train accuracy of {} for 100 times'.format(key))
        plt.show()

