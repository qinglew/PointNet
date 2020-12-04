"""
Small modules for the structure of PointNet.
Author: Carlos Leo
Time:   2020/10/15 - 15:12
Version: 1.0
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """
    T-Net which produce a transformation matrix to align input or feature space.

    k: the channel of the input
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k

        # shared mlp(64, 128, 1024)
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # fully connected layer(512, 256, k*k)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # max symmetric function which change dims (B, C, N)
        # into dims (B, C)
        x = torch.max(x, dim=-1)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        x = x.view(-1, self.k, self.k)

        eye = torch.eye(self.k)
        if x.is_cuda:
            eye = eye.cuda()
        eyes = eye.repeat(x.size()[0], 1, 1)
        x += eyes

        return x


def feature_transform_regularization(trans):
    """
    $L_{reg}=||I-AA^{T}||^2_F$, references the original implement

    :param trans: transformation matrix by T-Net in features
    :return:
    """
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def to_one_hots(y, categories):
    """
    Encode the labels into one-hot coding.

    :param y: labels for a batch data with size (B,)
    :param categories: total number of kinds for the label in the dataset
    :return: (B, categories)
    """
    y_ = torch.eye(categories)[y.data.cpu().numpy()]
    if y.is_cuda:
        y_ = y_.cuda()
    return y_


if __name__ == '__main__':

    def setup_seed(seed):
        """
        Set a random seed to generate same random numbers
        :param seed: random seed number
        :return:
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(20)

    # 随机生成数据测试T-Net
    x = torch.rand((10, 3, 1024))

    t_net = TNet()
    print(t_net(x).size())

    x2 = torch.rand((10, 64, 1024))
    t_net64 = TNet(k=64)
    print(t_net64(x2).size())
