"""
The models of the PointNet, which contains three models for
three different tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import TNet, to_one_hots


class ClassificationPointNet(nn.Module):
    """
    The architecture of the classification network.

    input_channels: point coordinate, maybe with normals
    input_transform: T-Net for input points
    feature_transform: T-Net for intermediate features
    """
    def __init__(self, categories, input_channels=3, input_transform=True, feature_transform=True):
        super(ClassificationPointNet, self).__init__()
        self.input_channels = input_channels  # 考虑到可能加上法向量
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.t_net1 = TNet(k=self.input_channels)
        if self.feature_transform:
            self.t_net2 = TNet(k=64)

        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, categories)

    def forward(self, x):
        # t-net transform for input point cloud
        transformer1 = None
        if self.input_transform:
            transformer1 = self.t_net1(x)
            x = x.permute(0, 2, 1)
            x = torch.bmm(x, transformer1)
            x = x.permute(0, 2, 1)

        # mlp(64, 64)
        x = F.relu(self.bn1(self.conv1(x)))

        # t-net transform for feature
        transformer2 = None
        if self.feature_transform:
            transformer2 = self.t_net2(x)
            x = x.permute(0, 2, 1)
            x = torch.bmm(x, transformer2)
            x = x.permute(0, 2, 1)

        # mlp(64, 128, 1024)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # max symmetric function
        x = torch.max(x, dim=-1)[0]

        # fully connected layers (512, 256, k)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(F.dropout(self.fc2(x), 0.3)))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)  # (batch_size, categories)
        return x, transformer1, transformer2


class PartSegmentationPointNet(nn.Module):
    """
    The original architecture of part segmentation task.

    input_channels: point coordinate, maybe with normals
    m: the kinds of scene semantic segmentation
    input_transform: T-Net for input points
    feature_transform: T-Net for intermediate features
    """
    def __init__(self, input_channels=3, num_part=2, categories=16, input_transform=True, feature_transform=True):
        super(PartSegmentationPointNet, self).__init__()
        self.input_channels = input_channels
        self.num_part = num_part
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.t_net1 = TNet(self.input_channels)
        if self.feature_transform:
            self.t_net2 = TNet(64)

        self.shared_mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.shared_mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.shared_mlp3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.shared_mlp4 = nn.Conv1d(128, self.num_part, 1)

    def forward(self, x):
        batch_size, _, n = x.size()

        # T-Net transform 1
        transformer1 = None
        if self.input_transform:
            transformer1 = self.t_net1(x)
            x = x.permute(0, 2, 1)
            x = torch.bmm(x, transformer1)
            x = x.permute(0, 2, 1)

        features = self.shared_mlp1(x)

        # T-Net transform 2
        transformer2 = None
        if self.feature_transform:
            transformer2 = self.t_net2(features)
            features = features.permute(0, 2, 1)
            features = torch.bmm(features, transformer2)
            features = features.permute(0, 2, 1)

        x = self.shared_mlp2(features)


        global_feature = torch.max(x, dim=-1, keepdim=True)[0]
        global_features = global_feature.view(-1, 1024, 1).repeat(1, 1, n)

        # concatenate features and global features
        x = torch.cat([features, global_features], dim=1)

        x = self.shared_mlp3(x)
        x = self.shared_mlp4(x)

        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.num_part), dim=-1)
        x = x.view(batch_size, n, self.num_part)

        return x, transformer1, transformer2


class ModifiedPartSegmentationNetwork(nn.Module):
    """
    The architecture of the part segmentation network in the supplementary materials.

    input_channels: point coordinate, maybe with normals
    num_part: the number of parts in total categories
    categories: the kinds of object in dataset
    """
    def __init__(self, input_channels=3, num_part=2, categories=16):
        super(ModifiedPartSegmentationNetwork, self).__init__()

        self.input_channels = input_channels
        self.num_part = num_part
        self.categories = categories

        self.t_net1 = TNet(k=self.input_channels)
        self.t_net2 = TNet(k=128)

        self.conv1 = nn.Conv1d(input_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 2048, 1)
        self.bn5 = nn.BatchNorm1d(2048)

        self.fc1 = nn.Conv1d(3024, 256, 1)
        self.bn6 = nn.BatchNorm1d(256)
        self.fc2 = nn.Conv1d(256, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.fc3 = nn.Conv1d(256, 128, 1)
        self.bn8 = nn.BatchNorm1d(128)
        self.fc4 = nn.Conv1d(128, self.num_part, 1)

    def forward(self, x, y):
        """
        :param x: data with size (B, C, N)
        :param y: labels with size of (B,)
        :return:
        """
        batch_size, _, n = x.size()

        # T-Net transform for input
        transformer1 = self.t_net1(x)
        x = x.permute(0, 2, 1)
        x = torch.bmm(x, transformer1)
        x = x.permute(0, 2, 1)

        # shared mlp(64, 128, 128)
        x1 = F.relu(self.bn1(self.conv1(x)))  # 64
        x2 = F.relu(self.bn2(self.conv2(x1)))  # 128
        x3 = F.relu(self.bn3(self.conv3(x2)))  # 128

        # T-Net transform for feature
        transformer2 = self.t_net2(x3)
        x = x3.permute(0, 2, 1)
        x = torch.bmm(x, transformer2)
        x4 = x.permute(0, 2, 1)  # 128

        # shared mlp(512, 2048)
        x5 = F.relu(self.bn4(self.conv4(x4)))  # 512
        x = F.relu(self.bn5(self.conv5(x5)))

        # max symmetric function
        x6 = torch.max(x, dim=-1)[0]  # 2048
        x6 = torch.cat([x6, to_one_hots(y, self.categories)], dim=1)
        x6 = x6.view(batch_size, -1, 1).repeat(1, 1, n)  # 2048 + 16

        # concatenate skip links
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)

        # fully-connected layers
        x = F.relu(self.bn6(self.fc1(x)))
        x = F.relu(self.bn7(self.fc2(x)))
        x = F.relu(self.bn8(self.fc3(x)))

        x = self.fc4(x).transpose(1, 2).contiguous()
        x = F.log_softmax(x.view(-1, self.num_part), dim=1)
        x = x.view(batch_size, n, self.num_part)

        return x, transformer1, transformer2


class SceneSemanticSegmentationPointNet(nn.Module):
    """
    The architecture of part segmentation task or scene semantic segmentation task.

    input_channels: point coordinate, maybe with normals
    m: the kinds of scene semantic segmentation
    input_transform: T-Net for input points
    feature_transform: T-Net for intermediate features
    """
    def __init__(self, input_channels=9, m=13, input_transform=True, feature_transform=True):
        super(SceneSemanticSegmentationPointNet, self).__init__()
        self.input_channels = input_channels
        self.m = m
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.t_net1 = TNet(self.input_channels)
        if self.feature_transform:
            self.t_net2 = TNet(64)

        self.shared_mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.shared_mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.shared_mlp3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.shared_mlp4 = nn.Conv1d(128, m, 1)

    def forward(self, x):
        batch_size, _, n = x.size()

        # T-Net transform 1
        transformer1 = None
        if self.input_transform:
            transformer1 = self.t_net1(x)
            x = x.permute(0, 2, 1)
            x = torch.bmm(x, transformer1)
            x = x.permute(0, 2, 1)

        features = self.shared_mlp1(x)

        # T-Net transform 2
        transformer2 = None
        if self.feature_transform:
            transformer2 = self.t_net2(features)
            features = features.permute(0, 2, 1)
            features = torch.bmm(features, transformer2)
            features = features.permute(0, 2, 1)

        x = self.shared_mlp2(features)

        global_features = torch.max(x, dim=-1, keepdim=True)[0].view(-1, 1024, 1).repeat(1, 1, n)

        # concatenate features and global features
        x = torch.cat([features, global_features], dim=1)

        x = self.shared_mlp3(x)
        x = self.shared_mlp4(x)

        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.m), 1)
        x = x.view(batch_size, n, self.m)

        return x, transformer1, transformer2  # (B, n, m) and (B, 64, 64)


if __name__ == '__main__':
    """
    testing for four classes above
    """
    x = torch.rand((10, 3, 1024))
    cls_net = ClassificationPointNet(40)
    result1 = cls_net(x)
    print(result1[0].size(), result1[1].size(), result1[2].size())

    part_seg_net = PartSegmentationPointNet()
    result2 = part_seg_net(x)
    print(result2[0].size(), result2[1].size(), result2[2].size())

    y = torch.randint(0, 2, [10]).clone().detach().type_as(torch.tensor(0.1))
    modified_part_seg_net = ModifiedPartSegmentationNetwork()
    result3 = modified_part_seg_net(x, y)
    print(result3[0].size(), result3[1].size(), result3[2].size())

    x4 = torch.rand((10, 9, 1024))
    sss_net = SceneSemanticSegmentationPointNet()
    result4 = sss_net(x4)
    print(result4[0].size(), result4[1].size(), result4[2].size())
