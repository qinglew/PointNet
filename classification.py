import math

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data

from pointnet import ClassificationPointNet
from dataset import ModelNet40
from utils import feature_transform_regularization


EPOCHS = 200  # training epochs
N = 1024  # number of points
BATCH_SIZE = 32  # the number of samples in a batch
NUM_WORKERS = 4
INPUT_TRANSFORM = True  # whether use T-Net for input point clouds
FEATURE_TRANSFORM = True  # whether use T-Net for features of point clouds


def train_and_eval():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    train_dataset = ModelNet40(n=N, split='train')
    test_dataset = ModelNet40(n=N, split='test')
    train_data_loader = Data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_data_loader = Data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Network Model
    categories = len(train_dataset.code2category)
    model = ClassificationPointNet(categories, input_transform=INPUT_TRANSFORM, feature_transform=FEATURE_TRANSFORM)
    model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_batch_num = math.ceil(len(train_dataset) / BATCH_SIZE)
    test_batch_num = math.ceil(len(test_dataset) / BATCH_SIZE)

    max_overall_accuracy, counterpart_avg_class_accuracy = 0.0, 0.0
    counterpart_overall_accuracy, max_avg_class_accuracy = 0.0, 0.0

    epoch1, epoch2 = 0, 0  # records the epoch of best performance for overall and avg. class accuracy

    for epoch in range(1, EPOCHS + 1):
        for i, (point_clouds, labels) in enumerate(train_data_loader):
            # a batch of data
            point_clouds = point_clouds.transpose(2, 1)
            labels = labels.view(-1)
            point_clouds, labels = point_clouds.to(device), labels.to(device)

            # forward-propagation
            optimizer.zero_grad()
            model.train()
            preds, transformer1, transformer2 = model(point_clouds)
            loss = F.nll_loss(preds, labels)
            if FEATURE_TRANSFORM:
                loss += feature_transform_regularization(transformer2) * 0.001

            # back-propagation and gradient descent
            loss.backward()
            optimizer.step()

            # metrics on train
            predictions = preds.data.max(1)[1]
            comparison = predictions.data.eq(labels).cpu()
            correct = torch.sum(comparison).item()

            if (i + 1) % 50 == 0:
                print("Training epoch {} iteration {}/{} ==> Loss: {}, Accuracy: {}".format(epoch,
                                                                                            i + 1,
                                                                                            train_batch_num,
                                                                                            loss.item(),
                                                                                            correct / len(labels)))

        scheduler.step()

        test_total_correct = 0
        correct_nums_per_cat = np.zeros((categories,))
        with torch.no_grad():
            for i, (data, target) in enumerate(test_data_loader):
                data = data.transpose(2, 1)
                target = target.view(-1)
                data, target = data.to(device), target.to(device)
                model.eval()
                preds, _, _ = model(data)
                predictions = preds.data.max(1)[1]
                comparison = predictions.data.eq(target).cpu()
                correct = torch.sum(comparison).item()
                test_total_correct += correct
                for index, cat in enumerate(target):
                    correct_nums_per_cat[cat] += comparison[index]

        overall_accuracy = test_total_correct / len(test_dataset)
        avg_class_accuracy = np.mean(correct_nums_per_cat / np.array(test_dataset.categories_nums))
        print("Evaluation on testing dataset ==> Overall Accuracy: {}, "
              "Avg. class Accuracy: {}".format(overall_accuracy, avg_class_accuracy))

        if overall_accuracy > max_overall_accuracy:
            max_overall_accuracy = overall_accuracy
            counterpart_avg_class_accuracy = avg_class_accuracy
            torch.save(model.state_dict(), "max_overall_accuracy_epoch_{}.pth".format(epoch))
            epoch1 = epoch

        if avg_class_accuracy > max_avg_class_accuracy:
            max_avg_class_accuracy = avg_class_accuracy
            counterpart_overall_accuracy = overall_accuracy
            torch.save(model.state_dict(), "max_avg_accuracy_epoch_{}.pth".format(epoch))
            epoch2 = epoch

    print("Max overall accuracy {} in epoch {}, and the avg class accuracy is {}".format(max_overall_accuracy,
                                                                                         epoch1,
                                                                                         counterpart_avg_class_accuracy))
    print("Max avg accuracy {} in epoch {}, and the overall accuracy is {}".format(max_avg_class_accuracy,
                                                                                   epoch2,
                                                                                   counterpart_overall_accuracy))


if __name__ == '__main__':
    train_and_eval()
