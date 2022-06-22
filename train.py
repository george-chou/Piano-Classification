
import csv
from string import digits
import time
import torch
import torch.nn as nn
from datetime import datetime
import torch.utils.data
import torch.optim as optim
from model import Net
from data import prepare_data, create_dir, classes
from focaloss import FocalLoss
from plotter import save_acc, save_loss, save_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

backbone_network = 'vgg19_bn'


def eval_model_train(model, trainLoader, device, tra_acc_list):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in trainLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print('Accuracy of training    : ' + str(round(acc, 2)) + '%')
    tra_acc_list.append(acc)


def eval_model_valid(model, validationLoader, device, val_acc_list):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in validationLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print('Accuracy of validation  : ' + str(round(acc, 2)) + '%')
    val_acc_list.append(acc)


def eval_model_test(model, testLoader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    report = classification_report(
        y_true, y_pred, target_names=classes, digits=3)
    cm = confusion_matrix(y_true, y_pred, normalize='all')

    return report, cm


def time_stamp(timestamp=None):
    if timestamp != None:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))


def save_log(start_time, finish_time, cls_report, cm, log_dir):

    log_backbone = 'Backbone     : ' + args.model
    log_start_time = 'Start time   : ' + time_stamp(start_time)
    log_finish_time = 'Finish time  : ' + time_stamp(finish_time)
    log_time_cost = 'Time cost    : ' + \
        str((finish_time - start_time).seconds) + 's'

    with open(log_dir + '/result.log', 'w', encoding='utf-8') as f:
        f.write(cls_report + '\n')
        f.write(log_backbone + '\n')
        f.write(log_start_time + '\n')
        f.write(log_finish_time + '\n')
        f.write(log_time_cost)
    f.close()

    # save confusion_matrix
    np.savetxt(log_dir + '/mat.csv', cm, delimiter=',')
    save_confusion_matrix(cm, log_dir)

    print(cls_report)
    print('Confusion matrix :')
    print(str(cm.round(3)) + '\n')
    print(log_backbone)
    print(log_start_time)
    print(log_finish_time)
    print(log_time_cost)


def save_history(model, tra_acc_list, val_acc_list, loss_list, lr_list, cls_report, cm, start_time, finish_time):
    create_dir('./logs')
    log_dir = './logs/history_' + time_stamp()
    create_dir(log_dir)

    acc_len = len(tra_acc_list)
    with open(log_dir + "/acc.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tra_acc_list", "val_acc_list", "lr_list"])
        for i in range(acc_len):
            writer.writerow([tra_acc_list[i], val_acc_list[i], lr_list[i]])

    loss_len = len(loss_list)
    with open(log_dir + "/loss.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["loss_list"])
        for i in range(loss_len):
            writer.writerow([loss_list[i]])

    torch.save(model.state_dict(), log_dir + '/save.pt')
    print('Model saved.')

    save_acc(tra_acc_list, val_acc_list, log_dir)
    save_loss(loss_list, log_dir)
    save_log(start_time, finish_time, cls_report, cm, log_dir)


def train(epoch_num=40, iteration=10, lr=0.001):

    tra_acc_list, val_acc_list, loss_list, lr_list = [], [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    trainLoader, validLoader, testLoader = prepare_data()

    # init model
    model = Net(backbone=args.model)
    # print(model)

    #optimizer and loss
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(class_num=len(classes))
    optimizer = optim.SGD(model.classifier.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=lr, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = model.to(device)
    criterion = criterion.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # train process
    start_time = datetime.now()
    print('Start training at ' + time_stamp(start_time))
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        epoch_str = f' Epoch {epoch + 1}/{epoch_num} '
        lr_str = optimizer.param_groups[0]["lr"]
        lr_list.append(lr_str)
        print(f'{epoch_str:-^40s}')
        print(f'Learning rate: {lr_str}')
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            # get the inputs
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % iteration == iteration - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / iteration))
                loss_list.append(running_loss / iteration)
            running_loss = 0.0

        eval_model_train(model, trainLoader, device, tra_acc_list)
        eval_model_valid(model, validLoader, device, val_acc_list)
        scheduler.step(loss.item())

    finish_time = datetime.now()
    cls_report, cm = eval_model_test(model, testLoader, device)
    save_history(model, tra_acc_list, val_acc_list, loss_list,
                 lr_list, cls_report, cm, start_time, finish_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--model', type=str, default=backbone_network,
                        help='Select a pre-trained model.')
    args = parser.parse_args()

    train(epoch_num=1)
