import csv
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from datetime import datetime
from model import Net
from focalLoss import FocalLoss
from data import prepare_data, classes
from utils import time_stamp, create_dir, toCUDA
from plot import save_acc, save_loss, save_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def eval_model_train(model, trainLoader, tra_acc_list):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in trainLoader:
            inputs, labels = data[0], toCUDA(data[1])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print('Training acc   : ' + str(round(acc, 2)) + '%')
    tra_acc_list.append(acc)


def eval_model_valid(model, validationLoader, val_acc_list):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in validationLoader:
            inputs, labels = data[0], toCUDA(data[1])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print('Validation acc : ' + str(round(acc, 2)) + '%')
    val_acc_list.append(acc)


def eval_model_test(model, testLoader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = data[0], toCUDA(data[1])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    report = classification_report(
        y_true, y_pred, target_names=classes, digits=3)
    cm = confusion_matrix(y_true, y_pred, normalize='all')

    return report, cm


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
    log_dir = './logs/' + args.model + '__' + time_stamp()
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


def train(backbone_ver='alexnet', epoch_num=40, iteration=10, lr=0.001):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tra_acc_list, val_acc_list, loss_list, lr_list = [], [], [], []

    # init model
    model = Net(m_ver=backbone_ver, deep_finetune=args.deepfinetune)

    # load data
    trainLoader, validLoader, testLoader = prepare_data(
        batch_size=4, input_size=model.input_size)

    #optimizer and loss
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=lr, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        criterion = criterion.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # train process
    start_time = datetime.now()
    print('Start training [' + args.model + '] at ' + time_stamp(start_time))
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        epoch_str = f' Epoch {epoch + 1}/{epoch_num} '
        lr_str = optimizer.param_groups[0]["lr"]
        lr_list.append(lr_str)
        print(f'{epoch_str:-^40s}')
        print(f'Learning rate: {lr_str}')
        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            # get the inputs
            inputs, labels = data[0], toCUDA(data[1])
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print every 2000 mini-batches
            if i % iteration == iteration - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / iteration))
                loss_list.append(running_loss / iteration)
            running_loss = 0.0

        eval_model_train(model, trainLoader, tra_acc_list)
        eval_model_valid(model, validLoader, val_acc_list)
        scheduler.step(loss.item())

    finish_time = datetime.now()
    cls_report, cm = eval_model_test(model, testLoader)
    save_history(model, tra_acc_list, val_acc_list, loss_list,
                 lr_list, cls_report, cm, start_time, finish_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--model', type=str, default='alexnet')
    parser.add_argument('--deepfinetune', type=bool, default=False)
    args = parser.parse_args()

    train(backbone_ver=args.model, epoch_num=40)
