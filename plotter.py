import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy import signal as ss
from data import create_dir, classes


def show_point(max_id, list):
    show_max = '(' + str(max_id + 1) + ', ' + str(round(list[max_id], 2)) + ')'
    plt.annotate(show_max, xytext=(
        max_id + 1, list[max_id]), xy=(max_id + 1, list[max_id]), fontsize=6)


def smooth(y):
    return ss.savgol_filter(y, 95, 3)


def plot_acc(tra_acc_list, val_acc_list):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    max1 = np.argmax(y1)
    max2 = np.argmax(y2)

    plt.title('Accuracy of training and validation')
    plt.xlabel('Epoch')
    plt.ylabel('Acc(%)')
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")
    plt.plot(1+max1, y1[max1], 'r-o')
    plt.plot(1+max2, y2[max2], 'r-o')
    show_point(max1, y1)
    show_point(max2, y2)
    plt.legend()


def save_acc(tra_acc_list, val_acc_list, save_path):
    x_acc = []
    for i in range(len(tra_acc_list)):
        x_acc.append(i + 1)

    x = np.array(x_acc)
    y1 = np.array(tra_acc_list)
    y2 = np.array(val_acc_list)
    max1 = np.argmax(y1)
    max2 = np.argmax(y2)

    plt.title('Accuracy of training and validation')
    plt.xlabel('Epoch')
    plt.ylabel('Acc(%)')
    plt.plot(x, y1, label="Training")
    plt.plot(x, y2, label="Validation")
    plt.plot(1+max1, y1[max1], 'r-o')
    plt.plot(1+max2, y2[max2], 'r-o')
    show_point(max1, y1)
    show_point(max2, y2)
    plt.legend()
    plt.savefig(save_path + "/acc.png")
    plt.close()


def save_loss(loss_list, save_path):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.title('Loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(x_loss, smooth(loss_list))
    plt.savefig(save_path + "/loss.png")
    plt.close()


def plot_loss(loss_list):
    x_loss = []
    for i in range(len(loss_list)):
        x_loss.append(i + 1)

    plt.title('Loss curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.plot(x_loss, smooth(loss_list))


def get_latest_log(path):
    if not os.path.exists(path):
        print('Please train a model first.')
        exit()

    lists = os.listdir(path)
    if len(lists) == 0:
        print('No history found in logs.')
        exit()

    lists = os.listdir(path)
    lists.sort(key=lambda x: os.path.getctime((path + "\\" + x)))
    return '/' + lists[-1]


def valid_path(log_path, latest_log):
    if latest_log == '':
        return False

    dir_path = log_path + '/' + latest_log
    return os.path.exists(dir_path)


def load_history(log_dir='./logs', latest_log=''):

    create_dir(log_dir)

    if len(os.listdir(log_dir)) == 0:
        print('Please finish training first.')
        exit()

    if valid_path(log_dir, latest_log):
        latest_log = '/' + latest_log
    else:
        latest_log = get_latest_log(log_dir)

    latest_acc = log_dir + latest_log + '/acc.csv'
    latest_loss = log_dir + latest_log + '/loss.csv'
    acc_list = pd.read_csv(latest_acc)
    tra_acc_list = acc_list['tra_acc_list'].tolist()
    val_acc_list = acc_list['val_acc_list'].tolist()
    loss_list = pd.read_csv(latest_loss)['loss_list'].tolist()

    cm = np.loadtxt(open(log_dir + latest_log + "/mat.csv", "rb"),
                    delimiter=",", skiprows=0)

    return tra_acc_list, val_acc_list, loss_list, cm


def plot_confusion_matrix(cm, title='Confusion matrix', labels_name=classes):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # Normalized
    # Display an image on a specific window
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)    # image caption
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    # print the labels on the x-axis coordinates
    plt.xticks(num_local, labels_name, rotation=90)
    # print the label on the y-axis coordinate
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def save_confusion_matrix(cm, save_path, title='Confusion matrix', labels_name=classes):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # Normalized
    # Display an image on a specific window
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)    # image caption
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    # print the labels on the x-axis coordinates
    plt.xticks(num_local, labels_name, rotation=90)
    # print the label on the y-axis coordinate
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path + '/mat.png')
    plt.close()


def plot_all(latest_log=''):
    tra_acc_list, val_acc_list, loss_list, cm = load_history(
        latest_log=latest_log)

    plt.figure(figsize=(9, 7))
    plt.subplot(221)
    plot_acc(tra_acc_list, val_acc_list)
    plt.subplot(222)
    plot_loss(loss_list)
    plt.subplot(224)
    plot_confusion_matrix(cm)
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()
    plt.show()  # Plot latest log


def save_all(latest_log):
    tra_acc_list, val_acc_list, loss_list, cm = load_history(
        latest_log=latest_log)
    save_acc(tra_acc_list, val_acc_list, './logs/' + latest_log)
    save_loss(loss_list, './logs/' + latest_log)
    save_confusion_matrix(cm, './logs/' + latest_log)
    print('Re-saved.')


if __name__ == "__main__":
    plot_all()
    # plot_all(latest_log='history_2022-06-16_06-20-26')
    # save_all('history_2022-06-20_20-55-57')
