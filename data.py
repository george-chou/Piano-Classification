from math import floor
import os
import wave
import torch
import shutil
import random
import zipfile
import librosa
import contextlib
import librosa.display
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

audio_dir = './audio'
img_dir = './image'
data_dir = './dataset'
tra_dir = data_dir + '/tra'
val_dir = data_dir + '/val'
tes_dir = data_dir + '/tes'

input_size = [299, 299]


def load_cls():
    cls = []
    for _, dirnames, _ in os.walk(audio_dir):
        for dirname in dirnames:
            cls.append(dirname.split('_')[-1])

    return cls


classes = load_cls()


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def trans(audio_dir, img_dir):
    print('Pre-processing data...')
    create_dir(img_dir)

    if len(os.listdir(img_dir)) > 0:
        print('Data already pre-processed.')
        return

    for _, dirnames, _ in os.walk(audio_dir):
        for dirname in dirnames:
            trans_files(audio_dir + '/' + dirname, img_dir)

    print('Data pre-processed.')


def trans_files(cls_dir, img_dir):
    for _, _, filenames in os.walk(cls_dir):
        print('Converting ' + cls_dir.split('/')[-1] + '...')
        for filename in filenames:
            to_mel(cls_dir + '/' + filename, img_dir, width=0.2)


def get_duration_wav(audio_path):
    with contextlib.closing(wave.open(audio_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return round(duration, 1)   # One decimal place


def to_mel(audio_path, img_dir, width=1.0, step=0.2):
    dur = get_duration_wav(audio_path)
    audio_name = audio_path.split('/')[-1][:-4]
    print('Duration of audio ' + audio_name + ': ' + str(dur) + 's')
    for i in np.arange(0.0, dur - width + step, step):
        index = round(i, 1)
        outpath = img_dir + '/' + audio_name + '[' + str(index) + '].png'
        y, sr = librosa.load(audio_path, offset=index, duration=width)
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect)
        plt.axis('off')
        plt.savefig(outpath, bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def embedding(file_path):
    # dataset
    transform = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(300),
        # transforms.RandomAffine(5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    inputSet = ImageFolder(file_path,  transform=transform)
    return torch.utils.data.DataLoader(inputSet, batch_size=4, shuffle=True, num_workers=2, drop_last=True)


def load_data(img_dir, data_dir, force_reload=True):
    print('Loading data...')

    if os.path.exists(data_dir) and not force_reload:
        print('Data already loaded.')
        return

    if os.path.exists(data_dir) and force_reload:
        shutil.rmtree(data_dir)

    create_dir(data_dir)
    create_dir(tra_dir)
    create_dir(val_dir)
    create_dir(tes_dir)

    for _, _, filenames in os.walk(img_dir):

        length = len(filenames)
        p10 = floor(length / 10)
        p20 = 2 * p10

        val_test = random.sample(filenames, p20)
        trainset = list(set(filenames) - set(val_test))

        print('Copying validation data...')
        for i in range(p10):
            copy_img(val_test[i], val_dir)

        print('Copying test data...')
        for i in range(p10, p20):
            copy_img(val_test[i], tes_dir)

        print('Copying training data...')
        for filename in trainset:
            copy_img(filename, tra_dir)

    print('Data loaded.')


def copy_img(img_name, tag_dir):
    cls_id = int(img_name[0])
    cls = str(cls_id) + '_' + classes[cls_id - 1]
    outdir = tag_dir + '/' + cls
    create_dir(outdir)
    shutil.copy('./image/' + img_name, outdir)


def prepare_data():

    if(not os.path.exists(audio_dir)):
        unzip_file('./audio.zip', './')

    trans(audio_dir, img_dir)
    load_data(img_dir, data_dir, force_reload=False)

    print('Embedding data...')
    trainLoader = embedding(tra_dir)
    validLoader = embedding(val_dir)
    testLoader = embedding(tes_dir)
    print('Data embedded.')

    return trainLoader, validLoader, testLoader


if __name__ == "__main__":
    trans(audio_dir, img_dir)
    load_data(img_dir, data_dir)
