from train import train, time_stamp
import os
import torch
import argparse
import torchvision.transforms as transforms
from PIL import Image
from plotter import get_latest_log, valid_path
from data import get_duration_wav, classes, input_size
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from model import Net_eval, backbone_network


def embed(audio_path):
    img_path = audio_path[:-4] + '_' + time_stamp() + '.png'
    dur = get_duration_wav(audio_path)
    y, sr = librosa.load(audio_path, offset=0.0, duration=dur)
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect)
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=-0.1)
    plt.close()

    return embed_img(img_path)


def embed_img(img_path, rm_cache=True):
    transform = transforms.Compose([
        transforms.Resize(input_size),
        # transforms.CenterCrop(300),
        # transforms.RandomAffine(5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(img_path).convert("RGB")

    if rm_cache and os.path.exists(img_path):
        os.remove(img_path)

    return transform(img)


def eval(log_dir='./logs', history=''):

    tag = args.target

    if not os.path.exists(tag):
        print('Target not found.')
        exit()

    if valid_path(log_dir, history):
        history = '/' + history
    else:
        history = get_latest_log(log_dir)

    saved_model_path = log_dir + history + '/save.pt'

    if not os.path.exists(saved_model_path):
        print('No history found, start a new term of training...')
        train()

    model = Net_eval(saved_model_path, backbone_network)
    input = embed(tag).unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.cuda()
        input = input.cuda()

    with torch.no_grad():
        output = model(input)
        print(output)
        pred_id = torch.max(output.data, 1)[1]
        predict = classes[pred_id]
        print(predict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict')
    parser.add_argument('--target', type=str,
                        default='./test/KAWAI-E1.wav', help='Select wav to be predicted.')
    parser.add_argument('--log', type=str,
                        default='', help='Select a training history.')
    args = parser.parse_args()
    # eval(latest_log=args.log)
    eval()
