import os
import torch
import requests
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from data import create_dir, classes

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth'
}

backbone_network = 'densenet201'


def url_download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def download_model(backbone=backbone_network):
    pre_model_url = model_urls[backbone]
    model_dir = './model/'
    pre_model_path = model_dir + (pre_model_url.split('/')[-1])

    create_dir(model_dir)

    if not os.path.exists(pre_model_path):
        url_download(pre_model_url, pre_model_path)

    return pre_model_path


def Classifier():
    return torch.nn.Sequential(
        # nn.Dropout(),
        # nn.Linear(256 * 6 * 6, 4096),
        # nn.ReLU(inplace=True),
        # nn.Dropout(),
        # nn.Linear(4096, 4096),
        # nn.ReLU(inplace=True),
        # nn.Dropout(),
        # nn.Linear(4096, 1000),
        # nn.ReLU(inplace=True),
        nn.Linear(in_features=1920, out_features=len(classes), bias=True)
    )


def Net(backbone=backbone_network):
    model = eval('models.%s()' % backbone)
    pre_model_path = download_model(backbone)
    checkpoint = torch.load(pre_model_path)
    model.load_state_dict(checkpoint, False)

    for parma in model.parameters():
        parma.requires_grad = False

    model.classifier = Classifier()
    model.train()

    return model


def Net_eval(saved_model_path, backbone):

    model = eval('models.%s()' % backbone)
    model.classifier = Classifier()
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint, False)
    model.eval()

    return model
