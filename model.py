import os
import torch
import requests
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
from data import create_dir, classes

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    "mnasnet0_5": "https://download.pytorch.org/models/mnasnet0.5_top1_67.823-3ffadce67e.pth",
    "mnasnet1_0": "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth"
}

backbone_network = 'mobilenet_v2'


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
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=1000, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(1000, 1000),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(1000, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, len(classes))
    )


def Net(backbone=backbone_network):
    model = eval('models.%s()' % backbone)
    pre_model_path = download_model(backbone)
    model.load_state_dict(torch.load(pre_model_path))

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
