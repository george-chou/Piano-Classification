import os
import time
import torch
import zipfile
import requests
from tqdm import tqdm

audio_dir = './audio'
img_dir = './image'
data_dir = './dataset'
tra_dir = data_dir + '/tra'
val_dir = data_dir + '/val'
tes_dir = data_dir + '/tes'
backbone_list_path = './backbone.csv'

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


def create_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def time_stamp(timestamp=None):
    if timestamp != None:
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))


def toCUDA(x):
    if hasattr(x, 'cuda'):
        if torch.cuda.is_available():
            return x.cuda()

    return x
