# Piano-Classification

Classify piano sound quality

<!-- [![license](https://img.shields.io/github/license/george-chou/Piano-Classification.svg)](https://github.com/george-chou/Piano-Classification/blob/master/LICENSE)
[![Python application](https://github.com/george-chou/Piano-Classification/workflows/Python%20application/badge.svg)](https://github.com/george-chou/Piano-Classification/actions)
[![Github All Releases](https://img.shields.io/github/downloads-pre/george-chou/Piano-Classification/v1.2/total)](https://github.com/george-chou/Piano-Classification/releases) -->

## Dataset

Download at <https://github.com/george-chou/Piano-Classification/releases/download/v0.1/audio.zip>

Extract it into the project directory

## Usage

### Download

```
git clone -b vgg https://github.com/george-chou/Piano-Classification.git
cd Piano-Classification
```

### Train

```
python train.py
```

### Plot results

```
python plotter.py
```

### Predict

```
python evaluate.py --target ./test/KAWAI.wav
```

## Results

| <img src="./results/loss.png"/> |  <img src="./results/acc.png"/>  |
| :-----------------------------: | :------------------------------: |
|           Loss curve            | Training and validation accuracy |
