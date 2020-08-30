# Hand classifier

This repo contains how we develop a 5-layer convolutional neural network in PyTorch. The code works with CPU and GPU. 

## Our approaches

Our CNN architecture is based on the R-Net in MTCNN[1].

![](https://i.loli.net/2020/08/27/cbMxJI7q3YiOesQ.png)

Our dataset is based on [EPIC-KITCHENS-100](https://github.com/epic-kitchens/epic-kitchens-100-annotations) [2], an egocentric video dataset. In the image below are four examples from our dataset, where `c0`, `c1` mean `left_hand`, `right_hand`.

![](https://i.loli.net/2020/08/27/ByqMCiG7OJfeITD.png)

We also apply PyTorch transforms to enable Data Augmentation in training. The transforms we utilise are as follows. Kind reminder: **Do NOT enable flip in the training due to the reflectional symmetry**.

```python
transforms.RandomCrop(20),
transforms.RandomAffine(degrees=(30),translate=(0.1, 0.2)),
transforms.Resize(28)
```

## Requirements

Python 3.7 or later with all `requirements.txt` dependencies installed, including `torch >= 1.6`. To install run:

```bash
$ pip install -U -r requirements.txt
```

## Tutorials

- [Google Colab Notebook](https://colab.research.google.com/github/JinhangZhu/hand-classifier/blob/master/quick_start.ipynb) <a href="https://colab.research.google.com/github/JinhangZhu/hand-classifier/blob/master/quick_start.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```bash
$ python .\hand_cnn.py -h                                 
usage: hand_cnn.py [-h] [--mode MODE] [--dataset DATASET]
                   [--save-txt SAVE_TXT] [--batch-size BATCH_SIZE]
                   [--epochs EPOCHS] [--lr LR] [--augment] [--weight WEIGHT]
                   [--source SOURCE]
```

### Training

```bash
optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           Runing mode: train, test, detect
  --dataset DATASET     Path to dataset
  --save-txt SAVE_TXT   Path to results
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --lr LR               Learning rate
  --augment             Data augmentation
  --weight WEIGHT       Weight file to load.
```

Training command example command.

```bash
python hand_cnn.py --mode train --dataset datasets/handcrops --save-txt results --batch-size 100 --epochs 3 --lr 0.0001 --augment
```

Resuming training example command. We require to set the number of epochs a larger value than one saved in the checkpoint.

```bash
python hand_cnn.py --mode train --dataset datasets/handcrops --weight weights/handcnn.pt --save-txt results --batch-size 100 --epochs 10 --lr 0.0001 --augment
```

Plot training process command:

```bash
from utils import plot_process
plot_process('results')
```

### Evaluation

```bash
optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           Runing mode: train, test, detect
  --dataset DATASET     Path to dataset
  --save-txt SAVE_TXT   Path to results
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --lr LR               Learning rate
  --augment             Data augmentation
  --weight WEIGHT       Weight file to load.
  --source SOURCE       Source images for detection.
```

Evaluation on test set example with trained weight example command.

```bash
python hand_cnn.py --mode test --batch-size 100 --weight weights/handcnn.pt
```

### Detection

```bash
optional arguments:
  -h, --help            show this help message and exit
  --mode MODE           Runing mode: train, test, detect
  --dataset DATASET     Path to dataset
  --save-txt SAVE_TXT   Path to results
  --batch-size BATCH_SIZE
  --epochs EPOCHS
  --lr LR               Learning rate
  --augment             Data augmentation
  --weight WEIGHT       Weight file to load.
  --source SOURCE       Source images for detection.
```

Detection of samples example command:

```bash
python hand_cnn.py --mode detect --weight weights/handcnn.pt --source samples
```

### Easy integration

Our implementation should be easy to be integrated into your codes.

```python
model = HandCropCNN()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
weight_file = 'weights/handcnn.pt'

patches = []
imgs = sorted(glob.glob('samples/' + '*jpg'))
for im_path in imgs:
    img = cv2.imread(im_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    patches.append(img)
patches = tuple(patches)

with open('datasets/handcrops/classes.names', 'r') as f:
    classes = [line.rstrip() for line in f]

cls_conf, labels = model.detect(device, weight_file, patches, classes)
```

## References

1. Zhang, K. et al. (2016) ‘Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks’, IEEE Signal Processing Letters. Institute of Electrical and Electronics Engineers Inc., 23(10), pp. 1499–1503. doi: 10.1109/LSP.2016.2603342.
2. Damen, D. et al. (2020) ‘The EPIC-KITCHENS Dataset: Collection, Challenges and Baselines’, IEEE Transactions on Pattern Analysis and Machine Intelligence. Institute of Electrical and Electronics Engineers (IEEE), pp. 1–1. Available at: http://arxiv.org/abs/2005.00343 (Accessed: 16 August 2020).