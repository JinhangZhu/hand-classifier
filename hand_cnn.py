import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data  # Batch training
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from utils import imgtensor_from_array, imgtensor_from_file

# classes = ('left', 'right')


class CustomDataset(Data.Dataset):
    def __init__(self, folder_path, transform=None):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        Args:
            folder_path (string): path to image folder
        Reference:
            https://github.com/utkuozbulak/pytorch-custom-dataset-examples/blob/master/src/custom_dataset_from_file.py
        """
        # Get image list
        self.image_list = glob.glob(folder_path+'/*.jpg')
        # Calculate len
        self.data_len = len(self.image_list)
        self.transform = transform

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_path = self.image_list[index]
        # Open image
        im_as_im = Image.open(single_image_path)

        if self.transform is not None:
            im_as_im = self.transform(im_as_im)

        im_as_np = np.asarray(im_as_im)/255
        # print(im_as_np.shape)

        # Transform image to tensor, change data type
        im_as_ten = torch.from_numpy(im_as_np).float()
        im_as_ten = im_as_ten.permute(2, 0, 1)

        # Get label(class) of the image based on the file name
        class_indicator_location = single_image_path.rfind('_c')
        label = int(single_image_path[class_indicator_location+2:class_indicator_location+3])

        return (im_as_ten, label)

    def __len__(self):
        return self.data_len


class HandCropCNN(nn.Module):
    def __init__(self):
        super(HandCropCNN, self).__init__()
        self.conv1 = nn.Sequential(  # -> (3,28,28)
            nn.Conv2d(3, 32, 3, 1),
            nn.PReLU(32),
            nn.MaxPool2d(2, 2, padding=1)
        )
        self.conv2 = nn.Sequential(  # -> (32,14,14)
            nn.Conv2d(32, 48, 3, 1),
            nn.BatchNorm2d(48),
            nn.PReLU(48),
            nn.MaxPool2d(2, 2, padding=1)
        )
        self.conv3 = nn.Sequential(  # -> (48,7,7)
            nn.Conv2d(48, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(64)  # ,  # -> (64,5,5)
        )
        self.fc4 = nn.Linear(64*5*5, 128)
        self.fc5 = nn.Linear(128, 2)  # Flattened -> 2 classes
        self.sm6 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)   # (batch,64,7,7)
        x = self.conv3(x)   # (batch,64,7,7)
        x = x.view(x.size(0), -1)   # (batch,64*7*7)
        x = self.fc4(x)
        x = self.fc5(x)

        output = self.sm6(x)

        return output

    def train(self, device, opt):
        # Hyperparameters
        n_epochs = opt.epochs
        lr = opt.lr
        batch_size = opt.batch_size
        weight_file = opt.weight
        save_txt = opt.save_txt

        if opt.augment:
            # Data augmentation
            # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
            # https://github.com/utkuozbulak/pytorch-custom-dataset-examples#using-torchvision-transforms
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(20),
                    transforms.RandomAffine(
                        degrees=(30),
                        translate=(0.1, 0.2)
                    ),
                    transforms.Resize(28)
                ]
            )

        # Dataloader
        train_set = CustomDataset(opt.dataset + os.sep + 'train', transform=transform)
        train_loader = Data.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=2
        )

        # Save training process
        loss_his = []
        acc_his = []
        if not os.path.isdir(opt.save_txt):
            os.makedirs(opt.save_txt)

        # Training configurations
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        start_epoch = 0

        if not os.path.isfile(weight_file):
            loss_func = nn.CrossEntropyLoss()
            if os.path.isfile(opt.save_txt + os.sep + 'loss_results.txt'):
                os.remove(opt.save_txt + os.sep + 'loss_results.txt')
                os.remove(opt.save_txt + os.sep + 'accuracy_results.txt')
        else:
            checkpoint = torch.load(weight_file)
            self.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            loss_func = checkpoint['loss']

        print(('\n' + '%10s %15s %15s') % ('Epoch', 'Train loss', 'Test accuracy'))

        start_time = time.time()
        for epoch in range(start_epoch, n_epochs):
            epoch_start_time = time.time()
            nb = len(train_loader)
            for step, (x, y) in tqdm(enumerate(train_loader), total=nb):  # Extract tensors
                x, y = x.to(device), y.to(device)
                output = self(x)
                loss = loss_func(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_his.append(loss.data)
                with open(save_txt + os.sep + 'loss_results.txt', 'a') as f:
                    f.write('%.4f\n' % loss.data)

            accuracy = self.test(device, opt)
            acc_his.append(accuracy)
            with open(save_txt + os.sep + 'accuracy_results.txt', 'a') as f:
                f.write('%.4f\n' % accuracy)

            print(('\n%10s %15.4g %15.4g') % (epoch, loss.data, accuracy))

            # Save weight file
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_func
                }, weight_file
            )
            epoch_duration = time.time() - epoch_start_time
            print('\nEpoch lasts: %s' % (time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

        duration = time.time() - start_time
        print('\nTraining duration: %s' % (time.strftime('%H:%M:%S', time.gmtime(duration))))

        return (loss_his, acc_his)

    def test(self, device, opt):
        batch_size = opt.batch_size

        if opt.mode == 'test':    # Evaluation on test set
            weight_file = opt.weight
            checkpoint = torch.load(weight_file)
            self.load_state_dict(checkpoint['model_state_dict'])

            test_set = CustomDataset(opt.dataset + os.sep + 'test')
            data_loader = Data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=True,
                # num_workers=2
            )
        else:   # Valid set
            valid_set = CustomDataset(opt.dataset + os.sep + 'validation')
            data_loader = Data.DataLoader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=True,
                # num_workers=2
            )

        correct = 0
        total = 0
        with torch.no_grad():
            for (x, y) in data_loader:
                x, y = x.to(device), y.to(device)
                test_output = self(x)
                pred_y = torch.max(test_output, 1)[1].squeeze()
                total += y.size(0)
                correct += sum(pred_y == y).item()

        accuracy = correct / total
        if opt.mode == 'test':
            print('Evaluated accuracy: %.4f' % accuracy)
            if opt.save_txt:
                with open(save_txt + os.sep + 'evaluation.txt', 'w') as f:
                    f.write('%.4f' % accuracy)

        return accuracy

    def detect(self, device, weight_file, imgs, classes):
        checkpoint = torch.load(weight_file)
        self.load_state_dict(checkpoint['model_state_dict'])

        if isinstance(imgs, list):
            img_tensor = imgtensor_from_file(imgs).to(device)
        else:   # tuple
            img_tensor = imgtensor_from_array(imgs).to(device)

        start_time = time.time()
        with torch.no_grad():
            out = self(img_tensor)
            pred = torch.max(out, 1)[1].squeeze()

            if isinstance(imgs, list):
                print('%-40s%-15s%15s:%-15s' %
                      ('Image', 'Prediction', classes[0], classes[1]))
                for i in range(len(imgs)):
                    print('%-40s%-15s%15.4f:%-15.4f' %
                          (imgs[i], classes[pred[i].item()], out[i, 0].item(), out[i, 1].item()))
            else:
                print('%-15s%15s:%-15s' %
                      ('Prediction', classes[0], classes[1]))
                for i in range(len(imgs)):
                    print('%-15s%15.4f:%-15.4f' %
                          (classes[pred[i].item()], out[i, 0].item(), out[i, 1].item()))

        duration = time.time() - start_time
        print('Mean classification time: %.2gs' % (duration / len(imgs)))

        return out, pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='Runing mode: train, test, detect')
    parser.add_argument('--dataset', type=str, default='datasets/handcrops', help='Path to dataset')
    parser.add_argument('--save-txt', type=str, default='', help='Path to results')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--augment', action='store_true', help='Data augmentation')
    parser.add_argument('--weight', type=str, default='weights/handcnn.pt', help='Weight file to load.')
    parser.add_argument('--source', default='samples', help='Source images for detection.')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available():
        print('Using CUDA')
        device = torch.device('cuda:0')
    else:
        print('Using CPU')
        device = 'cpu'

    # Configure modes
    if opt.mode == 'train':
        print("Training...")
        cnn = HandCropCNN()
        if device != 'cpu':
            cnn.cuda()
        cnn.train(device, opt)

    elif opt.mode == 'test':
        print("Evaluating...")
        cnn = HandCropCNN()
        if device != 'cpu':
            cnn.cuda()
        cnn.test(device, opt)

    else:
        print("Detecting...")
        cnn = HandCropCNN()
        if device != 'cpu':
            cnn.cuda()
        imgs = opt.source
        if isinstance(imgs, str):
            imgs = sorted(glob.glob(imgs + os.sep + '*.jpg'))
        weight_file = opt.weight
        with open(opt.dataset + os.sep + 'classes.names', 'r') as f:
            classes = [line.rstrip() for line in f]
        cnn.detect(device, weight_file, imgs, classes)
