# References:
# https://github.com/ultralytics/yolov3/blob/master/utils/torch_utils.py

import os
# import time
import torch
import cv2
import matplotlib.pyplot as plt
import glob


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 480, 640),), verbose=False)
        fs = ', %.1f GFLOPS' % (macs / 1E9 * 2)
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def imgtensor_from_array(imgs):
    """
    imgs: (img, img, img...)
    """
    cat = 0
    for im in imgs:
        dsize = (28, 28)
        im = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)
        im_as_np = im/255
        im_as_ten = torch.from_numpy(im_as_np).float()
        im_as_ten = im_as_ten.permute(2, 0, 1).unsqueeze(0)
        if not cat:
            img_tensor = im_as_ten
            cat = 1
        else:
            img_tensor = torch.cat((img_tensor, im_as_ten), 0)
    return img_tensor


def imgtensor_from_file(im_path):
    cat = 0
    for ip in im_path:
        im = cv2.imread(ip)
        dsize = (28, 28)
        im = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)
        im_as_np = im/255
        im_as_ten = torch.from_numpy(im_as_np).float()
        im_as_ten = im_as_ten.permute(2, 0, 1).unsqueeze(0)
        if not cat:
            img_tensor = im_as_ten
            cat = 1
        else:
            img_tensor = torch.cat((img_tensor, im_as_ten), 0)
    return img_tensor


def plot_process(save_path):
    loss_path = save_path + os.sep + 'loss_results.txt'
    acc_path = save_path + os.sep + 'accuracy_results.txt'
    with open(loss_path, 'r') as f:
        loss_his = [float(i) for i in f.readlines()]
    with open(acc_path, 'r') as f:
        acc_his = [float(i) for i in f.readlines()]

    plt.figure(dpi=300)
    plt.title('Training Process')
    plt.subplot(121)
    plt.plot(loss_his)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    # plt.ylim((0, 3))

    plt.subplot(122)
    plt.plot(acc_his)
    plt.xlabel('per 50 steps')
    plt.ylabel('Accuracy')
    # plt.ylim((0, 1))

    plt.savefig(save_path + os.sep + 'Training process.png')
    plt.show()


def plot_samples(samples_path):
    imgs = sorted(glob.glob(samples_path + '/' + '*jpg'))
    n = len(imgs)
    plt.figure(dpi=300)
    for i, im_path in enumerate(imgs):
        img = cv2.imread(im_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
