import argparse
import os
import shutil
import time
import logging
import random
import pdb
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.optim
from tqdm import tqdm
cudnn.benchmark = True

import multicrop
import numpy as np

from medpy import metric
import nibabel as nib

import models
from models import criterions
from data import datasets
from data.data_utils import get_all_coords, _shape
from utils import Parser

path = os.path.dirname(__file__)


def calculate_metrics(pred, target):
    sens = metric.sensitivity(pred, target)
    spec = metric.specificity(pred, target)
    dice = metric.dc(pred, target)


eps = 1e-5


def f1_score(o, t):
    num = 2 * (o * t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num / den


# https://github.com/ellisdg/3DUnetCNN
# https://github.com/ellisdg/3DUnetCNN/blob/master/brats/evaluate.py
# https://github.com/MIC-DKFZ/BraTS2017/blob/master/utils_validation.py
def dice(output, target):
    ret = []
    # whole
    o = output > 0
    t = target > 0
    ret += (f1_score(o, t),)
    # core
    o = (output == 1) | (output == 4)
    t = (target == 1) | (target == 4)
    ret += (f1_score(o, t),)
    # active
    o = output == 4
    t = target == 4
    ret += (f1_score(o, t),)

    return ret


keys = "whole", "core", "enhancing", "loss"




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def nib_load(file_name):
    proxy = nib.load(file_name)
    data = proxy.get_data()
    proxy.uncache()
    return data


def testBraTS(scanpath, seg_path):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--cfg", default="deepmedic_ce_50_50_fold0", type=str)
    parser.add_argument("-gpu", "--gpu", default="", type=str)

    args = parser.parse_args()

    args = Parser(args.cfg, log="test").add_args(args)
    print(args)
    args.ckpts = os.path.join("ckpts", args.cfg)
    args.data_dir = "test"
    args.valid_list = os.path.join(args.data_dir, "test.txt")
    args.saving = True
    args.scoring = True

    args.ckpt = "model_last.tar"

    if args.saving:
        out_dir = "test/pred/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir
    else:
        args.out_dir = ""

    image = np.array(nib_load(scanpath), dtype="float32", order="C")

    mask = image > 0
    x = image[...]
    y = x[mask]
    lower = np.percentile(y, 0.2)
    upper = np.percentile(y, 99.8)
    x[mask & (x < lower)] = lower
    x[mask & (x > upper)] = upper
    y = x[mask]
    x -= y.mean()
    x /= y.std()
    image[...] = x
    image = np.expand_dims(image, axis=0)
    image = image.repeat(4, axis=0)
    image = torch.from_numpy(image)
    image = image.permute(0, 3, 1, 2).contiguous()

    print(image.shape)
    Network = getattr(models, args.net)
    model = Network(**args.net_params)

    model = model.to(device)
    ckpts = args.ckpts
    model_file = os.path.join(ckpts, args.ckpt)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    # print(model)
    # import pdb; pdb.set_trace()
    state_dict = {
        k: v for k, v in checkpoint["state_dict"].items() if k in model.state_dict()
    }
    model.load_state_dict(state_dict)
    print(args.dataset)

    H, W, T = 240, 240, 155
    target_size = args.target_size
    h, w, t = np.ceil(np.array(_shape, dtype="float32") / target_size).astype(
            "int"
        )
    h, w, t = int(h), int(w), int(t)
    sample_size = args.sample_size
    sub_sample_size = args.sub_sample_size
    dtype = torch.float32
    batch_size = args.batch_size
    model.eval()
    x = image.to(device)
    coords = get_all_coords(target_size)

    outputs = torch.zeros(
        (5, h * w * t, target_size, target_size, target_size), dtype=dtype
    )
    # targets = torch.zeros((h*w*t, 9, 9, 9), dtype=torch.uint8)

    for b, coord in tqdm(enumerate(coords.split(batch_size))):
        x1 = multicrop.crop3d_gpu(
            x, coord, sample_size, sample_size, sample_size, 1, True
        )
        x2 = multicrop.crop3d_gpu(
            x, coord, sub_sample_size, sub_sample_size, sub_sample_size, 3, True
        )

        # pdb.set_trace()
        # compute output
        logit = model((x1, x2))  # nx5x9x9x9, target nx9x9x9
        output = F.softmax(logit, dim=1)

        # copy output
        start = b * batch_size
        end = start + output.shape[0]
        outputs[:, start:end] = output.permute(1, 0, 2, 3, 4).cpu()

        # targets[start:end] = target.type(dtype).cpu()

    outputs = outputs.view(5, h, w, t, 9, 9, 9).permute(0, 1, 4, 2, 5, 3, 6)
    outputs = outputs.reshape(5, h * 9, w * 9, t * 9)
    outputs = outputs[:, :H, :W, :T].numpy()

    # targets = targets.view(h, w, t, 9, 9, 9).permute(0, 3, 1, 4, 2, 5).reshape(h*9, w*9, t*9)
    # targets = targets[:H, :W, :T].numpy()


    if out_dir:
        # np.save(os.path.join(out_dir, name + '_preds'), outputs)  # to save in .npy format
        preds = outputs.argmax(0).astype("uint8")
        img = nib.Nifti1Image(preds, None)
        nib.save(img, os.path.join(seg_path))


testBraTS('test/scans/Brats18_CBICA_AWG_1_t2.nii.gz', "test.nii.gz")