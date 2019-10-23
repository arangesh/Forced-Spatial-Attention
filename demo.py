import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import argparse
import numpy as np
import torchvision.transforms as transforms
import os
import time
import glob
from statistics import mean
import random
import model
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
from PIL import Image

parser = argparse.ArgumentParser('Options for FAFAC model demo in PyTorch...')
parser.add_argument('--video', type=str, default=None, help='path to input video')
parser.add_argument('--snapshot', type=str, default=None, help='use a pre-trained model snapshot')
parser.add_argument('--version', type=str, default=None, help='which version of SqueezeNet to load (1_0/1_1)')
parser.add_argument('--num-classes', type=int, default=5, help="how many classes to train for")
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')


args = parser.parse_args()
# check args
if args.snapshot is None:
    assert False, 'No model snapshot provided for inference!'
if args.video is None:
    assert False, 'No video provided!'
# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()

prepare_input = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
normalize = transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
activity_classes = ['Away from pedals', 'Hovering over Acc', 'Hovering over Brake', 'On Accelerator', 'On Brake']


def mark_image(im):
    im[:10, :, 0] = 0
    im[:10, :, 1] = 255
    im[:10, :, 2] = 0
    im[-10:, :, 0] = 0
    im[-10:, :, 1] = 255
    im[-10:, :, 2] = 0
    im[:, :10, 0] = 0
    im[:, :10, 1] = 255
    im[:, :10, 2] = 0
    im[:, -10:, 0] = 0
    im[:, -10:, 1] = 255
    im[:, -10:, 2] = 0
    return im


# inference function
def infer(net, im):
    X = normalize(prepare_input(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))))
    X = X.unsqueeze(0)
    if args.cuda:
        X = X.cuda()

    scores, masks = net(X)
    mask_shape = (int(0.2*im.shape[1]), int(0.2*im.shape[1]))
    away_from_pedals_mask  = cv2.resize(masks[0, 0, :, :].view(13, 13).detach().cpu().numpy(), mask_shape)
    hovering_over_acc_mask = cv2.resize(masks[0, 1, :, :].view(13, 13).detach().cpu().numpy(), mask_shape)
    hovering_over_brk_mask = cv2.resize(masks[0, 2, :, :].view(13, 13).detach().cpu().numpy(), mask_shape)
    on_acc_mask            = cv2.resize(masks[0, 3, :, :].view(13, 13).detach().cpu().numpy(), mask_shape)
    on_brk_mask            = cv2.resize(masks[0, 4, :, :].view(13, 13).detach().cpu().numpy(), mask_shape)

    class_1_mask = 255*away_from_pedals_mask
    class_2_mask = 255*hovering_over_acc_mask
    class_3_mask = 255*on_acc_mask
    class_4_mask = 255*hovering_over_brk_mask
    class_5_mask = 255*on_brk_mask
    class_1_mask = cv2.applyColorMap(class_1_mask.astype(np.uint8), cv2.COLORMAP_JET)
    class_2_mask = cv2.applyColorMap(class_2_mask.astype(np.uint8), cv2.COLORMAP_JET)
    class_3_mask = cv2.applyColorMap(class_3_mask.astype(np.uint8), cv2.COLORMAP_JET)
    class_4_mask = cv2.applyColorMap(class_4_mask.astype(np.uint8), cv2.COLORMAP_JET)
    class_5_mask = cv2.applyColorMap(class_5_mask.astype(np.uint8), cv2.COLORMAP_JET)
    
    scores = scores.view(-1, args.num_classes)
    pred = scores.data.max(1)[1].item()
    if pred == 0:
        class_1_mask = mark_image(class_1_mask)
    elif pred == 1:
        class_2_mask = mark_image(class_2_mask)
    elif pred == 3:
        class_3_mask = mark_image(class_3_mask)
    elif pred == 2:
        class_4_mask = mark_image(class_4_mask)
    elif pred == 4:
        class_5_mask = mark_image(class_5_mask)

    im_out = np.vstack((im, np.hstack((class_2_mask, class_3_mask, class_1_mask, class_4_mask, class_5_mask))))
    im_out = im_out.astype(np.uint8)

    cv2.putText(im_out, activity_classes[pred], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
    cv2.putText(im_out, activity_classes[pred], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

    return im_out


if __name__ == '__main__':
    net = model.squeezenet(args.version, args.snapshot)
    if args.cuda:
        net.cuda()
    net.eval()

    file_vidout = os.path.join(os.path.splitext(args.video)[0] + '_out.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid_shape = (320, 302) # (640, 608)
    out_vid = cv2.VideoWriter(file_vidout, fourcc, 30, out_vid_shape)

    cap = cv2.VideoCapture(args.video)
    frame_count_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    while True:
        captured, im = cap.read()
        if captured == False:
            break
        im_out = infer(net, im)
        im_out = cv2.resize(im_out, out_vid_shape)
        out_vid.write(im_out)
        print('Frame %d/%d....' % (frame_count, frame_count_total))
        frame_count += 1

    cap.release()
    out_vid.release()