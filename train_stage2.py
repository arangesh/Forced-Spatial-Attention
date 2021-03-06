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
import json
from datetime import datetime
from statistics import mean
import random
import model
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
from scipy.io import loadmat
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

parser = argparse.ArgumentParser('Options for training models using Forced Spatial Attention in PyTorch...')
parser.add_argument('--dataset-root-path', type=str, default=None, help='path to dataset')
parser.add_argument('--version', type=str, default=None, help='which version of SqueezeNet to load (1_0/1_1/FC)')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='use a pre-trained model snapshot')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for training')
parser.add_argument('--num-classes', type=int, default=5, help="how many classes to train for")
parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.0005, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum for gradient step')
parser.add_argument('--weight-decay', type=float, default=0.0005, metavar='WD', help='weight decay')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--FSA', action='store_true', default=False, help='use forced spatial attention loss for training')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--random-transforms', action='store_true', default=False, help='apply random transforms to input while training')


args = parser.parse_args()
# check args
if args.dataset_root_path is None:
    assert False, 'Path to dataset not provided!'
if args.FSA and (args.version == 'FC'):
    assert False, 'Cannot use this model version for forced attention loss!'
if all(args.version != x for x in ['1_0', '1_1', 'FC']):
    assert False, 'Model version not recognized!'

# Output class labels
activity_classes = ['Away from pedals', 'Hovering over Acc', 'Hovering over Brake', 'On Accelerator', 'On Brake']

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M")
    args.output_dir = os.path.join('.', 'experiments', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    assert False, 'Output directory already exists!'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig3, ax3 = plt.subplots()
    im = ax3.imshow(cm, interpolation='nearest', cmap=cmap)
    ax3.figure.colorbar(im, ax=ax3)
    # We want to show all ticks...
    ax3.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.output_dir, 'confusion_matrix.jpg'))
    return


def FSA_loss(output_masks, target_masks, labels):
    loss = 0
    for i, label in enumerate(labels):
        if label == 0:
            continue

        best_mse = np.inf
        for j in range(target_masks[i, :, :, :].size()[0]):
            cur_mse = torch.sum((output_masks[i, label, :, :] - target_masks[i, j, :, :]) ** 2).item()
            if cur_mse < best_mse:
                best_mse = cur_mse
                mask_id = j
        loss += F.mse_loss(output_masks[i, label, :, :], target_masks[i, mask_id, :, :])
        for l in range(args.num_classes):
            if l == label:
                continue
            loss += 0.2*torch.mean(output_masks[i, l, :, :] * target_masks[i, mask_id, :, :])
    return 10.0*loss/args.batch_size


def get_classification_data(split):
    # subjects for cross-subject validation
    val_subjects = ['subject009_drive001', 'subject009_drive002', 'subject009_drive003', 'subject009_drive004', 'subject009_drive005',
    'subject009_drive006', 'subject009_drive007', 'subject009_drive008', 'subject009_drive009', 'subject009_drive010']

    if split == 'train':
        all_images = []
        all_labels = []
        dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'Away from pedals', '*', '*.jpg')
        tmp = sorted(glob.glob(dir_tmp))
        all_labels += [0]*len(tmp)
        all_images += tmp

        dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'Hovering over Acc', '*', '*.jpg')
        tmp = sorted(glob.glob(dir_tmp))
        all_labels += [1]*len(tmp)
        all_images += tmp

        dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'Hovering over Brake', '*', '*.jpg')
        tmp = sorted(glob.glob(dir_tmp))
        all_labels += [2]*len(tmp)
        all_images += tmp

        dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'On Accelerator', '*', '*.jpg')
        tmp = sorted(glob.glob(dir_tmp))
        all_labels += [3]*len(tmp)
        all_images += tmp

        dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'On Brake', '*', '*.jpg')
        tmp = sorted(glob.glob(dir_tmp))
        all_labels += [4]*len(tmp)
        all_images += tmp

        images = []
        labels = []
        for idx, path in enumerate(all_images):
            if any(x in path for x in val_subjects):
                continue
            else:
                images.append(path)
                labels.append(all_labels[idx])

        print('Loaded %d foot images!' % len(labels))
    else:
        images = []
        labels = []
        for val_subj in val_subjects:
            dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'Away from pedals', val_subj, '*.jpg')
            tmp = sorted(glob.glob(dir_tmp))
            labels += [0]*len(tmp)
            images += tmp

        for val_subj in val_subjects:
            dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'Hovering over Acc', val_subj, '*.jpg')
            tmp = sorted(glob.glob(dir_tmp))
            labels += [1]*len(tmp)
            images += tmp

        for val_subj in val_subjects:
            dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'Hovering over Brake', val_subj, '*.jpg')
            tmp = sorted(glob.glob(dir_tmp))
            labels += [2]*len(tmp)
            images += tmp

        for val_subj in val_subjects:
            dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'On Accelerator', val_subj, '*.jpg')
            tmp = sorted(glob.glob(dir_tmp))
            labels += [3]*len(tmp)
            images += tmp

        for val_subj in val_subjects:
            dir_tmp = os.path.join(args.dataset_root_path, 'raw_images', 'On Brake', val_subj, '*.jpg')
            tmp = sorted(glob.glob(dir_tmp))
            labels += [4]*len(tmp)
            images += tmp

        print('Loaded %d foot images!' % len(labels))

    attention_masks = loadmat('./matlab/stage2_attention_mask.mat')
    time.sleep(1)
    return images, np.array(labels, dtype='int64'), attention_masks


class Dataset(data.Dataset):
    def __init__(self, split='train', random_transforms=False):
        'Initialization'
        print('Preparing '+split+' dataset...')
        self.split = split
        
        self.prepare_input = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        self.normalize = transforms.Normalize((0.491399689874, 0.482158419622, 0.446530924224), (0.247032237587, 0.243485133253, 0.261587846975))
        if random_transforms:
            self.transforms = transforms.Compose([transforms.Resize((256, 256)),
                transforms.RandomRotation((-15, 15)), 
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
                transforms.ToTensor()])
        else:
            self.transforms = None

        self.images, self.labels, self.attention_masks = get_classification_data(self.split)
        print('Finished preparing '+split+' dataset!')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        y = self.labels[index]
        im = Image.fromarray(cv2.cvtColor(cv2.imread(self.images[index]), cv2.COLOR_BGR2RGB))
        original_masks = self.attention_masks['class'+str(y)+'_masks']

        if self.transforms is None:
            X = self.normalize(self.prepare_input(im))
            attention_masks = self.prepare_input(Image.fromarray(original_masks[:, :, :3]))
            if original_masks.shape[2] > 3:
                for ch in range(3, original_masks.shape[2], 3):
                    attention_masks = torch.cat((attention_masks, self.prepare_input(Image.fromarray(original_masks[:, :, ch:ch+3]))), 0)
        else:
            # create explicit seed so that same random transform is applied to both image and mask
            seed = random.randint(0,2**32)
            random.seed(seed)
            X = self.normalize(self.transforms(im))
            random.seed(seed)
            attention_masks = self.transforms(Image.fromarray(original_masks[:, :, :3]))
            if original_masks.shape[2] > 3:
                for ch in range(3, original_masks.shape[2], 3):
                    random.seed(seed)
                    attention_masks = torch.cat((attention_masks, self.transforms(Image.fromarray(original_masks[:, :, ch:ch+3]))), 0)
        return X, attention_masks, y


kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}
train_loader = torch.utils.data.DataLoader(Dataset('train', args.random_transforms), **kwargs)
val_loader = torch.utils.data.DataLoader(Dataset('val', False), **kwargs)

# global var to store best validation accuracy across all epochs
best_accuracy = 0.0


# training function
def train(net, epoch):
    epoch_loss = list()
    correct = 0
    net.train()
    for b_idx, (data, attention, targets) in enumerate(train_loader):
        if args.cuda:
            data, attention, targets = data.cuda(), attention.cuda(), targets.cuda()
        # convert the data and targets into Variable and cuda form
        data, attention, targets = Variable(data), Variable(attention), Variable(targets)

        # train the network
        optimizer.zero_grad()

        if args.FSA:
            scores, masks = net.forward(data)
            scores = scores.view(-1, args.num_classes)
            loss = F.nll_loss(scores, targets) + FSA_loss(masks, attention[:, :, 8:216:16, 8:216:16], targets)
        else:
            scores, masks = net.forward(data)
            scores = scores.view(-1, args.num_classes)
            loss = F.nll_loss(scores, targets)
        # compute the accuracy
        pred = scores.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data).cpu().sum()

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        if b_idx % args.log_schedule == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (b_idx+1) * len(data), len(train_loader.dataset),
                100. * (b_idx+1)*len(data) / len(train_loader.dataset), loss.item()))
            with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
                f.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                epoch, (b_idx+1) * len(data), len(train_loader.dataset),
                100. * (b_idx+1)*len(data) / len(train_loader.dataset), loss.item()))

    # now that the epoch is completed calculate statistics and store logs
    avg_loss = mean(epoch_loss)
    print("------------------------\nAverage loss for epoch = {:.2f}".format(avg_loss))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\nAverage loss for epoch = {:.2f}\n".format(avg_loss))

    train_accuracy = 100.0*float(correct)/float(len(train_loader.dataset))
    print("Accuracy for epoch = {:.2f}%\n------------------------".format(train_accuracy))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("Accuracy for epoch = {:.2f}%\n------------------------\n".format(train_accuracy))

    return net, avg_loss, train_accuracy


# validation function
def val(net):
    global best_accuracy
    correct = 0
    net.eval()
    pred_all = np.array([], dtype='int64')
    target_all = np.array([], dtype='int64')
    
    for idx, (data, attention, target) in enumerate(val_loader):
        if args.cuda:
            data, attention, target = data.cuda(), attention.cuda(), target.cuda()
        data, attention, target = Variable(data), Variable(attention), Variable(target)

        # do the forward pass
        scores = net(data)[0]
        scores = scores.view(-1, args.num_classes)
        pred = scores.data.max(1)[1]  # got the indices of the maximum, match them
        correct += pred.eq(target.data).cpu().sum()
        print('Done with image {} out of {}...'.format(min(args.batch_size*(idx+1), len(val_loader.dataset)), len(val_loader.dataset)))
        pred_all   = np.append(pred_all, pred.cpu().numpy())
        target_all = np.append(target_all, target.cpu().numpy())

    print("------------------------\nPredicted {} out of {}".format(correct, len(val_loader.dataset)))
    val_accuracy = 100.0*float(correct)/len(val_loader.dataset)
    print("Validation accuracy = {:.2f}%\n------------------------".format(val_accuracy))
    with open(os.path.join(args.output_dir, "logs.txt"), "a") as f:
        f.write("\n------------------------\nPredicted {} out of {}\n".format(correct, len(val_loader.dataset)))
        f.write("Validation accuracy = {:.2f}%\n------------------------\n".format(val_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        # save the model
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'squeezenet_' + args.version + '.pth'))
        plot_confusion_matrix(target_all, pred_all, activity_classes)

    return val_accuracy


if __name__ == '__main__':
    # get the model, load pretrained weights, and convert it into cuda for if necessary
    net = model.squeezenet(args.version, args.snapshot)
    if args.cuda:
        net.cuda()
    print(net)

    # create a temporary optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    fig1, ax1 = plt.subplots()
    plt.grid(True)
    train_loss = list()

    fig2, ax2 = plt.subplots()
    plt.grid(True)
    ax2.plot([], 'g', label='Train accuracy')
    ax2.plot([], 'b', label='Validation accuracy')
    ax2.legend()
    train_acc, val_acc = list(), list()
    for i in range(1, args.epochs+1):
        net, avg_loss, acc = train(net, i)
        # plot the loss
        train_loss.append(avg_loss)
        ax1.plot(train_loss, 'k')
        fig1.savefig(os.path.join(args.output_dir, "train_loss.jpg"))

        # plot the train and val accuracies
        train_acc.append(acc)
        val_acc.append(val(net))
        ax2.plot(train_acc, 'g', label='Train accuracy')
        ax2.plot(val_acc, 'b', label='Validation accuracy')
        fig2.savefig(os.path.join(args.output_dir, 'trainval_accuracy.jpg'))
    plt.close('all')
