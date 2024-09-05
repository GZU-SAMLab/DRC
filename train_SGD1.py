import os
import sys
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import time
from model import resnet34
from model import resnet50
from model import resnet101
from model import resnet152
from model import resnext50_32x4d
from model import resnext101_32x8d
from efficientnet_pytorch import EfficientNet
from model_dense import densenet121, load_state_dict
from pvt import pvt_small
from cls_cvt import ConvolutionalVisionTransformer

from torch.optim.lr_scheduler import StepLR
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# parser.add_argument("--world_size",type=int)
# parser.add_argument("--node_rank",type=int)
# parser.add_argument("--master_port",default="29500",type=str)
parser.add_argument("--batch_size",default=128,type=int)
parser.add_argument("--nw",type=int)
# parser.add_argument("--addr_file",type=str)
args = parser.parse_args()

def main():
    global device
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), './fine'))  # get data root path
    image_path = os.path.join(data_root)  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),
                                         transform=data_transform['train'])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = args.batch_size
    nw = args.nw

    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform['val'])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    directory = 'jobs/job608/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    net = ConvolutionalVisionTransformer()
    model_name = 'CVT-13'
    save_path = directory + model_name + '.pth'

    # efficientNet-B3:1536; densenet121:1024; PVT-S:512; Swin:768; CVT:384; Poolformer:512, pvig-t:1024
    in_channel = 384
    num_class = 100

    # net.fc = nn.Linear(in_channel, num_class)
    # net._fc = nn.Linear(in_channel, num_class)  #for effiNet
    # net.classifier = nn.Linear(in_channel, num_class)  # for DenseNet
    net.head = nn.Linear(in_channel, num_class)  # for CVT, PvT, Swin, poolformer

    net.bce = False
    net.include_top = True
    net.to(device)
    # / 22-7-22-10:03/

    loss_function = nn.CrossEntropyLoss()
    lr = 0.1
    milestones = [150, 225]
    gamma = 0.1
    epochs = 300
    best_acc = 0.0
    warmup_epochs = 0
    weight_decay = 1e-4
    nesterov = True

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, dampening=0, momentum=0.9, nesterov=nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    print('job:%s' % directory)
    print('lr:%f' % lr)
    print('milestones:%s' % milestones)
    print('gamma:%f' % gamma)
    print('model:%s' % model_name)
    print('weight_decay:%f' % weight_decay)

    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            lr_lambda = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_lambda

        train_time1 = time.time()
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        train_acc = 0.0
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            (logits) = net(images.to(device))
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.cuda()).sum().item()
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_time2 = time.time()
        scheduler.step()
        train_accurate = train_acc / train_num

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            running_val_loss = 0
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_loss = loss_function(outputs, val_labels.to(device))
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                running_val_loss += val_loss.item()
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_CE_loss: %.3f   train_accuracy:%.4f  val_accuracy: %.4f' %
              (epoch + 1, running_loss / train_steps, running_val_loss / val_steps, train_accurate, val_accurate))
        print('train_time cost: %d' % (
                train_time2 - train_time1))
        if val_accurate >= best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
            print('saved!')

    print('Finished Training')

if __name__ == '__main__':
    main()