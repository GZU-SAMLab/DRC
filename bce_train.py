import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR
import torch.optim.lr_scheduler as lr_scheduler
import time
import numpy as np
import math
from model import resnet34
from model import resnet50
from model import resnet101
from model import resnet152
from model import resnext50_32x4d
from model import resnext101_32x8d
from efficientnet_pytorch import EfficientNet
from model_dense import densenet121, load_state_dict
from pvt import pvt_small
from swin_transformer import SwinTransformer
from utils import load_pretrained
from cls_cvt import ConvolutionalVisionTransformer
from poolformer import poolformer_s24
from pyramid_vig import pvig_ti_224_gelu

from gcn_lib import Grapher, act_layer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = sys.argv

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

    # data_root = os.path.abspath(os.path.join(os.getcwd(), '../../CUB-200'))  # CUB-200
    # data_root = os.path.abspath(os.path.join(os.getcwd(), '../../ImageNet'))  # ImageNet-1k
    # data_root = os.path.abspath(os.path.join(os.getcwd(), './'))  # tiny-imageNet
    # data_root = os.path.abspath(os.path.join(os.getcwd(), './fine'))  # cifar100
    data_root = os.path.abspath(os.path.join(os.getcwd(), '../../cifar10/whole_data'))  # cifar10
    # data_root = os.path.abspath(os.path.join(os.getcwd(), '../../cifar100/class20/'))  # cifar20
    image_path = os.path.join(data_root)  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train_10%'),
    #                                      transform=data_transform['train'])
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'),      # 使用一半的训练集
                                         transform=data_transform['train'])

    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # batch_size = 128
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 10])  # number of workers
    batch_size = int(args[1])
    nw = int(args[2])

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
    directory = 'jobs/job951_0/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    net = resnet34()
    # net = resnet50()
    # net = EfficientNet.from_name('efficientnet-b3')
    # net = densenet121()
    # net = resnext50_32x4d()
    # net = pvt_small()
    # net = SwinTransformer()
    # net = ConvolutionalVisionTransformer()
    # net = poolformer_s24()
    # net = pvig_ti_224_gelu()

    model_name = 'resnet34_bce'
    # model_name = 'resnet50_bce'
    # model_name = 'effNetb3_bce'
    # model_name = 'densenet121_bce'
    # model_name = 'resneXt50_bce'
    # model_name = 'PVT-s_bce'
    # model_name = ' Swin-T_bce'
    # model_name = 'CVT'
    # model_name = 'Poolformer'
    # model_name = 'pvig-ti'

    model_weight_path = "./resnet34-pre.pth"
    # model_weight_path = "./resnet50-pre.pth"
    # model_weight_path = "./efficientnet-b3.pth"
    # model_weight_path = "./densenet121.pth"
    # model_weight_path = "./resnext50-pre.pth"
    # model_weight_path = "./pvt_small.pth"
    # model_weight_path = "./swin_tiny_patch4_window7_224.pth"
    # model_weight_path = "./CvT-13.pth"
    # model_weight_path = "./poolformer_s24.pth.tar"
    # model_weight_path = './pvig_ti_78.5.pth.tar'

    save_path = directory + model_name + '.pth'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)

    # in_channel = net.fc.in_features
    in_channel = 512
    # efficientNet -b2:1408,-b3:1536; densenet121:1024; PVT-s:512; Swin:768; CVT:384; poolformer:512; pvig:1024

    # num_class = 1000 # imagenet-1k
    # num_class = 200 # tiny-Imagenet, CUB-200
    # num_class = 100  # cifar100
    num_class = 10  # cifar10
    # num_class = 20  # cifar20

    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))  #for PvT, effnet, CVT, poolforemr, pvig
    # load_pretrained(model_weight_path, net) #for swin
    # load_state_dict(net, model_weight_path)  # for DenseNet

    del net.fc
    # del net._fc  #for effiNet
    # del net.classifier  # for DenseNet
    # del net.head    # for Pvt, CvT, SwinT, poolformer
    # net.prediction_bce = nn.Sequential(nn.Conv2d(384, in_channel, 1, bias=True),
    #                                nn.BatchNorm2d(1024),
    #                                act_layer('gelu'),
    #                                )  # for pVig

    net.include_top = False
    net.bce = True
    net.to(device)
    # / 22-7-22-10:03/

    lr = 0.0001
    lrf = 0.1
    weight_decay = 0
    momentum = 0.9
    epochs = 60
    best_acc = 0.0

    length_threshold = 100
    length_top = 100
    lambda1 = 20
    m = 3
    alpha = 0.01
    warmup_epochs = 0
    train_steps = len(train_loader)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    #optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    milestones = [20, 40]
    #milestones = [50, 100]
    gamma = 0.2
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    generate_proxies = torch.rand(num_class, in_channel).to(device)
    generate_proxies = F.normalize(generate_proxies, dim=1)
    indices = torch.topk(generate_proxies, length_top).indices
    binary_proxies = torch.zeros_like(generate_proxies)
    binary_proxies.scatter_(1, indices, 1)

    norm_bProxies = F.normalize(binary_proxies, dim=1, p=2)
    torch.save(generate_proxies, directory + 'proxies.pt')
    print('saved the generated proxies!')
    loss_function = nn.CrossEntropyLoss()
    loss_function1 = nn.MSELoss()

    print('job:%s' % directory)
    print('lr:%f' % lr)
    print('lrf:%f' % lrf)
    print('length_top:%d' % length_top)
    print('length_threshold:%d' % length_threshold)
    print('milestones:%s' % milestones)
    print('gamma:%f' % gamma)
    print('lambda1:%f' % lambda1)
    print('m:%d' % m)
    print('alpha:%f' % alpha)
    print('model:%s' % model_name)
    print('warmup_epochs:%d' % warmup_epochs)

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
            loss = 0
            (vectors, logits) = net(images.to(device), proxies=binary_proxies, length_threshold=length_threshold, m=m,
                                    alpha=alpha)
            predict_y = torch.max(logits, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.cuda()).sum().item()
            for i, j in enumerate(vectors):
                loss += loss_function1(j, norm_bProxies[labels[i]])
            loss *= lambda1
            print('loss_mse:%.3f' % loss)
            loss += loss_function(logits, labels.to(device))
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
        evaluate_time1 = time.time()
        acc = 0.0  # accumulate accurate number / epoch
        acc_top5 = 0
        # topN_mat_des = torch.load(directory + 'topN_set.pt')
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                _, outputs = net(val_images.to(device), proxies=binary_proxies, length_threshold=length_threshold, m=m,
                                 alpha=alpha)
                predict_y = torch.max(outputs, dim=1)[1]
                predict_top5 = torch.sort(outputs, dim=1, descending=True)[1][:, :5]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                for i, j in enumerate(predict_top5):
                    if val_labels[i] in j:
                        acc_top5 += 1

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
        val_accurate = acc / val_num
        val_accurateTop5 = acc_top5 / val_num
        evaluate_time2 = time.time()
        print('[epoch %d] train_loss: %.3f  train_accuracy:%.4f  val_accuracy: %.4f  val_acc5: %.4f' %
              (epoch + 1, running_loss / train_steps, train_accurate, val_accurate, val_accurateTop5))
        print('train_time cost: %d       eval_time cost: %d' % (
        train_time2 - train_time1, evaluate_time2 - evaluate_time1))
        if (val_accurate > best_acc):
            torch.save(net.state_dict(), save_path)
            print('saved the model!')
            best_acc = val_accurate

    torch.save(net.state_dict(), save_path)
    print('saved the model!')
    print('Finished Training')


if __name__ == '__main__':
    main()
