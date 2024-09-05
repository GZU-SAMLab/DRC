import os
import sys
import json
import argparse
import subprocess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import time
import numpy as np
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

import torch.distributed as dist
import torch.multiprocessing as mp
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--world_size",type=int)
parser.add_argument("--node_rank",type=int)
# parser.add_argument("--master_addr",default="127.0.0.1",type=str)
parser.add_argument("--master_port",default="29500",type=str)
parser.add_argument("--batch_size",default=128,type=int)
parser.add_argument("--nw",type=int)
parser.add_argument("--addr_file",type=str)
args = parser.parse_args()

directory = 'jobs/job601/'
if not os.path.exists(directory):
    os.makedirs(directory)

# with open(args.addr_file, "w") as file:
#     pass
# print(f"空文件{args.addr_file} 已创建")
# os.environ['MASTER_ADDR'] = '10.0.0.1'
# os.environ['MASTER_PORT'] = '2000'
string= """192.168.0.1 workstation workstation.example.com
192.168.0.2 head head.example.com
192.168.0.3 cpu1 cpu1.example.com
192.168.0.4 cpu2 cpu2.example.com
192.168.0.5 cpu3 cpu3.example.com
192.168.0.6 cpu4 cpu4.example.com
192.168.0.7 cpu5 cpu5.example.com
192.168.0.8 cpu6 cpu6.example.com
192.168.0.9 cpu7 cpu7.example.com
192.168.0.10 cpu8 cpu8.example.com
192.168.0.11 cpu9 cpu9.example.com
192.168.0.12 cpu10 cpu10.example.com
192.168.0.13 cpu11 cpu11.example.com
192.168.0.14 cpu12 cpu12.example.com
192.168.0.15 cpu13 cpu13.example.com
192.168.0.16 gpu1 gpu1.example.com
192.168.0.17 gpu2 gpu2.example.com
192.168.0.18 gpu3 gpu3.example.com
192.168.0.19 gpu4 gpu4.example.com
192.168.0.20 gpu5 gpu5.example.com
192.168.0.21 gpu6 gpu6.example.com
192.168.0.22 gpu7 gpu7.example.com
192.168.0.23 gpu8 gpu8.example.com
192.168.0.24 gpu9 gpu9.example.com
192.168.0.25 gpu10 gpu10.example.com
192.168.0.26 gpu11 gpu11.example.com
192.168.0.27 gpu12 gpu12.example.com
192.168.0.28 gpu13 gpu13.example.com
192.168.0.29 gpu14 gpu14.example.com
192.168.0.30 gpu15 gpu15.example.com
192.168.0.31 gpu16 gpu16.example.com
192.168.0.32 gpu17 gpu17.example.com
192.168.0.33 gpu18 gpu18.example.com
192.168.0.34 gpu19 gpu19.example.com
192.168.0.35 gpu20 gpu20.example.com
192.168.0.36 gpu21 gpu21.example.com
192.168.0.37 gpu22 gpu22.example.com
192.168.0.38 gpu23 gpu23.example.com
192.168.0.39 gpu24 gpu24.example.com
192.168.0.40 cpu14 cpu14.example.com
192.168.0.41 cpu15 cpu15.example.com
192.168.0.42 cpu16 cpu16.example.com
192.168.0.43 cpu17 cpu17.example.com
192.168.0.44 cpu18 cpu18.example.com
192.168.0.45 cpu19 cpu19.example.com
192.168.0.46 cpu20 cpu20.example.com
192.168.0.47 cpu21 cpu21.example.com
192.168.0.48 cpu22 cpu22.example.com
192.168.0.49 cpu23 cpu23.example.com
192.168.0.50 cpu24 cpu24.example.com
192.168.0.51 cpu25 cpu25.example.com
192.168.0.52 cpu26 cpu26.example.com
192.168.0.53 cpu27 cpu27.example.com
192.168.0.54 cpu28 cpu28.example.com
192.168.0.55 gpu25 gpu25.example.com
192.168.0.56 gpu26 gpu26.example.com
192.168.0.57 gpu27 gpu27.example.com
192.168.0.58 gpu28 gpu28.example.com
192.168.0.59 gpu29 gpu29.example.com
192.168.0.60 gpu30 gpu30.example.com
192.168.0.61 gpu31 gpu31.example.com
192.168.0.62 gpu32 gpu32.example.com
192.168.0.63 gpu33 gpu33.example.com
192.168.0.64 gpu34 gpu34.example.com
192.168.0.65 gpu35 gpu35.example.com
192.168.0.66 gpu36 gpu36.example.com
192.168.0.67 gpu37 gpu37.example.com
192.168.0.68 gpu38 gpu38.example.com
192.168.0.69 gpu39 gpu39.example.com
192.168.0.70 gpu40 gpu40.example.com
192.168.0.71 gpu41 gpu41.example.com
192.168.0.72 gpu42 gpu42.example.com
192.168.0.73 gpu43 gpu43.example.com
192.168.0.74 gpu44 gpu44.example.com
192.168.0.75 gpu45 gpu45.example.com
192.168.0.76 gpu46 gpu46.example.com
192.168.0.77 gpu47 gpu47.example.com
192.168.0.78 gpu48 gpu48.example.com
192.168.0.79 gpu49 gpu49.example.com
192.168.0.80 gpu50 gpu50.example.com
192.168.0.81 cpu29 cpu29.example.com
192.168.0.82 cpu30 cpu30.example.com
192.168.0.83 cpu31 cpu31.example.com
192.168.0.84 cpu32 cpu32.example.com
192.168.0.85 gpu51 gpu51.example.com
192.168.0.86 gpu52 gpu52.example.com
"""
gpu_dict = {}
for line in string.split('\n'):
    if not line.strip():
        continue
    # 分割每行，提取IP地址和GPU名称
    ip, gpu_name, _ = line.split(' ')
    gpu_dict[gpu_name] = ip

def main():
    local_size = torch.cuda.device_count()
    print('local_size: %s'%local_size)
    mp.spawn(train,
             args=(args.node_rank, local_size, args.world_size), nprocs=local_size, join=True)

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

from torch.utils.data.distributed import DistributedSampler

def get_dataloader(batch_size, nw, data_transform, image_path, rank, world_size):
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=nw,
                                               sampler=train_sampler,
                                               pin_memory=True)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    validate_sampler = DistributedSampler(validate_dataset, num_replicas=world_size, rank=rank)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  sampler=validate_sampler,
                                                  pin_memory=True)

    return train_loader, validate_loader

def setup_model(local_rank, device):
    model_name = 'EfficientNet-b3'
    # model_name = 'PVT-S'
    net = EfficientNet.from_name('efficientnet-b3')
    # net = pvt_small()
    del net._fc
    # net.fc = nn.Linear(in_channel, num_class)
    # net._fc = nn.Linear(in_channel, num_class)  # for effiNet
    # net.classifier = nn.Linear(in_channel, num_class)  #for DenseNet
    # net.head = nn.Linear(in_channel, num_class)  # for CVT, PvT, Swin, poolformer

    net.include_top = False
    net.bce = True
    net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])
    return net, model_name

def train(local_rank, node_rank, local_size, world_size):
    rank = local_rank + node_rank * local_size
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    torch.cuda.set_device(local_rank)

    print('node_rank: %d '%node_rank)
    os.environ['MASTER_PORT'] = args.master_port
    if node_rank == 0:
        node_list = os.environ["SLURM_NODELIST"]
        master_name = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        with open(args.addr_file,'w') as file:
            file.write(master_name)
    else:
        with open(args.addr_file,'r') as file:
            master_name = file.read()
    os.environ['MASTER_ADDR'] = gpu_dict[master_name]
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    device = torch.device("cuda", local_rank)

    print('Using {} dataloader workers every process'.format(args.nw))
    data_root = os.path.abspath(os.path.join(os.getcwd(), './fine'))  # get data root path
    image_path = os.path.join(data_root)  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 数据加载器、模型、损失函数和优化器的设置
    train_loader, validate_loader = get_dataloader(args.batch_size, args.nw, data_transform, image_path, rank, world_size)
    net, model_name = setup_model(local_rank, device)
    save_path = directory + model_name + '.pth'

    lr = 0.1
    epochs = 300
    best_acc = 0.0

    in_channel = 1536
    # efficientNet-B3:1536; densenet121:1024; PVT-S:512; Swin:768; CVT:384; Poolformer:512, pvig-t:1024
    num_class = 100
    length_threshold = 30
    length_top = 30
    length_channel = in_channel
    params = [p for p in net.parameters() if p.requires_grad]
    # optimizer = optim.Adam(params, lr=lr)
    weight_decay = 1e-4
    momentum = 0.9
    nesterov = True
    optimizer = optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    milestones = [150, 225]
    gamma = 0.1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    generate_proxies = torch.rand(num_class, length_channel).to(device)
    generate_proxies = F.normalize(generate_proxies, dim=1)
    indices = torch.topk(generate_proxies, length_top).indices
    binary_proxies = torch.zeros_like(generate_proxies)
    binary_proxies.scatter_(1, indices, 1)

    norm_bProxies = F.normalize(binary_proxies, dim=1, p=2)
    torch.save(generate_proxies, directory + 'proxies.pt')
    print('saved the generated proxies!')
    loss_function = nn.CrossEntropyLoss()
    loss_function1 = nn.MSELoss()
    lambda1 = 3
    m = 3
    alpha = 0.01
    warmup_epochs = 0
    print('job:%s' % directory)
    print('lr:%f' % lr)
    print('length_top:%d' % length_top)
    print('length_threshold:%d' % length_threshold)
    print('milestones:%s' % milestones)
    print('gamma:%f' % gamma)
    print('lambda1:%f' % lambda1)
    print('m:%d' % m)
    print('alpha:%f' % alpha)
    print('model:%s' % model_name)
    print('warmup_epochs:%d' % warmup_epochs)

    # ... 损失函数和优化器的设置 ...
    for epoch in range(epochs):
    # ... 训练循环 ...
        if epoch < warmup_epochs:
            lr_lambda = (epoch+1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * lr_lambda

        train_time1=time.time()
        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            loss = 0
            (vectors,logits) = net(images.to(device), proxies=binary_proxies, length_threshold=length_threshold, m=m, alpha=alpha)
            for i,j in enumerate(vectors):
                loss += loss_function1(j, norm_bProxies[labels[i]])
            loss *= lambda1
            loss = loss_function(logits,labels.to(device))
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)

        train_time2 = time.time()
        scheduler.step()
        # validate
        net.eval()
        evaluate_time1=time.time()
        acc = torch.tensor(0.0).to(device) # accumulate accurate number / epoch
        acc_top5 = torch.tensor(0).to(device)
        #val_num = torch.tensor(len(validate_loader)*batch_size).to(device)
        val_num = torch.tensor(10000).to(device)
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                _,outputs = net(val_images.to(device),proxies=binary_proxies, length_threshold=length_threshold, m=m, alpha=alpha)
                predict_y = torch.max(outputs, dim=1)[1]
                predict_top5 = torch.sort(outputs, dim=1, descending=True)[1][:, :5]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                for i, j in enumerate(predict_top5):
                    if val_labels[i] in j:
                        acc_top5 += 1

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        print("The %d process's acc is %f:" % (rank, acc))
        dist.reduce(acc, 0, op=dist.ReduceOp.SUM)
        dist.reduce(acc_top5, 0, op=dist.ReduceOp.SUM)
        if rank==0:
            val_accurate = acc / val_num
            val_accurateTop5 = acc_top5 / val_num
            evaluate_time2=time.time()
            print('acc_reduce:%f'% acc)
            print('[epoch %d]  val_accuracy: %.4f  val_acc5: %.4f' %
                  (epoch + 1, val_accurate, val_accurateTop5))
            print('train_time cost: %d       eval_time cost: %d'%(train_time2-train_time1, evaluate_time2-evaluate_time1))
            if(val_accurate>best_acc):
                torch.save(net.state_dict(), save_path)
                print('saved the model!')
                best_acc=val_accurate
    cleanup()
    print('Finished Training')
    if os.path.exists(args.addr_file):
        os.remove(args.addr_file)
if __name__ == '__main__':
    main()
