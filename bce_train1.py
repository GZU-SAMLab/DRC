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
import time
import numpy as np
#from resnest import resnest50
from model import resnet50
from model import resnext50_32x4d
from efficientnet_pytorch import EfficientNet
from model_dense import densenet121, load_state_dict
from pvt import pvt_small
from swin_transformer import SwinTransformer
from utils import load_pretrained
import torch.distributed as dist
import torch.multiprocessing as mp
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = sys.argv
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '20002'

directory = 'jobs/job231/'
if not os.path.exists(directory):
    os.makedirs(directory)

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
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

from torch.utils.data.distributed import DistributedSampler
def get_dataloader(batch_size, nw, data_transform, image_path, rank, world_size):
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=nw,
                                               sampler=train_sampler)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    validate_sampler = DistributedSampler(validate_dataset, num_replicas=world_size, rank=rank)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw,
                                                  sampler=validate_sampler)

    return train_loader, validate_loader

def setup_model(rank, device):
    net = resnet50()
    # net = EfficientNet.from_name('efficientnet-b2',num_classes=1000)
    # net = densenet121()
    # net = pvt_small()
    # net = SwinTransformer()

    model_weight_path = "./resnet50-pre.pth"
    # model_weight_path = "./efficientnet-b2.pth"
    # model_weight_path = './densenet121.pth'
    # model_weight_path = "./pvt_small.pth"
    # model_weight_path = "./swin_tiny_patch4_window7_224.pth"

    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    # load_state_dict(net, model_weight_path)
    # load_pretrained(model_weight_path, net) # for swin

    del net.fc
    #del net._fc
    #del net.classifier
    #del net.head

    net.include_top = False
    net.bce = True
    net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    return net

def train(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    model_name = 'res50'
    # model_name = 'effNet-b2'
    # model_name = 'dense-121'
    # model_name = 'PVT-S'
    # model_name = 'Swin-T'

    save_path = directory + model_name + '.pth'
    #batch_size = 128
    batch_size = int(args[1])
    nw = int(args[2])
    #nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, os.cpu_count()//torch.cuda.device_count()])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    data_root = os.path.abspath(os.path.join(os.getcwd(), '.'))  # get data root path
    image_path = os.path.join(data_root)  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 数据加载器、模型、损失函数和优化器的设置
    train_loader, validate_loader = get_dataloader(batch_size, nw, data_transform, image_path, rank, world_size)
    net = setup_model(rank, device)

    lr = 0.0001
    epochs = 120
    best_acc = 0.0

    in_channel = 2048
    # efficientNetv2:1408; densenet121:1024; PVT-S:512; Swin:768

    num_class = 200
    length_threshold = 50
    length_top = 50
    length_channel = in_channel
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    milestones = [30, 60, 90]
    gamma = 0.2
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
    warmup_epochs = 5
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
        # dist.all_reduce(acc, op=dist.ReduceOp.SUM, group=None, async_op=False)
        # dist.all_reduce(val_num, op=dist.ReduceOp.SUM, group=None, async_op=False)
        # dist.all_reduce(acc_top5, op=dist.ReduceOp.SUM, group=None, async_op=False)
        # dist.all_reduce(running_loss, op=dist.ReduceOp.SUM, group=None, async_op=False)
        # dist.all_reduce(train_steps.to(device), op=dist.ReduceOp.SUM, group=None, async_op=False)
        dist.reduce(acc, 0, op=dist.ReduceOp.SUM)
        #dist.reduce(val_num, 0, op=dist.ReduceOp.SUM)
        dist.reduce(acc_top5, 0, op=dist.ReduceOp.SUM)
        # dist.reduce(running_loss, 0, op=dist.ReduceOp.SUM)
    #    dist.reduce(train_steps.to(device), 0, op=dist.ReduceOp.SUM)
        if rank==0:
            val_accurate = acc / val_num
            val_accurateTop5 = acc_top5 / val_num
            evaluate_time2=time.time()
            print('[epoch %d]  val_accuracy: %.4f  val_acc5: %.4f' %
                  (epoch + 1, val_accurate, val_accurateTop5))
            print('train_time cost: %d       eval_time cost: %d'%(train_time2-train_time1, evaluate_time2-evaluate_time1))
            if(val_accurate>best_acc):
                torch.save(net.state_dict(), save_path)
                print('saved the model!')
                best_acc=val_accurate
    cleanup()
    print('Finished Training')

if __name__ == '__main__':
    main()
