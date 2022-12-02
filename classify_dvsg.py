# -*- coding: utf-8 -*-
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets import random_temporal_delete
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime
import h5py
from spikingjelly.activation_based import lava_exchange

from lava.lib.dl import netx, slayer
from lava.proc import io
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_conditions import RunSteps

def export_hdf5(net, filename):
    # network export to hdf5 format
    h = h5py.File(filename, 'w')  #写文件
    layer = h.create_group('layer')
    for i, b in enumerate(net):
        handle = layer.create_group(f'{i}')
        b.export_hdf5(handle)



class DVSGNet(nn.Module):
    def __init__(self, size: int, ds: int = 1, channels: int = 16):
        super().__init__()

        conv_fc = [
            lava_exchange.BlockContainer(
                nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                          surrogate_function=surrogate.ATan(), detach_reset=True)
            ),

            lava_exchange.BlockContainer(
                nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                          surrogate_function=surrogate.ATan(),
                                          detach_reset=True)
            ),
        ]

        for i in range(ds - 1):
            conv_fc.append(
                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                    lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                              surrogate_function=surrogate.ATan(), detach_reset=True)
                )
            )

            conv_fc.append(
                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                    lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                              surrogate_function=surrogate.ATan(),
                                              detach_reset=True)
                )
            )


        conv_fc.append(
            lava_exchange.BlockContainer(
                nn.Flatten(),
                None
            )
        )

        conv_fc.append(
            lava_exchange.BlockContainer(
                nn.Linear(channels * (size >> ds) * (size >> ds), 11, bias=False),
                lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                          surrogate_function=surrogate.ATan(),
                                          detach_reset=True)
            )
        )


        self.conv_fc = nn.Sequential(*conv_fc)

    def to_lava(self):
        ret = [
            # slayer.block.cuba.Input()
        ]

        for i in range(self.conv_fc.__len__()):
            m = self.conv_fc[i]
            if isinstance(m, lava_exchange.BlockContainer):
                ret.append(m.to_lava_block())

        return nn.Sequential(*ret)


    def forward(self, x):
        return self.conv_fc(x)

class DVSGNet2(nn.Module):
    def __init__(self, size: int, ds: int = 1, channels: int = 16):
        super().__init__()

        conv_fc = [
            lava_exchange.BlockContainer(
                nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                          surrogate_function=surrogate.ATan(), detach_reset=True)
            ),

            lava_exchange.BlockContainer(
                nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                          surrogate_function=surrogate.ATan(),
                                          detach_reset=True)
            ),
        ]

        for i in range(ds - 1):

            conv_fc.append(
                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels * 2, kernel_size=3, stride=1, padding=1, bias=False),
                    lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                              surrogate_function=surrogate.ATan(), detach_reset=True)
                )
            )

            conv_fc.append(
                lava_exchange.BlockContainer(
                    nn.Conv2d(channels * 2, channels * 2, kernel_size=2, stride=2, bias=False),
                    lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                              surrogate_function=surrogate.ATan(),
                                              detach_reset=True)
                )
            )
            channels *= 2


        conv_fc.append(
            lava_exchange.BlockContainer(
                nn.Flatten(),
                None
            )
        )

        conv_fc.append(
            lava_exchange.BlockContainer(
                nn.Linear(channels * (size >> ds) * (size >> ds), 11, bias=False),
                lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                          surrogate_function=surrogate.ATan(),
                                          detach_reset=True)
            )
        )


        self.conv_fc = nn.Sequential(*conv_fc)

    def to_lava(self):
        ret = [
            # slayer.block.cuba.Input()
        ]

        for i in range(self.conv_fc.__len__()):
            m = self.conv_fc[i]
            if isinstance(m, lava_exchange.BlockContainer):
                ret.append(m.to_lava_block())

        return nn.Sequential(*ret)


    def forward(self, x):
        return self.conv_fc(x)

class DVSGNetBN(nn.Module):
    def __init__(self, size: int, ds: int = 1, channels: int = 16):
        super().__init__()

        conv_fc = [
            lava_exchange.BlockContainer(
                nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1, bias=False),
                lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                          surrogate_function=surrogate.ATan(), detach_reset=True,
                                          norm=lava_exchange.BatchNorm2d(channels))
            ),

            lava_exchange.BlockContainer(
                nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                          surrogate_function=surrogate.ATan(),
                                          detach_reset=True, norm=lava_exchange.BatchNorm2d(channels))
            ),
        ]

        for i in range(ds - 1):
            conv_fc.append(
                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                    lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                              surrogate_function=surrogate.ATan(), detach_reset=True, norm=lava_exchange.BatchNorm2d(channels))
                )
            )

            conv_fc.append(
                lava_exchange.BlockContainer(
                    nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                    lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                              surrogate_function=surrogate.ATan(),
                                              detach_reset=True, norm=lava_exchange.BatchNorm2d(channels))
                )
            )


        conv_fc.append(
            lava_exchange.BlockContainer(
                nn.Flatten(),
                None
            )
        )

        conv_fc.append(
            lava_exchange.BlockContainer(
                nn.Linear(channels * (size >> ds) * (size >> ds), 11, bias=False),
                lava_exchange.CubaLIFNode(current_decay=1., voltage_decay=0.5, v_threshold=1.,
                                          surrogate_function=surrogate.ATan(),
                                          detach_reset=True)
            )
        )


        self.conv_fc = nn.Sequential(*conv_fc)

    def to_lava(self):
        ret = [
            # slayer.block.cuba.Input()
        ]

        for i in range(self.conv_fc.__len__()):
            m = self.conv_fc[i]
            if isinstance(m, lava_exchange.BlockContainer):
                ret.append(m.to_lava_block())

        return nn.Sequential(*ret)


    def forward(self, x):
        return self.conv_fc(x)

def encoder(x_seq: torch.Tensor, size: int):
    # x_seq.shape = [T, N, C, 128, 128]
    if size == 128:
        pass
    else:
        T, N, C, H, W = x_seq.shape
        x_seq = x_seq.flatten(0, 1)
        k = 128 // size
        x_seq = F.avg_pool2d(x_seq, k, k) * (k * k)
        x_seq = x_seq.view(T, N, C, size, size)

    return surrogate.heaviside(x_seq - 1.)

def main():
    # python classify_dvsg.py -T 16 -device cuda:0 -b 1 -epochs 128 -data-dir /datasets/DVSGesture/ -amp -opt adam -lr 0.005 -j 8 -channels 16 -T-train 12 -size 32 -ds 4

    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-T-train', default=12, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-opt', type=str, default='adam', help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.005, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    parser.add_argument('-size', default=128, type=int, help='input size')
    parser.add_argument('-ds', default=1, type=int, help='down sample number')
    parser.add_argument('-model', type=str, default='DVSGNet')
    parser.add_argument('-test', type=str, help='resume from the checkpoint path')


    args = parser.parse_args()
    print(args)

    if args.model == 'DVSGNet':
        net = DVSGNet(size=args.size, channels=args.channels, ds=args.ds)
    elif args.model == 'DVSGNet2':
        net = DVSGNet2(size=args.size, channels=args.channels, ds=args.ds)

    elif args.model == 'DVSGNetBN':
        net = DVSGNetBN(size=args.size, channels=args.channels, ds=args.ds)


    functional.set_step_mode(net, 'm')


    print(net)


    net.to(args.device)

    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')



    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )


    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    if args.model == 'DVSGNet':
        prefix = ''
    else:
        prefix = args.model + '_'
    out_dir = os.path.join(args.out_dir, f'{prefix}size{args.size}_ds_{args.ds}_T{args.T}_{args.T_train}_b{args.b}_e{args.epochs}_{args.opt}_lr{args.lr}_c{args.channels}')

    if args.amp:
        out_dir += '_amp'

    if args.test:
        checkpoint = torch.load(args.test, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])



    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    if not (args.test):
        for epoch in range(start_epoch, args.epochs):

            start_time = time.time()
            net.train()
            train_loss = 0
            train_acc = 0
            train_samples = 0
            for frame, label in train_data_loader:
                optimizer.zero_grad()
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                frame = encoder(frame, args.size)  #正常encode的输入应该是[T, N, C, H, W]的格式
                if args.T_train < args.T:
                    frame = random_temporal_delete(frame, args.T_train, batch_first=False)

                label = label.to(args.device)


                if scaler is not None:
                    with amp.autocast():
                        out_fr = net(frame).mean(0)
                        loss = F.cross_entropy(out_fr, label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)  #清空梯度
                    scaler.update()
                else:
                    print("no")
                    out_fr = net(frame).mean(0)
                    loss = F.cross_entropy(out_fr, label)
                    loss.backward()
                    optimizer.step() # 更新梯度

                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (out_fr.argmax(1) == label).float().sum().item()

                functional.reset_net(net)

            train_time = time.time()
            train_speed = train_samples / (train_time - start_time)
            train_loss /= train_samples
            train_acc /= train_samples

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            lr_scheduler.step()

            net.eval()
            test_loss = 0
            test_acc = 0
            test_samples = 0
            with torch.no_grad():
                for frame, label in test_data_loader:
                    frame = frame.to(args.device)
                    frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                    frame = encoder(frame, args.size)
                    label = label.to(args.device)
                    out_fr = net(frame).mean(0)
                    loss = F.cross_entropy(out_fr, label)
                    test_samples += label.numel()
                    test_loss += loss.item() * label.numel()
                    test_acc += (out_fr.argmax(1) == label).float().sum().item()
                    functional.reset_net(net)
            test_time = time.time()
            test_speed = test_samples / (test_time - train_time)
            test_loss /= test_samples
            test_acc /= test_samples
            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)

            save_max = False
            if test_acc > max_test_acc:
                max_test_acc = test_acc
                save_max = True

            checkpoint = {
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }

            if save_max:
                torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

            print(args)
            print(out_dir)
            print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
            print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
            print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

    if (args.test):

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)  #[1,64,2,128,128]
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]  [64,1,2,128,128]
                frame = encoder(frame, args.size) # [64,1,2,64,64]
                label = label.to(args.device)
                out_fr = net(frame).mean(0) #[1,11]
                loss = F.cross_entropy(out_fr, label)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()   #out_fr:(1,11)  ;out_fr.argmax(1) == label:(true)
                print("正确个数1：", test_acc)
                print("样本个数1：", test_samples)
                functional.reset_net(net)
        test_time = time.time()
        test_acc /= test_samples
        print("test_acc：",test_acc)

        net_ladl = net.to_lava()  # 这个net是conv和CubaLIFNode都有的
        print(net_ladl)
        with torch.no_grad():
            print(net_ladl(torch.rand([args.b, 2, 64, 64, args.T])).shape)  # [N, C, H, W, T],  [1,2,64,64,64]
        export_hdf5(net_ladl, './net_lava_dl.net')
        net_lava = netx.hdf5.Network(net_config='./net_lava_dl.net', input_shape=(64, 64, 2))  # 再读取这个权重文件,注意这里 input_shape=(64, 64, 2)后面改成2，因为两个channel
        print('成功读取！')
        net_ladl = net_ladl.to(args.device);

        test_loss = 0
        test_acc = 0
        test_samples = 0


        for frame, label in test_data_loader:
            with torch.no_grad():
                print(f'label = {label}')
                frame = frame.to(args.device)  # [1,64,2,128,128]ddf
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]  [64,1,2,128,128]
                frame = encoder(frame, args.size)  # [64,1,2,64,64] # 要求输入格式必须为NCHWT  #正常encode的输入应该是[T, N, C, H, W]的格式
                frame = frame.permute(1, 2, 3, 4, 0)  # [T, N, C, H, W]  -> 【N, C, H, W, T】
                label = label.to(args.device)
                out_fr = net_ladl(frame).mean(2)  # [1,11]  注意这里必须是mean（2）
                # out_fr = net_ladl(frame)  # [1,11]

                test_samples += label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                print("正确个数2：",test_acc)
                print("样本个数2：", test_samples)
                functional.reset_net(net)
                y = net_ladl(frame)   # 跑一遍网络[1,2,64,64,64]
                print('y(ladl)=', y.sum(-1).argmax().item())  # 投票，选最大


        test_time = time.time()
        test_acc /= test_samples
        print("final_test_acc：",test_acc)

        print('执行最终网络！')
        frame = frame.squeeze(0)
        x = frame.permute(2, 1, 0, 3)
        x = x.cpu().numpy()

        source = io.source.RingBuffer(data=x)  # 来自循环数据缓冲区的尖峰生成器进程  x:（64,64,2,64）
        sink = io.sink.RingBuffer(shape=(11,), buffer=args.T + 6)  # 将任意形状的数据接收到环形缓冲区的过程记忆力 用作探测的替代品  buffer：数据宿缓冲区的大小
          # shape这里应该是11，因为输出的是11类，就像MNIST输出的是10一样
        source.s_out.connect(net_lava.inp)  # 输出端口连接到对等进程的其他输入端口或其他输出端口,将此OutPort的父进程作为子进程的进程.应该和线程交互有关 shape:10
            #Shapes (64, 64, 2) and (64, 64, 1) are incompatible.
        net_lava.out.connect(sink.a_in)  # connect用于将OutPort连接到另一个进程的其他InPort或到其父进程的OutPort。
        run_condition = RunSteps(num_steps=args.T + 6,
                                )  # s设置时间步长   如果这里不加 blocking=False，则会在run—start阻塞.加上则会在这行后面output是none（改完gevent后都是none）
        run_config = Loihi1SimCfg(select_tag='fixed_pt')  # 设置运行配置

        net_lava.run(condition=run_condition, run_cfg=run_config)
        output = sink.data.get()
        net_lava.stop()
        print('最终网络执行成功，y(lava)=', output.sum(-1).argmax())
        exit()

if __name__ == '__main__':
    main()

# import torch
#
# print(torch.cuda.is_available())
# a=torch.Tensor([1,2])
# a=a.cuda()
# print(a)
# conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
