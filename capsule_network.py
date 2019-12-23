"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation by Kenta Iwasaki @ Gram.AI.
"""


import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

BATCH_SIZE = 10
NUM_CLASSES = 10
NUM_EPOCHS = 5
NUM_ROUTING_ITERATIONS = 3


def softmax(input, dim=1):
	'''

	'''
	# 先将dim和最后一个维度交换
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    # contiguous 是将转置后的数据局转换成连续内存，然后才可以进行view
    # 然后转换成-1 *channels 的形状
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
	'''
	数据增强
	'''    
    _, _, height, width = x.size()
	# 产生切片索引
    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)
    # 从原图中扣出来一大部分，然后转移到一个中的一个随机部分
    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes# [-1,32*6*6]
        self.num_iterations = num_iterations # 3 超参数
        self.num_capsules = num_capsules# [8,10]

        
        # 这个是digit,返回形状num_capsules, num_route_nodes, in_channels, out_channels
        if num_route_nodes != -1:#[10,32*6*6,8,16]
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:# 这个是primary,第一次有8个胶囊，分出来8个二维卷积函数。# 输入256，输出32通道。
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
    	'''
		((||s||)**2/(1+(||s||**2)))*(s/||s||)
        # 对于primay ,最后一个维度是胶囊个数。
    	'''
    	# 对最后一个维度求和，又变成一个胶囊的大小，所以可以对应相除？
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)

    def forward(self, x):
        if self.num_route_nodes != -1:# digit
            # 叉乘。x-->[b,32*6*6,8]
            # 增加维度，x[1,b,32*6*6,1,8]@[10,1,32*6*6,8,16]，需要关注一下。
            # 最后两个维度可以做叉乘，其他的维度相同，只在最后两个维度叉乘吗？
            #uj|i=wij(权重)*Ui(上一层输出)
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]
            # 初始化bij，全0向量
            logits = Variable(torch.zeros(*priors.size())).cuda()
            # 权重更新逻辑。
            for i in range(self.num_iterations):# 
                # cij=softmax(bij)
                probs = softmax(logits, dim=2)
                # 计算输出，第二个维度是啥？
                # sj=cij*uj|i
                # vj=squash(sj)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))
                # 如果不是循环的最后一次，更新logits
                if i != self.num_iterations - 1:
                    # 然后按照最后一个维度求和。
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:# primary
        	# 每一个胶囊都是一个二维卷积，通道256-->32,卷积后改变形状-->[b,c*h*w,1]
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            #从最后一个维度连接起来，[b,c*h*w,8]
            outputs = torch.cat(outputs, dim=-1)
            # 对最后一个维度保持形状压缩，然后除以该值
            outputs = self.squash(outputs)

        return outputs


class CapsuleNet(nn.Module):
    def __init__(self):
        super(CapsuleNet, self).__init__()
        # x-->[b,256,h,w]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        #[b,c*h*w,8]-->[b,32*6*6,8]
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        # 
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=False)

    def forward(self, images, labels, classes, reconstructions):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)


if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    from torchvision.utils import make_grid
    from torchvision.datasets.mnist import MNIST
    from tqdm import tqdm
    import torchnet as tnt

    model = CapsuleNet()
    model.load_state_dict(torch.load('epochs/epoch_1.pt'))
    model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_error_logger = VisdomPlotLogger('line', opts={'title': 'Train Accuracy'})
    test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'})
    test_accuracy_logger = VisdomPlotLogger('line', opts={'title': 'Test Accuracy'})
    confusion_logger = VisdomLogger('heatmap', opts={'title': 'Confusion matrix',
                                                     'columnnames': list(range(NUM_CLASSES)),
                                                     'rownames': list(range(NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', opts={'title': 'Reconstruction'})

    capsule_loss = CapsuleLoss()


    def get_iterator(mode):
        dataset = MNIST(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])

        return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)


    def processor(sample):
        data, labels, training = sample

        data = augmentation(data.unsqueeze(1).float() / 255.0)
        labels = torch.LongTensor(labels)

        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()

        engine.test(processor, get_iterator(False))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

        # Reconstruction visualization.

        test_sample = next(iter(get_iterator(False)))

        ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)
        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data

        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
        reconstruction_logger.log(
            make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=NUM_EPOCHS, optimizer=optimizer)

