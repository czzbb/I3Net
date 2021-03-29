import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from layers import *
from data import voc, coco
import os
import math

class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        #pdb.set_trace()
        return (grad_output * -self.lambd)
def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)
def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

class netD_pixel(nn.Module):
    def __init__(self,context=False):
        super(netD_pixel, self).__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.context:
          feat = F.avg_pool2d(x, (x.size(2), x.size(3)))
          x = self.conv3(x)
          return F.sigmoid(x),feat
        else:
          x = self.conv3(x)
          return F.sigmoid(x)

class netD(nn.Module):
    def __init__(self,context=False):
        super(netD, self).__init__()
        self.conv1 = conv3x3(512, 256, stride=2)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = conv3x3(256, 128, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = conv3x3(128, 128, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(128,2)
        self.context = context
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    def forward(self, x):
        x = F.dropout(F.relu(self.bn1(self.conv1(x))),training=self.training)
        x = F.dropout(F.relu(self.bn2(self.conv2(x))),training=self.training)
        x = F.dropout(F.relu(self.bn3(self.conv3(x))),training=self.training)
        x = F.avg_pool2d(x,(x.size(2),x.size(3)))
        x = x.view(-1,128)
        if self.context:
          feat = x
        x = self.fc(x)
        if self.context:
          return x,feat
        else:
          return x

class net_gcr(nn.Module):
    def __init__(self, out_channels):
        super(net_gcr, self).__init__()
        self.conv1 = conv3x3(512, 512, stride=2)
        self.conv2 = conv3x3(512, 256, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv3 = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)
        self._init_weights()
    def _init_weights(self):
      def normal_init(m, mean, stddev, truncated=False):
        """
        weight initalizer: truncated normal and random normal.
        """
        # x is a parameter
        if truncated:
          m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
        else:
          m.weight.data.normal_(mean, stddev)
          #m.bias.data.zero_()
      normal_init(self.conv1, 0, 0.01)
      normal_init(self.conv2, 0, 0.01)
      normal_init(self.conv3, 0, 0.01)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.avg_pool(x)
        x = self.conv3(x).squeeze(-1).squeeze(-1)
        return x

class net_gcr_simple(nn.Module):
    def __init__(self, out_channels):
        super(net_gcr_simple, self).__init__()
        self.conv1 = nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x).squeeze(-1).squeeze(-1)
        return x

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)# 2
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes, cfg, pa_list):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes # 21
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward()) #self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        self.pa_list = pa_list
        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        self.netD = netD()
        self.netD_pixel = netD_pixel()

        self.conv_gcr = net_gcr_simple(num_classes-1)

        self.RandomLayer = RandomLayer([512, num_classes*4], 1024)
        self.RandomLayer.cuda()
        self.softmax = nn.Softmax(dim=-1)
        self.fea_lists = [[torch.tensor([]) for _ in range(num_classes-1)] for _ in range(len(pa_list))]

        if phase == 'test':
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x, target = False):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(14):
            x = self.vgg[k](x)

        ## local
        if self.phase != 'test':
            domain_l = self.netD_pixel(grad_reverse(x))

        for k in range(14, 23):
            x = self.vgg[k](x)
        ## global
        if self.phase != 'test':
            domain_g = self.netD(grad_reverse(x))

        # gcr
        gcr_pre = self.conv_gcr(x)

        # for get global feature
        feat1 = x

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        fea_lists = []
        pre_lists = []
        # apply multibox head to source layers
        for i, (x, l, c) in enumerate(zip(sources, self.loc, self.conf)):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())

            conf_one = c(x).permute(0, 2, 3, 1).contiguous()
            conf.append(conf_one)
            if self.phase != 'test' and i in self.pa_list:
                fea_list = self.get_fea_list(x.permute(0, 2, 3, 1).contiguous(), conf_one, self.num_classes)
                fea_lists.append(fea_list)
                pre_lists.append(conf_one)
            if self.phase != 'test' and i == 0:
                feat2 = conf_one
                g_feat = self.get_feature_vector(feat1, feat2.detach()) 
        self.Moving_average(fea_lists)
        loss_kl = self.get_kl_loss(pre_lists) if target else torch.tensor(0)

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
            return output
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
            return output, domain_g, domain_l, self.fea_lists, gcr_pre, g_feat, loss_kl

    def get_kl_loss(self, pre_lists):
        kl_lists = []
        for pre in pre_lists:
            pre = pre.view(-1, pre.size(-1))/2
            pre = pre.view(pre.size(0), -1, self.num_classes)
            pre = self.softmax(pre).mean(1)
            _, max_ind = torch.max(pre, -1)
            kl_list = []
            for i in range(1, self.num_classes):
                max_ind_i = (max_ind == i)
                if  pre[max_ind_i].numel():
                    kl_list.append(pre[max_ind_i].mean(0)+1e-6)
                else:
                    kl_list.append(torch.tensor([]))
            #
            kl_lists.append(kl_list)
        #
        loss = torch.tensor(0).float().cuda()
        cnt = 0
        p_list, q_list = kl_lists
        for i in range(self.num_classes-1):
            p, q = p_list[i], q_list[i]
            if p.numel() and q.numel():
                tmp = p*torch.log(p/q) + q*torch.log(q/p)
                loss += tmp.mean()/2
                cnt += 1
        if cnt:
            loss /= cnt
        return loss


    def Moving_average(self, cur_fea_lists):
        for i in range(len(cur_fea_lists)):
            for j in range(len(cur_fea_lists[0])):
                if cur_fea_lists[i][j].numel():
                    if self.fea_lists[i][j].numel():
                        self.fea_lists[i][j] = self.fea_lists[i][j].detach()*0.7 + cur_fea_lists[i][j]*0.3
                    else:
                        self.fea_lists[i][j] = cur_fea_lists[i][j]
                else:
                    self.fea_lists[i][j] = self.fea_lists[i][j].detach()

    def get_feature_vector(self, f1, f2):

        bs = f1.size(0)
        f1 = f1.permute(0, 2, 3, 1).contiguous()
        f1 = f1.view(-1, f1.size(-1))
        f2 = f2.view(-1, f2.size(-1))
        f2 = f2.view(f2.size(0), -1, self.num_classes)
        f2 = self.softmax(f2).view(f2.size(0), -1)
        feat = self.RandomLayer([f1, f2])
        feat = feat.pow(2).mean(1)
        feat = feat.view(bs, -1)
        feat = F.normalize(feat, p=2, dim = 1).mean(0)

        return feat


    def get_fea_list(self, fea, pre, num_classes):
        fea = fea.view(-1, fea.size(-1))
        pre = pre.view(-1, pre.size(-1))
        _, max_ind = torch.max(pre, -1)
        max_ind %= num_classes
        fea_list = []
        for i in range(1, num_classes):
            max_ind_i = (max_ind == i)
            if fea[max_ind_i].numel():
                fea_list.append(fea[max_ind_i].mean(0))
            else:
                fea_list.append(torch.tensor([]))
        return fea_list


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, cfg, size=300, num_classes=21, pa_list = []):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes, cfg, pa_list)
