from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd_I3N import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import pdb
from tensorboardX import SummaryWriter
import torch.nn.functional as F

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
#
parser.add_argument('--name', default='voc2clipart',type=str)
parser.add_argument('--dataset', default='VOC',
                    choices=['VOC'],
                    type=str)
parser.add_argument('--dataset_target', default='clipart',
                    choices=['clipart', 'water', 'comic'],
                    type=str)
parser.add_argument('--dcbr_weight', default=0.05, type=float)
parser.add_argument('--gcr_weight', default=1, type=float)
parser.add_argument('--kl_weight', default=0.1, type=float)
parser.add_argument('--pa_list', default=[1, 3],
                    help='value: 0, 1, 2, 3, 4, 5',
                    type=list)
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--gamma_fl', default=5, type=float,
                    help='Gamma for focal loss')
### resume
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--end_epoch', default=70, type=int)
parser.add_argument('--lr_epoch', default=[51])
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
### learning rate
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
###
parser.add_argument('--disp_interval', default=20, type=int,
                    help='Number of iterations to display')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    help='Use CUDA to train model')
parser.add_argument('--basenet', default='./weights/vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


args.save_folder  = os.path.join(args.save_folder, args.name)
if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)

logger = SummaryWriter("logs")

def train():

    if args.dataset == 'VOC':
        cfg = voc
        dataset = VOCDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset == 'cs':
        cfg = cs
        dataset = CSDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))

    if args.dataset_target == 'clipart':
        dataset_t = CLPDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset_target == 'water':
        dataset_t = WATERDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))
    elif args.dataset_target == 'comic':
        dataset_t = COMICDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))

    ssd_net = build_ssd('train', cfg, cfg['min_dim'], cfg['num_classes'], args.pa_list)
    net = ssd_net

    FL = FocalLoss(class_num=2, gamma=args.gamma_fl)

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()

    step_per_epoch = len(dataset) // args.batch_size

    print('The number of dataset %s is %d' % (args.dataset, len(dataset)))
    print('The number of target dataset %s is %d' % (args.dataset_target, len(dataset_t)))
    print('Using the specified args:')
    print(args)
    print('Loading the dataset...')
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    data_loader_t = data.DataLoader(dataset_t, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)

    img_per_epoch = step_per_epoch * args.batch_size
    old_state = torch.zeros(cfg['num_classes']-1)
    for epoch in range(args.start_epoch, args.end_epoch+1):
        epoch_time = time.time()
        if epoch in args.lr_epoch:
            adjust_learning_rate(optimizer, args.gamma)

        all_loss = 0
        reg_loss = 0
        cls_loss = 0
        gpa_loss = 0
        grc_loss = 0
        gf_loss = 0
        start_time = time.time()
        batch_iterator = iter(data_loader)
        batch_iterator_t = iter(data_loader_t)
        new_state = torch.zeros(cfg['num_classes']-1)
        for iteration in range(1, step_per_epoch+1):
            if epoch == 1 and iteration <= 160:
                warm_up_lr(optimizer, iteration, 160)
            # load train data
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets = next(batch_iterator)
            try:
                images_t, targets_t = next(batch_iterator_t)
            except StopIteration:
                batch_iterator_t = iter(data_loader_t)
                images_t, targets_t = next(batch_iterator_t)

            cls_onehot = gt_classes2cls_onehot(targets, cfg['num_classes'] - 1)  # shape = [bs, num_classes]
            cls_onehot = Variable(torch.from_numpy(cls_onehot).cuda())

            images = Variable(images.cuda())
            targets = [Variable(ann.cuda()) for ann in targets]
            images_t = Variable(images_t.cuda())

            optimizer.zero_grad()
            #### forward
            # source domain
            out, domain_g, domain_l, fea_lists, gcr_pre, global_feat, _ = net(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            # target domain
            _, domain_g_t, domain_l_t, fea_lists_t, gcr_pre_t, global_feat_t, loss_kl = net(images_t, target = True)
            loss_kl *= args.kl_weight
            #### calculate new state and get w
            # new state
            ind_max_cls = torch.argmax(gcr_pre_t.detach(), 1)
            for i in ind_max_cls:
                new_state[i] += 1
            # w1
            w1 = dcbr_w1_weight(gcr_pre_t.sigmoid().detach())
            # w2
            w2 = torch.exp(1 - old_state[ind_max_cls]/img_per_epoch)
            weight = (w1+w2)*0.5 if epoch > 20 else torch.ones(w1.size(0))# [bs]

            ################## domain adaptation loss  ##################
            ###### source
            ## local
            dloss_l = 0.5 * torch.mean(domain_l ** 2)
            ## global
            # focal loss
            # domain_s = Variable(torch.zeros(domain_g.size(0)).long().cuda())
            # dloss_g = 0.5 * FL(domain_g, domain_s)
            # weighted ce loss
            dloss_g = 0.5 * weight_ce_loss(domain_g, 0, torch.ones(domain_g.size(0))) * 0.1

            ###### target
            ## local
            dloss_l_t = 0.5 * torch.mean((1-domain_l_t) ** 2)
            ## global
            # focal loss
            # domain_s_t = Variable(torch.ones(domain_g_t.size(0)).long().cuda())
            # dloss_g_t = 0.5 * FL(domain_g_t, domain_s_t)
            # weighted ce loss
            dloss_g_t = 0.5 * weight_ce_loss(domain_g_t, 1, weight) * args.dcbr_weight

            ###### gf : global feat loss
            loss_gf = 38 * torch.pow(global_feat-global_feat_t, 2.0).mean()
            loss += loss_gf
            ###### gcr loss
            loss_gcr = nn.BCEWithLogitsLoss()(gcr_pre, cls_onehot) * args.gcr_weight
            loss += loss_gcr

            ###### pa
            if epoch > 20:
                loss_gpa = get_pa_losses(fea_lists, fea_lists_t)
                loss += loss_gpa
                #
                loss += loss_kl
            ################## domain adaptation loss  ##################

            ### backward
            loss += (dloss_g + dloss_g_t + dloss_l + dloss_l_t)
            loss.backward()
            optimizer.step()
            #
            all_loss += loss.item()
            reg_loss += loss_l.item()
            cls_loss += loss_c.item()
            grc_loss += loss_gcr.item()
            gf_loss += loss_gf.item()
            if epoch > 20 and loss_gpa:
                gpa_loss += loss_gpa.item()

            if iteration % args.disp_interval == 0:
                ## display
                all_loss /= args.disp_interval
                reg_loss /= args.disp_interval
                cls_loss /= args.disp_interval
                gpa_loss /= args.disp_interval
                grc_loss /= args.disp_interval
                gf_loss /= args.disp_interval

                det_loss = reg_loss+cls_loss
                end_time = time.time()

                get_lr = optimizer.param_groups[0]['lr']
                print('[epoch %2d][iter %4d/%4d]|| Loss: %.4f || lr: %.2e || Time: %.2f sec'
                      % (epoch, iteration, step_per_epoch, all_loss, get_lr, end_time - start_time))
                print('\t det_loss: %.4f || reg_loss: %.4f || cls_loss: %.4f || la_ga_loss: %.4f || gpa_loss: %.4f || gcr_loss: %.4f || gf_loss: %.6f' %
                      (det_loss, reg_loss, cls_loss, all_loss - det_loss - gpa_loss - grc_loss - gf_loss, gpa_loss, grc_loss, gf_loss))
                ## log
                info = {
                    'all_loss': all_loss,
                    'det_loss': det_loss,
                    'reg_loss': reg_loss,
                    'cls_loss': cls_loss,
                    'la_ga_loss' : all_loss - det_loss -gpa_loss - grc_loss - gf_loss,
                    'gpa_loss': gpa_loss,
                    'gcr_loss': grc_loss,
                    'gf_loss': gf_loss,
                }
                logger.add_scalars(args.name, info, iteration+(epoch-1)*step_per_epoch)
                ## reset
                all_loss = 0
                reg_loss = 0
                cls_loss = 0
                gpa_loss = 0
                grc_loss = 0
                gf_loss = 0
                start_time = time.time()

        old_state = new_state
        print('This epoch cost %.4f sec'%(time.time()-epoch_time))

        if (epoch % 10 == 0) or (epoch in [54, 58, 62, 66, 70]):
            save_pth = os.path.join(args.save_folder, str(epoch)+'.pth')
            print('Saving state', save_pth)
            torch.save(ssd_net.state_dict(), save_pth)


def adjust_learning_rate(optimizer, gamma):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma

def warm_up_lr(optimizer, cur_step, max_step):
    if cur_step == 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / max_step
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * cur_step /(cur_step - 1)

def xavier(param):
    init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def get_gpa_losses(fea_lists, fea_lists_t):
    # compute intra loss
    loss = 0
    for i in range(len(fea_lists)):
        fea_list = fea_lists[i]
        fea_list_t = fea_lists_t[i]
        loss += get_gpa_loss(fea_list, fea_list_t)
    return loss

def get_gpa_loss(fea_list, fea_list_t):
    # compute intra loss
    assert len(fea_list) == len(fea_list_t), " fea_list size wrong"

    intra_loss = 0
    cnt = 0
    for (fea, fea_t) in zip(fea_list, fea_list_t):
        if fea.numel() and fea_t.numel():
            intra_loss += torch.pow(fea-fea_t, 2.0).mean()
            cnt += 1
    if cnt:
        intra_loss /= cnt
    return intra_loss

def get_pa_losses(fea_lists, fea_lists_t):
    # compute intra and inter loss
    loss = 0
    for i in range(len(fea_lists)):
        fea_list = fea_lists[i]
        fea_list_t = fea_lists_t[i]
        loss += get_pa_loss(fea_list, fea_list_t)
    return loss

def get_pa_loss(fea_list, fea_list_t):
    # compute intra and inter loss

    ########intre loss
    intra_loss = 0
    cnt = 0
    for (fea, fea_t) in zip(fea_list, fea_list_t):
        if fea.numel() and fea_t.numel():
            intra_loss += torch.pow(fea-fea_t, 2.0).mean()
            cnt += 1
    if cnt:
        intra_loss /= cnt

    ######## inter loss
    inter_loss = 0
    cnt = 0
    cls_num = len(fea_list) #20
    for i in range(cls_num):
        src_1 = fea_list[i]
        tgt_1 = fea_list_t[i]
        for j in range(i+1, cls_num):
            src_2 = fea_list[j]
            tgt_2 = fea_list_t[j]

            if src_1.numel():
                if src_2.numel():
                    inter_loss += contrasive_separation(src_1, src_2)
                    cnt += 1
                if tgt_2.numel():
                    inter_loss += contrasive_separation(src_1, tgt_2)
                    cnt += 1
            if tgt_1.numel():
                if src_2.numel():
                    inter_loss += contrasive_separation(tgt_1, src_2)
                    cnt += 1
                if tgt_2.numel():
                    inter_loss += contrasive_separation(tgt_1, tgt_2)
                    cnt += 1
    if cnt:
        inter_loss /= cnt

    return intra_loss + inter_loss

def contrasive_separation(f1, f2):
    dis = torch.pow(f1-f2, 2.0).mean().sqrt()
    loss = torch.pow(torch.max(1 - dis, torch.tensor(0).float().cuda()),2.0)
    loss *= torch.pow(1-dis, 2.0)
    return loss


def gt_classes2cls_onehot(targets, classed_num = 20):
    bs = len(targets)
    cls_onehot = np.zeros((bs, classed_num), np.float32)
    for i in range(bs):
        target = targets[i].numpy()
        for one_target in target:
            cls_onehot[i, one_target[-1].astype(np.int32)] = 1
    return cls_onehot

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,sigmoid=False,reduce=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce
    def forward(self, inputs, targets):
        N = inputs.size(0) # N means batch size
        # print(N)
        C = inputs.size(1) # C means the number of classes
        if self.sigmoid:
            P = F.sigmoid(inputs)
            #F.softmax(inputs)
            if targets == 0:
                probs = 1 - P#(P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            #inputs = F.sigmoid(inputs)
            P = F.softmax(inputs, dim = 1).clamp(1e-6,1) # [N, C]

            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1) # [N, 1]
            class_mask.scatter_(1, ids.data, 1.) # [N, C]
            # print(class_mask)


            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1) # [N, 1]

            log_p = probs.log() # [N, 1]
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p # [N, 1]
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def weight_ce_loss(pre, target, weight):
    # target = 0/1
    # pre.shape = [bs, 2]
    # weight.shape = [bs]
    pre_softmax = F.softmax(pre, dim = 1).clamp(1e-6,1) # [bs, 2]
    #
    pre_softmax = pre_softmax[:, target] # [bs]
    loss = -weight*torch.log(pre_softmax)
    return loss.mean()

def dcbr_w1_weight(cls_pre):
    # cls_pre.shape = [bs, num_classes]
    ind = (cls_pre > 0.5).float()
    cls_pre_valid = cls_pre * ind
    w1 = cls_pre_valid.sum(1)/(ind.sum(1)+1e-7) + 1
    return w1


def self_entropy(prob):
    # prob.size() = [bs, C(2)]   C: Number of categories
    log_prob = torch.log(prob)
    H = - torch.sum(prob * log_prob, dim = 1) # [bs]
    return H

if __name__ == '__main__':
    train()
    logger.close()
    # pdb.set_trace()
    print("Bye...")
