import os
import torch
from util import util
from config import config
from got10k.trackers import Tracker
from tracking.network import SiameseAlexNet, SiameseAlexNetMultimodal
from loss import rpn_smoothL1, rpn_cross_entropy_balance


class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, modality=1, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)

        '''setup GPU device if available'''
        self.cuda   = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.modality = modality

        '''setup model'''
        if self.modality == 1:
            self.net = SiameseAlexNet()
        else:
            self.net = SiameseAlexNetMultimodal()


        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location = lambda storage, loc: storage ))
        if self.cuda:
            self.net = self.net.to(self.device)

        '''setup optimizer'''
        self.optimizer   = torch.optim.SGD(
            self.net.parameters(),
            lr           = config.lr,
            momentum     = config.momentum,
            weight_decay = config.weight_decay)


    def step(self, epoch, dataset, anchors, i = 0,  train=True):

        if train:
            self.net.train()
        else:
            self.net.eval()


        template_rgb, detection_rgb, template_ir, detection_ir, regression_target, conf_target = dataset

        if self.cuda:
            template_rgb, detection_rgb = template_rgb.cuda(), detection_rgb.cuda()
            template_ir, detection_ir = template_ir.cuda(), detection_ir.cuda()
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()


        if self.modality == 1:
            pred_score, pred_regression = self.net(template_rgb, detection_rgb)
        else:
            pred_score, pred_regression = self.net(template_rgb, detection_rgb, template_ir, detection_ir)


        pred_conf   = pred_score.reshape(-1, 2, config.size).permute(0, 2, 1)

        pred_offset = pred_regression.reshape(-1, 4, config.size).permute(0, 2, 1)

        cls_loss = rpn_cross_entropy_balance(   pred_conf,
                                                conf_target,
                                                config.num_pos,
                                                config.num_neg,
                                                anchors,
                                                ohem_pos=config.ohem_pos,
                                                ohem_neg=config.ohem_neg)

        reg_loss = rpn_smoothL1(pred_offset,
                                regression_target,
                                conf_target,
                                config.num_pos,
                                ohem=config.ohem_reg)

        loss = cls_loss + config.lamb * reg_loss

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), config.clip)
            self.optimizer.step()

        return cls_loss, reg_loss, loss

    '''save model'''
    def save(self,model, exp_name_dir, epoch):
        util.adjust_learning_rate(self.optimizer, config.gamma)

        model_save_dir_pth = '{}/model'.format(exp_name_dir)
        if not os.path.exists(model_save_dir_pth):
                os.makedirs(model_save_dir_pth)
        net_path = os.path.join(model_save_dir_pth, 'model_e%d.pth' % (epoch + 1))
        torch.save(model.net.state_dict(), net_path)


#
# class TrackerSiamFC(Tracker):
#
#     def __init__(self, net_path=None, modality=1, **kwargs):
#
#         super(TrackerSiamFC, self).__init__('SiamFC', True)
#
#         # setup GPU device if available
#         self.cuda = torch.cuda.is_available()
#         self.device = torch.device('cuda:0' if self.cuda else 'cpu')
#
#         self.modality = modality
#
#         print('Training on device {}'.format(self.device))
#
#         '''setup GPU device if available'''
#         self.cuda = torch.cuda.is_available()
#         self.device = torch.device('cuda:0' if self.cuda else 'cpu')
#
#
#         # settings
#         self.cfg = {
#             # basic parameters
#             'out_scale': 0.001,
#             'exemplar_sz': 127,
#             'instance_sz': 255,
#             'context': 0.5,
#
#             # inference parameters
#             'scale_num': 3,
#             'scale_step': 1.0375,
#             'scale_lr': 0.59,
#             'scale_penalty': 0.9745,
#             'window_influence': 0.176,
#             'response_sz': 17,
#             'response_up': 16,
#             'total_stride': 8,
#
#             # train parameters
#             'r_pos': 16,
#             'r_neg': 0
#         }
#
#
#         # setup model
#         if self.modality == 1:
#             self.net = Net(
#                 backbone=AlexNetV3(3),
#                 head=SiamFC(self.cfg['out_scale']))
#
#
#         # load checkpoint if provided
#         if net_path is not None:
#             self.net.load_state_dict(torch.load(
#                 net_path, map_location=lambda storage, loc: storage))
#         self.net = self.net.to(self.device)
#
#
#         # setup criterion
#         from network_siamfc import BalancedLoss
#         self.criterion = BalancedLoss()
#
#         # setup optimizer
#         self.optimizer = torch.optim.SGD(
#             self.net.parameters(),
#             lr=config.lr,
#             momentum=config.momentum,
#             weight_decay=config.weight_decay)
#
#
#     def step(self, epoch, dataset, anchors, i = 0,  train=True):
#
#         if train:
#             self.net.train()
#         else:
#             self.net.eval()
#
#         template_rgb, detection_rgb, template_ir, detection_ir, test , target = dataset
#
#         # TODO: convert target???
#         target = self._create_labels(target)
#
#         if self.cuda:
#             template_rgb, detection_rgb = template_rgb.cuda(), detection_rgb.cuda()
#             template_ir, detection_ir = template_ir.cuda(), detection_ir.cuda()
#             target = target.cuda()
#
#
#         if self.modality == 1:
#             responses = self.net(template_rgb, detection_rgb)
#         else:
#             responses = self.net(template_rgb, detection_rgb, template_ir, detection_ir)
#
#
#         loss = self.criterion(responses, target)
#
#
#         if train:
#             self.optimizer.zero_grad()
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.net.parameters(), config.clip)
#             self.optimizer.step()
#
#         return loss
#
#
#     def _create_labels(self, size):
#
#         # skip if same sized labels already created
#         if hasattr(self, 'labels') and self.labels.size() == size:
#             return self.labels
#
#         def logistic_labels(x, y, r_pos, r_neg):
#             dist = np.abs(x) + np.abs(y)  # block distance
#             labels = np.where(dist <= r_pos,
#                               np.ones_like(x),
#                               np.where(dist < r_neg,
#                                        np.ones_like(x) * 0.5,
#                                        np.zeros_like(x)))
#             return labels
#
#         # distances along x- and y-axis
#         n, c, h, w = size
#         x = np.arange(w) - (w - 1) / 2
#         y = np.arange(h) - (h - 1) / 2
#         x, y = np.meshgrid(x, y)
#
#         # create logistic labels
#
#
#         r_pos = self.cfg['r_pos'] / self.cfg['total_stride']
#         r_neg = self.cfg['r_neg'] / self.cfg['total_stride']
#         labels = logistic_labels(x, y, r_pos, r_neg)
#
#         # repeat to size
#         labels = labels.reshape((1, 1, h, w))
#         labels = np.tile(labels, (n, c, 1, 1))
#
#         # convert to tensors
#         labels = torch.from_numpy(labels).to(self.device).float()
#
#         return labels

