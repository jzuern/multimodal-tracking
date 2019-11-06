# -*- coding: utf-8 -*-
import os
import sys
import cv2
import time
import torch
import random
import numpy as np
import os.path as osp
from util import util
from PIL import Image
from config import config
from torch.utils.data import Dataset
from got10k.datasets import ImageNetVID, GOT10k
from torchvision import datasets, transforms, utils
from got10k.datasets import ImageNetVID, GOT10k
from custom_transforms import Normalize, ToTensor, RandomStretch, \
    RandomCrop, CenterCrop, RandomBlur, ColorAug
import glob


class TrainDataLoaderRGBT(Dataset):
    def __init__(self, seq_dataset, z_transforms, x_transforms, name = 'RGBT-234'):

        self.max_inter     = config.max_inter
        self.z_transforms  = z_transforms
        self.x_transforms  = x_transforms
        self.seq_dataset = seq_dataset
        self.ret           = {}
        self.count         = 0
        self.index         = 3000
        self.name          = name
        self.anchors       = util.generate_anchors( config.anchor_total_stride,
                                                    config.anchor_base_size,
                                                    config.anchor_scales,
                                                    config.anchor_ratios,
                                                    config.score_size)


    def _pick_img_pairs(self, index_of_subclass):

        assert index_of_subclass < len(self.seq_dataset), 'index_of_subclass should less than total classes'


        img_files_rgb = sorted(glob.glob(os.path.join(self.seq_dataset.seq_dirs[index_of_subclass], 'visible','*.jpg')))
        img_files_ir = sorted(glob.glob(os.path.join(self.seq_dataset.seq_dirs[index_of_subclass], 'infrared', '*.jpg')))
        video_gt = np.loadtxt(self.seq_dataset.anno_files[index_of_subclass], delimiter=',')


        video_num  = len(video_gt)


        status = True

        while status:
            if self.max_inter >= video_num-1:
                self.max_inter = video_num//2

            template_index = np.clip(random.choice(range(0, max(1, video_num - self.max_inter))), 0, video_num-1)
            detection_index= np.clip(random.choice(range(1, max(2, self.max_inter))) + template_index, 0, video_num-1)

            template_img_rgb_path = img_files_rgb[template_index]
            detection_img_rgb_path = img_files_rgb[detection_index]

            template_img_ir_path = img_files_ir[template_index]
            detection_img_ir_path = img_files_ir[detection_index]

            template_gt  = video_gt[template_index]
            detection_gt = video_gt[detection_index]


            if template_gt[2]*template_gt[3]*detection_gt[2]*detection_gt[3] != 0:
                status = False
            else:
                print(  'index_of_subclass:', index_of_subclass, '\n',
                        'template_index:', template_index, '\n',
                        'template_gt:', template_gt, '\n',
                        'detection_index:', detection_index, '\n',
                        'detection_gt:', detection_gt, '\n')


        # load infomation of template and detection
        self.ret['template_img_rgb_path']      = template_img_rgb_path
        self.ret['detection_img_rgb_path']     = detection_img_rgb_path
        self.ret['template_img_ir_path']      = template_img_ir_path
        self.ret['detection_img_ir_path']     = detection_img_ir_path

        self.ret['template_target_x1y1wh'] = template_gt
        self.ret['detection_target_x1y1wh']= detection_gt

        t1, t2 = self.ret['template_target_x1y1wh'].copy(), self.ret['detection_target_x1y1wh'].copy()

        self.ret['template_target_xywh']   = np.array([t1[0]+t1[2]//2, t1[1]+t1[3]//2, t1[2], t1[3]], np.float32)
        self.ret['detection_target_xywh']  = np.array([t2[0]+t2[2]//2, t2[1]+t2[3]//2, t2[2], t2[3]], np.float32)
        self.ret['anchors'] = self.anchors
        #self._average()

    def open(self):

        '''template'''
        template_img_rgb = Image.open(self.ret['template_img_rgb_path'])
        template_img_rgb = np.array(template_img_rgb)

        template_img_ir = Image.open(self.ret['template_img_ir_path']).convert('L')
        template_img_ir = np.array(template_img_ir)

        detection_img_rgb = Image.open(self.ret['detection_img_rgb_path'])
        detection_img_rgb = np.array(detection_img_rgb)

        detection_img_ir = Image.open(self.ret['detection_img_ir_path']).convert('L')
        detection_img_ir = np.array(detection_img_ir)


        if np.random.rand(1) < config.gray_ratio:

            template_img_rgb = cv2.cvtColor(template_img_rgb, cv2.COLOR_RGB2GRAY)
            template_img_rgb = cv2.cvtColor(template_img_rgb, cv2.COLOR_GRAY2RGB)

            detection_img_rgb = cv2.cvtColor(detection_img_rgb, cv2.COLOR_RGB2GRAY)
            detection_img_rgb = cv2.cvtColor(detection_img_rgb, cv2.COLOR_GRAY2RGB)



        # normalize image
        template_img_rgb = template_img_rgb / 255.
        template_img_ir = template_img_ir / 255.
        detection_img_rgb = detection_img_rgb / 255.
        detection_img_ir = detection_img_ir / 255.


        img_rgb_mean = np.mean(template_img_rgb, axis=(0, 1))
        img_ir_mean = np.mean(template_img_ir, axis=(0, 1))


        exemplar_img_rgb, scale_z, s_z, w_x, h_x = self.get_exemplar_image( template_img_rgb,
                                                                        self.ret['template_target_xywh'],
                                                                        config.template_img_size,
                                                                        config.context, img_rgb_mean )

        exemplar_img_ir, _, _, _,_ = self.get_exemplar_image( template_img_ir,
                                                                        self.ret['template_target_xywh'],
                                                                        config.template_img_size,
                                                                        config.context, img_ir_mean )


        size_x = config.template_img_size
        x1, y1 = int((size_x + 1) / 2 - w_x / 2), int((size_x + 1) / 2 - h_x / 2)

        self.ret['exemplar_img_rgb'] = exemplar_img_rgb
        self.ret['exemplar_img_ir'] = exemplar_img_ir

        '''detection'''
        d = self.ret['detection_target_xywh']
        cx, cy, w, h = d  # float type

        wc_z = w + 0.5 * (w + h)
        hc_z = h + 0.5 * (w + h)
        s_z = np.sqrt(wc_z * hc_z)

        s_x = s_z / (config.detection_img_size//2)
        img_rgb_mean_d = tuple(map(int, detection_img_rgb.mean(axis=(0, 1))))
        img_ir_mean_d = int(detection_img_ir.mean())

        a_x_ = np.random.choice(range(-12,12))
        a_x = a_x_ * s_x

        b_y_ = np.random.choice(range(-12,12))
        b_y = b_y_ * s_x

        instance_img_rgb, a_x, b_y, w_x, h_x, scale_x = self.get_instance_image(  detection_img_rgb, d,
                                                                    config.template_img_size, # 127
                                                                    config.detection_img_size,# 255
                                                                    config.context,           # 0.5
                                                                    a_x, b_y,
                                                                    img_rgb_mean_d )


        instance_img_ir, _, _, _, _, _ = self.get_instance_image(  detection_img_ir, d,
                                                                    config.template_img_size, # 127
                                                                    config.detection_img_size,# 255
                                                                    config.context,           # 0.5
                                                                    a_x, b_y,
                                                                    img_ir_mean_d )

        size_x = config.detection_img_size

        x1, y1 = int((size_x + 1) / 2 - w_x / 2), int((size_x + 1) / 2 - h_x / 2)
        x2, y2 = int((size_x + 1) / 2 + w_x / 2), int((size_x + 1) / 2 + h_x / 2)

        w  = x2 - x1
        h  = y2 - y1

        self.ret['instance_img_rgb'] = instance_img_rgb
        self.ret['instance_img_ir'] = instance_img_ir
        self.ret['cx, cy, w, h'] = [int(a_x_), int(b_y_), w, h]



    def get_exemplar_image(self, img, bbox, size_z, context_amount, img_mean=None):
        cx, cy, w, h = bbox

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = size_z / s_z

        exemplar_img, scale_x = self.crop_and_pad_old(img, cx, cy, size_z, s_z, img_mean)

        w_x = w * scale_x
        h_x = h * scale_x

        return exemplar_img, scale_z, s_z, w_x, h_x

    def get_instance_image(self, img, bbox, size_z, size_x, context_amount, a_x, b_y, img_mean=None):

        cx, cy, w, h = bbox  # float type

        #cx, cy = cx - a_x , cy - b_y
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z) # the width of the crop box

        scale_z = size_z / s_z

        s_x = s_z * size_x / size_z
        instance_img, gt_w, gt_h, scale_x, scale_h, scale_w = self.crop_and_pad(img, cx, cy, w, h, a_x, b_y,  size_x, s_x, img_mean)
        w_x = gt_w #* scale_x #w * scale_x
        h_x = gt_h #* scale_x #h * scale_x

        #cx, cy = cx/ scale_w *scale_x, cy/ scale_h *scale_x
        #cx, cy = cx/ scale_w, cy/ scale_h
        a_x, b_y = a_x*scale_w, b_y*scale_h
        x1, y1 = int((size_x + 1) / 2 - w_x / 2), int((size_x + 1) / 2 - h_x / 2)
        x2, y2 = int((size_x + 1) / 2 + w_x / 2), int((size_x + 1) / 2 + h_x / 2)
        '''frame = cv2.rectangle(instance_img, (   int(x1+(a_x*scale_x)),
                                                int(y1+(b_y*scale_x))),
                                                (int(x2+(a_x*scale_x)),
                                                int(y2+(b_y*scale_x))),
                                                (0, 255, 0), 1)'''
        #cv2.imwrite('1.jpg', frame)
        return instance_img, a_x, b_y, w_x, h_x, scale_x

    def crop_and_pad(self, img, cx, cy, gt_w, gt_h, a_x, b_y, model_sz, original_sz, img_mean=None):

        #random = np.random.uniform(-0.15, 0.15)
        scale_h = 1.0 + np.random.uniform(-0.15, 0.15)
        scale_w = 1.0 + np.random.uniform(-0.15, 0.15)

        if len(img.shape) > 2:
            im_h, im_w, _ = img.shape
        else:
            im_h, im_w = img.shape

        xmin = (cx-a_x) - ((original_sz - 1) / 2)* scale_w
        xmax = (cx-a_x) + ((original_sz - 1) / 2)* scale_w

        ymin = (cy-b_y) - ((original_sz - 1) / 2)* scale_h
        ymax = (cy-b_y) + ((original_sz - 1) / 2)* scale_h

        #print('xmin, xmax, ymin, ymax', xmin, xmax, ymin, ymax)

        left   = int(self.round_up(max(0., -xmin)))
        top    = int(self.round_up(max(0., -ymin)))
        right  = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))

        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))


        if len(img.shape) > 2:
            r, c, k = img.shape
        else:
            r, c = img.shape
            k = 1



        if any([top, bottom, left, right]):

            if k > 1:
                te_im =  np.zeros((int((r + top + bottom)), int((c + left + right)), k), np.uint8)  # 0 is better than 1 initialization

                te_im[:, :, :] = img_mean
                te_im[top:top + r, left:left + c, :] = img

                if top:
                    te_im[0:top, left:left + c, :] = img_mean
                if bottom:
                    te_im[r + top:, left:left + c, :] = img_mean
                if left:
                    te_im[:, 0:left, :] = img_mean
                if right:
                    te_im[:, c + left:, :] = img_mean

                im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
            else:
                te_im = np.zeros((int((r + top + bottom)), int((c + left + right))),np.uint8)  # 0 is better than 1 initialization

                te_im[:, :] = img_mean
                te_im[top:top + r, left:left + c] = img

                if top:
                    te_im[0:top, left:left + c] = img_mean
                if bottom:
                    te_im[r + top:, left:left + c] = img_mean
                if left:
                    te_im[:, 0:left] = img_mean
                if right:
                    te_im[:, c + left:] = img_mean

                im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1)]

        else:
            if k > 1:
                im_patch_original = img[int(ymin):int((ymax) + 1), int(xmin):int((xmax) + 1), :]
            else:
                im_patch_original = img[int(ymin):int((ymax) + 1), int(xmin):int((xmax) + 1)]


        if not np.array_equal(model_sz, original_sz):

            if len(im_patch_original.shape) > 2:
                h, w, _ = img.shape
            else:
                h, w = img.shape

            if h < w:
                scale_h_ = 1
                scale_w_ = h/w
                scale = config.detection_img_size/h
            elif h > w:
                scale_h_ = w/h
                scale_w_ = 1
                scale = config.detection_img_size/w
            elif h == w:
                scale_h_ = 1
                scale_w_ = 1
                scale = config.detection_img_size/w

            gt_w = gt_w * scale_w_
            gt_h = gt_h * scale_h_

            gt_w = gt_w * scale
            gt_h = gt_h * scale

            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed

        else:
            im_patch = im_patch_original

        return im_patch, gt_w, gt_h, scale, scale_h_, scale_w_




    def crop_and_pad_old(self, img, cx, cy, model_sz, original_sz, img_mean=None):

        if len(img.shape) > 2:
            im_h, im_w, _ = img.shape
        else:
            im_h, im_w = img.shape

        xmin = cx - (original_sz - 1) / 2
        xmax = xmin + original_sz - 1
        ymin = cy - (original_sz - 1) / 2
        ymax = ymin + original_sz - 1

        left = int(self.round_up(max(0., -xmin)))
        top = int(self.round_up(max(0., -ymin)))
        right = int(self.round_up(max(0., xmax - im_w + 1)))
        bottom = int(self.round_up(max(0., ymax - im_h + 1)))

        xmin = int(self.round_up(xmin + left))
        xmax = int(self.round_up(xmax + left))
        ymin = int(self.round_up(ymin + top))
        ymax = int(self.round_up(ymax + top))


        if len(img.shape) > 2:
            r, c, k = img.shape
        else:
            r, c = img.shape
            k = 1



        if any([top, bottom, left, right]):

            if k > 1:
                te_im = np.zeros((r + top + bottom, c + left + right, k), np.uint8)  # 0 is better than 1 initialization
                te_im[top:top + r, left:left + c, :] = img
                if top:
                    te_im[0:top, left:left + c, :] = img_mean
                if bottom:
                    te_im[r + top:, left:left + c, :] = img_mean
                if left:
                    te_im[:, 0:left, :] = img_mean
                if right:
                    te_im[:, c + left:, :] = img_mean
                im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
            else:
                te_im = np.zeros((r + top + bottom, c + left + right), np.uint8)  # 0 is better than 1 initialization
                te_im[top:top + r, left:left + c] = img
                if top:
                    te_im[0:top, left:left + c] = img_mean
                if bottom:
                    te_im[r + top:, left:left + c] = img_mean
                if left:
                    te_im[:, 0:left] = img_mean
                if right:
                    te_im[:, c + left:] = img_mean
                im_patch_original = te_im[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1)]
        else:
            if k > 1:
                im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1), :]
            else:
                im_patch_original = img[int(ymin):int(ymax + 1), int(xmin):int(xmax + 1)]

        if not np.array_equal(model_sz, original_sz):

            im_patch = cv2.resize(im_patch_original, (model_sz, model_sz))  # zzp: use cv to get a better speed
        else:
            im_patch = im_patch_original
        scale = model_sz / im_patch_original.shape[0]

        return im_patch, scale

    def round_up(self, value):
        return round(value + 1e-6 + 1000) - 1000

    def _target(self):

        regression_target, conf_target = self.compute_target(self.anchors,
                                                             np.array(list(map(round, self.ret['cx, cy, w, h']))))

        # binary_target = self.compute_binary_target(self.anchors)

        return regression_target, conf_target

    def compute_target(self, anchors, box):

        # box = [-(box[0]), -(box[1]), box[2], box[3]]
        regression_target = self.box_transform(anchors, box)

        iou = self.compute_iou(anchors, box).flatten()

        pos_index = np.where(iou > config.pos_threshold)[0]
        neg_index = np.where(iou < config.neg_threshold)[0]
        label = np.ones_like(iou) * -1

        label[pos_index] = 1
        label[neg_index] = 0

        return regression_target, label

    def box_transform(self, anchors, gt_box):
        anchor_xctr = anchors[:, :1]
        anchor_yctr = anchors[:, 1:2]
        anchor_w = anchors[:, 2:3]
        anchor_h = anchors[:, 3:]
        gt_cx, gt_cy, gt_w, gt_h = gt_box

        target_x = (gt_cx - anchor_xctr) / anchor_w
        target_y = (gt_cy - anchor_yctr) / anchor_h
        target_w = np.log(gt_w / anchor_w)
        target_h = np.log(gt_h / anchor_h)
        regression_target = np.hstack((target_x, target_y, target_w, target_h))
        return regression_target

    def compute_iou(self, anchors, box):
        if np.array(anchors).ndim == 1:
            anchors = np.array(anchors)[None, :]
        else:
            anchors = np.array(anchors)
        if np.array(box).ndim == 1:
            box = np.array(box)[None, :]
        else:
            box = np.array(box)
        gt_box = np.tile(box.reshape(1, -1), (anchors.shape[0], 1))

        anchor_x1 = anchors[:, :1] - anchors[:, 2:3] / 2 + 0.5
        anchor_x2 = anchors[:, :1] + anchors[:, 2:3] / 2 - 0.5
        anchor_y1 = anchors[:, 1:2] - anchors[:, 3:] / 2 + 0.5
        anchor_y2 = anchors[:, 1:2] + anchors[:, 3:] / 2 - 0.5

        gt_x1 = gt_box[:, :1] - gt_box[:, 2:3] / 2 + 0.5
        gt_x2 = gt_box[:, :1] + gt_box[:, 2:3] / 2 - 0.5
        gt_y1 = gt_box[:, 1:2] - gt_box[:, 3:] / 2 + 0.5
        gt_y2 = gt_box[:, 1:2] + gt_box[:, 3:] / 2 - 0.5

        xx1 = np.max([anchor_x1, gt_x1], axis=0)
        xx2 = np.min([anchor_x2, gt_x2], axis=0)
        yy1 = np.max([anchor_y1, gt_y1], axis=0)
        yy2 = np.min([anchor_y2, gt_y2], axis=0)

        inter_area = np.max([xx2 - xx1, np.zeros(xx1.shape)], axis=0) * np.max([yy2 - yy1, np.zeros(xx1.shape)],
                                                                               axis=0)
        area_anchor = (anchor_x2 - anchor_x1) * (anchor_y2 - anchor_y1)
        area_gt = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
        iou = inter_area / (area_anchor + area_gt - inter_area + 1e-6)
        return iou

    def _tranform(self):

        self.ret['train_x_transforms_rgb'] = self.x_transforms(self.ret['instance_img_rgb'])
        self.ret['train_z_transforms_rgb'] = self.z_transforms(self.ret['exemplar_img_rgb'])
        self.ret['train_x_transforms_ir'] = self.x_transforms(self.ret['instance_img_ir'])
        self.ret['train_z_transforms_ir'] = self.z_transforms(self.ret['exemplar_img_ir'])


    def __getitem__(self, index):

        index = random.choice(range(len(self.seq_dataset)))

        if self.name == 'GOT-10k':
            if index == 4418 or index == 8627 or index == 8629 or index == 9057 or index == 9058:
                index += 3

        self._pick_img_pairs(index)
        self.open()
        self._tranform()
        regression_target, conf_target = self._target()
        self.count += 1

        return self.ret['train_z_transforms_rgb'], \
               self.ret['train_x_transforms_rgb'], \
               self.ret['train_z_transforms_ir'], \
               self.ret['train_x_transforms_ir'], \
               regression_target, \
               conf_target.astype(np.int64)


    def __len__(self):
        return config.train_epoch_size*64
