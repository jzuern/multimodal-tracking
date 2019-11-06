import cv2
import torch
import numpy as np
from tracking.util import util
import torch.nn.functional as F
import torchvision.transforms as transforms
from train.custom_transforms import ToTensor
from tracking.config import config
from got10k.trackers import Tracker
from PIL import Image
import time
import matplotlib.pyplot as plt
from tracking.network import SiameseAlexNet, SiameseAlexNetMultimodal
from tracking.data_loader import TrackerRGBTDataLoader


def show_frame(img_ir,
               img_rgb,
               tracker_name,
               plotter,
               boxes,
               box_fmt='ltwh',
               colors=None,
               thickness=3,
               fig_n=1,
               delay=1,
               visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):



    # if cvt_code is not None:
    #     img_ir = cv2.cvtColor(img_ir, cvt_code)
    #     img_rgb = cv2.cvtColor(img_rgb, cvt_code)

    # resize img if necessary
    max_size = 960

    if max(img_rgb.shape[:2]) > max_size:
        scale = max_size / max(img_rgb.shape[:2])
        out_size = (
            int(img_rgb.shape[1] * scale),
            int(img_rgb.shape[0] * scale))
        img_rgb = cv2.resize(img_rgb, out_size)

        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale


    if max(img_ir.shape[:2]) > max_size:
        scale = max_size / max(img_ir.shape[:2])
        out_size = (
            int(img_ir.shape[1] * scale),
            int(img_ir.shape[0] * scale))
        img_ir = cv2.resize(img_ir, out_size)



    assert box_fmt in ['ltwh', 'ltrb']
    boxes = np.array(boxes, dtype=np.int32)

    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)
    if box_fmt == 'ltrb':
        boxes[:, 2:] -= boxes[:, :2]

    # clip bounding boxes
    bound = np.array(img_rgb.shape[1::-1])[None, :]
    boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
    boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

    if colors is None:
        colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
            (0, 0, 128),
            (0, 128, 0),
            (128, 0, 0),
            (0, 128, 128),
            (128, 0, 128),
            (128, 128, 0)]

    colors = np.array(colors, dtype=np.int32)
    if colors.ndim == 1:
        colors = np.expand_dims(colors, axis=0)

    for i, box in enumerate(boxes):
        color = colors[i]
        pt1 = (box[0], box[1])
        pt2 = (box[0] + box[2], box[1] + box[3])

        # add boxes to image
        img_rgb = cv2.rectangle(img_rgb, pt1, pt2, color.tolist(), thickness)
        img_ir = cv2.rectangle(img_ir, pt1, pt2, color.tolist(), thickness)

    if visualize:

        if plotter is None:
            fig, axarr = plt.subplots(1, 2)
            plotter = (fig, axarr)
        else:
            plotter[1][0].clear()
            plotter[1][1].clear()

        plotter[1][0].imshow(img_rgb)
        plotter[1][0].set_title('RGB Image')
        plotter[1][0].plot(0, 0, "-", c='red', label=tracker_name)
        plotter[1][0].plot(0, 0, "-", c='cyan', label='Ground Truth')
        plotter[1][0].legend()

        plotter[1][1].imshow(img_ir)
        plotter[1][1].set_title('IR Image')
        plotter[1][1].plot(0, 0, "-", c='red', label=tracker_name)
        plotter[1][1].plot(0, 0, "-", c='cyan', label='Ground Truth')
        plotter[1][1].legend()

        plt.pause(0.01)
        plt.draw()

        return plotter

    return img_rgb



class TrackerSiamRPNEval(Tracker):

    def __init__(self, modality=1, model_path=None, **kargs):

        super(TrackerSiamRPNEval, self).__init__(name='SiamRPN', is_deterministic=True)

        self.modality = modality

        if modality == 1:
            self.model = SiameseAlexNet()
        else:
            self.model = SiameseAlexNetMultimodal()


        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        checkpoint = torch.load(model_path, map_location = self.device)

        if 'model' in checkpoint.keys():
            self.model.load_state_dict(torch.load(model_path, map_location = self.device)['model'])
        else:
            self.model.load_state_dict(torch.load(model_path, map_location = self.device))


        if self.cuda:
            self.model = self.model.cuda()

        self.model.eval()

        self.transforms = transforms.Compose([
            ToTensor()
        ])

        valid_scope = 2 * config.valid_scope + 1
        self.anchors = util.generate_anchors(   config.total_stride,
                                                config.anchor_base_size,
                                                config.anchor_scales,
                                                config.anchor_ratios,
                                                valid_scope)

        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()


        self.data_loader = TrackerRGBTDataLoader()


    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window


    def init(self, frame_rgb, frame_ir, bbox):

        """ initialize tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        frame_rgb = np.asarray(frame_rgb)
        frame_ir = np.asarray(frame_ir)


        self.pos = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2])  # center x, center y, zero based
        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
        self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]])

        self.origin_target_sz = np.array([bbox[2], bbox[3]])

        self.img_rgb_mean = np.mean(frame_rgb, axis=(0, 1))
        self.img_ir_mean = np.mean(frame_ir)


        exemplar_img_rgb, _, _ = self.data_loader.get_exemplar_image(frame_rgb,
                                                                    self.bbox,
                                                                    config.template_img_size,
                                                                    config.context_amount,
                                                                    self.img_rgb_mean)

        exemplar_img_ir, _, _ = self.data_loader.get_exemplar_image(frame_ir,
                                                                    self.bbox,
                                                                    config.template_img_size,
                                                                    config.context_amount,
                                                                    self.img_ir_mean)
        # get exemplar feature
        exemplar_img_rgb = self.transforms(exemplar_img_rgb)[None, :, :, :]
        exemplar_img_ir = self.transforms(exemplar_img_ir)[None, :, :, :]

        if self.cuda:
            exemplar_img_rgb = exemplar_img_rgb.cuda()
            exemplar_img_ir = exemplar_img_ir.cuda()


        if self.modality == 1:
            self.model.track_init(exemplar_img_rgb)
        else:
            self.model.track_init(exemplar_img_rgb, exemplar_img_ir)



    def update(self, frame_rgb, frame_ir):


        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        frame_rgb = np.asarray(frame_rgb)
        frame_ir = np.asarray(frame_ir)

        instance_img_rgb, _, _, scale_x = self.data_loader.get_instance_image(  frame_rgb,
                                                                            self.bbox,
                                                                            config.template_img_size,
                                                                            config.detection_img_size,
                                                                            config.context_amount,
                                                                            self.img_rgb_mean)

        instance_img_ir, _, _, scale_x = self.data_loader.get_instance_image(frame_ir,
                                                                            self.bbox,
                                                                            config.template_img_size,
                                                                            config.detection_img_size,
                                                                            config.context_amount,
                                                                            self.img_ir_mean)

        instance_img_rgb = self.transforms(instance_img_rgb)[None, :, :, :]
        instance_img_ir = self.transforms(instance_img_ir)[None, :, :, :]

        if self.cuda:
            instance_img_rgb = instance_img_rgb.cuda()
            instance_img_ir = instance_img_ir.cuda()


        if self.modality == 1:
            pred_score, pred_regression = self.model.track(instance_img_rgb)
        else:
            pred_score, pred_regression = self.model.track(instance_img_rgb, instance_img_ir)


        pred_conf   = pred_score.reshape(-1, 2, config.size ).permute(0, 2, 1)
        pred_offset = pred_regression.reshape(-1, 4, config.size ).permute(0, 2, 1)

        delta = pred_offset[0].cpu().detach().numpy()
        box_pred = util.box_transform_inv(self.anchors, delta)
        score_pred = F.softmax(pred_conf, dim=2)[0, :, 1].cpu().detach().numpy()

        s_c = util.change(util.sz(box_pred[:, 2], box_pred[:, 3]) / (util.sz_wh(self.target_sz * scale_x)))  # scale penalty
        r_c = util.change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)
        pscore = penalty * score_pred
        pscore = pscore * (1 - config.window_influence) + self.window * config.window_influence
        best_pscore_id = np.argmax(pscore)
        target = box_pred[best_pscore_id, :] / scale_x

        lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box


        res_x = np.clip(target[0] + self.pos[0], 0, frame_rgb.shape[1])
        res_y = np.clip(target[1] + self.pos[1], 0, frame_rgb.shape[0])


        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y])
        self.target_sz = np.array([res_w, res_h])

        bbox = np.array([res_x, res_y, res_w, res_h])

        self.bbox = (
            np.clip(bbox[0], 0, frame_rgb.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame_rgb.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame_rgb.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame_rgb.shape[0]).astype(np.float64))

        res_x = res_x - res_w/2 # x -> x1
        res_y = res_y - res_h/2 # y -> y1
        bbox = np.array([res_x, res_y, res_w, res_h])

        return bbox


    def track(self, img_rgb_files, img_ir_files, box, visualize=False):
        frame_num = len(img_rgb_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        plotter = None


        for f, (img_rgb_file, img_ir_file) in enumerate(zip(img_rgb_files, img_ir_files)):


            img_rgb = Image.open(img_rgb_file).convert('RGB')
            img_ir = Image.open(img_ir_file).convert('L')

            img_rgb = np.asarray(img_rgb) / 255.
            img_ir = np.asarray(img_ir) / 255.

            start_time = time.time()
            if f == 0:
                self.init(img_rgb, img_ir, box)
            else:
                boxes[f, :] = self.update(img_rgb, img_ir)
            times[f] = time.time() - start_time

            if visualize:
                plotter = show_frame(img_ir,
                                     img_rgb,
                                     self.name,
                                     plotter,
                                     boxes[f, :]
                                     )

        return boxes, times

