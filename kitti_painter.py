'''Painter: draw boxes in pc/img.'''
import os
import cv2
import subprocess
import numpy as np

from mayavi import mlab

from calibration import Calibration
from data_parser import KittiDataParser
from label_parser import KittiLabelParser

from utils import draw_point_cloud, draw_projected_box3d_on_image


class KittiPainter:
    def __init__(self, root):
        self.root = root
        self.data_parser = KittiDataParser(root)
        self.label_parser = KittiLabelParser(
            root=root, classes=KittiLabelParser.kitti_classes[:-1])

    def get_calib(self, name):
        calib_path = os.path.join(
            self.root, 'training', 'calib', '%s.txt' % fname)
        calib = Calibration(calib_path)
        return calib

    def draw_point_cloud(self, fname):
        pc = self.data_parser.load_point_cloud(fname)
        pc = pc[:, :3]
        draw_point_cloud(pc)

    def draw_point_cloud_in_FOV(self, fname):
        calib = self.get_calib(fname)

        pc = self.data_parser.load_point_cloud(fname)
        pc = pc[:, :3]
        x_ref = calib.project_velo_to_ref(pc)
        x_rect = calib.project_ref_to_cam(x_ref)
        x_img = calib.project_cam_to_img(x_rect)

        img_w, img_h = self.data_parser.get_image_size(fname)
        ids = (x_img[:, 0] < img_w) & (x_img[:, 0] >= 0) & \
            (x_img[:, 1] < img_h) & (x_img[:, 1] >= 0)
        clip_distance = 2.0
        ids = ids & (pc[:, 0] > clip_distance)
        pc = pc[ids, :]
        draw_point_cloud(pc)

    def draw_box3d_on_image(self, fname):
        calib = self.get_calib(fname)

        img = self.data_parser.load_image(fname)
        for x in self.label_parser.parse_boxes_in_camera(fname):
            x = calib.project_cam_to_img(x)  # 8x2
            draw_projected_box3d_on_image(img, x)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    def draw_box3d_in_point_cloud(self, fname):
        calib = self.get_calib(fname)
        pc = self.data_parser.load_point_cloud(fname)

        boxes = []
        for x in self.label_parser.parse_boxes_in_camera(fname):
            x = calib.project_cam_to_ref(x)
            x = calib.project_ref_to_velo(x)
            boxes.append(x)
        draw_point_cloud(pc, boxes)


if __name__ == '__main__':
    painter = KittiPainter(root='./data')
    fname = '000000'
    painter.draw_point_cloud(fname)
    painter.draw_point_cloud_in_FOV(fname)
    painter.draw_box3d_in_point_cloud(fname)
    painter.draw_box3d_on_image(fname)
