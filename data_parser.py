'''Parse KITTI image & point cloud.'''
import os
import cv2
import subprocess
import numpy as np


class KittiDataParser:
    def __init__(self, root):
        self.root = root

    def get_image_size(self, fname):
        '''Get image size without loading it.

        Args:
          fname: (str) file name.

        Returns:
          (tuple) of w, h.
        '''
        img_path = os.path.join(
            self.root, 'training', 'image_2', '%s.png' % fname)
        s = subprocess.getoutput('file %s' % img_path)
        w, h = [int(x) for x in s.split(',')[1].split('x')]
        return w, h

    def load_image(self, fname):
        img_path = os.path.join(
            self.root, 'training', 'image_2', '%s.png' % fname)
        img = cv2.imread(img_path)
        return img

    def load_point_cloud(self, fname):
        pc_path = os.path.join(
            self.root, 'training', 'velodyne', '%s.bin' % fname)
        pc = np.fromfile(pc_path, dtype=np.float32)
        pc = pc.reshape(-1, 4)  # Nx4
        return pc
