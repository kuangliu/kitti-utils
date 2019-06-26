import os
import numpy as np


class KittiLabelParser:
    kitti_classes = ['Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist',
                     'Tram', 'Misc', 'DontCare']

    def __init__(self, root, classes=None):
        '''
        Args:
          root: KITTI data root.
          classes: If specified, only parse these classes.
        '''
        self.root = os.path.join(root, 'training', 'label_2')
        self.classes = classes if classes else self.kitti_classes

        self.m = {}  # {fname: labels}
        for fname in os.listdir(self.root):
            labels = self.parse_file(fname)
            if len(labels) > 0:
                idx = fname[:-4]
                self.m[idx] = labels
        self.fnames = list(self.m.keys())

    def parse_file(self, fname):
        fname = fname if fname.endswith('.txt') else fname + '.txt'
        ret = []
        with open(os.path.join(self.root, fname)) as f:
            for line in f.readlines():
                sp = line.strip().split()
                cls = sp[0]
                if cls not in self.classes:
                    continue
                a = [float(x) for x in sp[4:]]
                box = [a[0], a[1], a[2], a[3]]
                height, width, length = a[4], a[5], a[6]
                t = [a[7], a[8], a[9]]
                ry = a[10]
                ret.append({
                    'cls': cls, 'box': box,
                    'w': width, 'h': height, 'l': length,
                    't': t, 'ry': ry,
                })
        return ret

    def parse_boxes_in_camera(self, fname):
        '''Parse annotated 3D bounding oxes in camera coordinate.

        Args:
          fname: (str) file name.

        Returns:
          (list(np.array)) list of 3D bounding boxes in camera coordinate, each sized [8,3].
        '''
        boxes = []
        for obj in self.parse_file(fname):
            w, h, l = obj['w'], obj['h'], obj['l']
            t = obj['t']
            ry = obj['ry']
            # obj coord.
            x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
            z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
            x = np.vstack([x_corners, y_corners, z_corners])  # 3x8
            # obj -> camera
            s = np.sin(ry)
            c = np.cos(ry)
            R_obj_to_cam = np.array([[c, 0, s],
                                     [0, 1, 0],
                                     [-s, 0, c]])
            x = np.dot(R_obj_to_cam, x)  # 3x8
            x[0, :] = x[0, :] + t[0]
            x[1, :] = x[1, :] + t[1]
            x[2, :] = x[2, :] + t[2]
            x = x.transpose()  # 8x3
            boxes.append(x)
        return boxes

    def get(self, fname):
        return self.m[fname]

    def __getitem__(self, idx):
        return self.m[self.fnames[idx]]

    def __len__(self):
        return len(self.fnames)


if __name__ == '__main__':
    parser = KittiLabelParser('./data/', classes=['Car'])
    print(len(parser))
    print(parser[0])


