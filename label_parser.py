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
        self.root = root
        self.classes = classes if classes else self.kitti_classes

        self.m = {}  # {fname: labels}
        for fname in os.listdir(self.root):
            labels = self.parse_file(fname)
            if len(labels) > 0:
                idx = fname[:-4]
                self.m[idx] = labels
        self.fnames = list(self.m.keys())

    def parse_file(self, fname):
        ret = []
        with open(os.path.join(self.root, fname)) as f:
            for line in f.readlines():
                sp = line.strip().split()
                cls = sp[0]
                if cls not in self.classes:
                    continue
                a = [float(x) for x in sp[4:]]
                bbox = [a[0], a[1], a[2], a[3]]
                height, width, length = a[4], a[5], a[6]
                t = [a[7], a[8], a[9]]
                ry = a[10]
                ret.append({
                    'cls': cls, 'box': bbox,
                    'w': width, 'h': height, 'l': length,
                    't': t, 'ry': ry,
                })
        return ret

    def get(self, fname):
        return self.m[fname]

    def __getitem__(self, idx):
        return self.m[self.fnames[idx]]

    def __len__(self):
        return len(self.fnames)


def test():
    parser = KittiLabelParser(
        './data/training/label_2/', classes=['Car'])
    print(len(parser))
    print(parser[0])


# test()
