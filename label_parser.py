import os
import numpy as np


class KittiLabelParser:
    def __init__(self, root):
        self.root = root
        self.m = {}
        for fname in os.listdir(self.root):
            idx = fname[:-4]
            self.m[idx] = self.parse_file(fname)
        self.fnames = list(self.m.keys())

    def parse_file(self, fname):
        ret = []
        with open(os.path.join(self.root, fname)) as f:
            for line in f.readlines():
                sp = line.strip().split()
                cls = sp[0]
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
    parser = KittiLabelParser('./data/training/label_2/')
    print(len(parser))
    print(parser[0])

# test()
