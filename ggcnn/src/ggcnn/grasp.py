import numpy as np

import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage.feature import peak_local_max


def _bb_text_to_no(l, offset=(0, 0)):
    # Get bounding box as pixel values.
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class BoundingBoxes:
    def __init__(self, bbs=None):
        if bbs:
            self.bbs = bbs
        else:
            self.bbs = []

    def __getitem__(self, item):
        return self.bbs[item]

    def __iter__(self):
        return self.bbs.__iter__()

    def __getattr__(self, attr):
        if hasattr(BoundingBox, attr) and callable(getattr(BoundingBox, attr)):
            return lambda *args, **kwargs: list(map(lambda bb: getattr(bb, attr)(*args, **kwargs), self.bbs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        bbs = []
        for i in range(arr.shape[0]):
            bbp = arr[i, :, :].squeeze()
            if bbp.max() == 0:
                break
            else:
                bbs.append(BoundingBox(bbp))
        return cls(bbs)

    @classmethod
    def load_from_file(cls, fname):
        bbs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    bb = np.array([
                        _bb_text_to_no(p0),
                        _bb_text_to_no(p1),
                        _bb_text_to_no(p2),
                        _bb_text_to_no(p3)
                    ])

                    bbs.append(BoundingBox(bb))

                except ValueError:
                    # Some files contain weird values.
                    continue
        return cls(bbs)

    def append(self, bb):
        self.bbs.append(bb)

    def copy(self):
        new_bbs = BoundingBoxes()
        for bb in self.bbs:
            new_bbs.append(bb.copy())
        return new_bbs

    def show(self, ax=None, shape=None):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True):
        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        for bb in self.bbs:
            rr, cc = bb.compact_polygon_coords(shape)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = bb.angle
            if width:
                width_out[rr, cc] = bb.length

        return pos_out, ang_out, width_out

    def to_array(self, pad_to=0):
        a = np.stack([bb.points for bb in self.bbs])
        if pad_to:
           if pad_to > len(self.bbs):
               a = np.concatenate((a, np.zeros((pad_to - len(self.bbs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        points = [bb.points for bb in self.bbs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)


class BoundingBox:
    def __init__(self, points):
        self.points = points

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2

    @property
    def as_grasp(self):
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        return self.points.mean(axis=0).astype(np.int)

    @property
    def length(self):
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        return Grasp(self.center, self.angle, self.length/3, self.width).as_bb.polygon_coords(shape)

    def iou(self, bb, angle_threshold=np.pi/6):
        if abs(self.angle - bb.angle) % np.pi > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(bb.points[:, 0], bb.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection/union

    def copy(self):
        return BoundingBox(self.points.copy())

    def offset(self, offset):
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def plot(self, ax, color=None):
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        T = np.array(
            [
                [1/factor, 0],
                [0, 1/factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


class Grasp:
    def __init__(self, center, angle, length=60, width=30, value=1.0):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width
        self.value = value

    def line_points(self, round=True):
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        if round:
            return ((int(x1), int(y1)), (int(x2), int(y2)))
        else:
            return ((x1, y1), (x2, y2))

    @property
    def as_bb(self):
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length / 2 * yo
        x1 = self.center[1] - self.length / 2 * xo
        y2 = self.center[0] - self.length / 2 * yo
        x2 = self.center[1] + self.length / 2 * xo

        return BoundingBox(np.array(
            [
             [y1 - self.width/2 * xo, x1 - self.width/2 * yo],
             [y2 - self.width/2 * xo, x2 - self.width/2 * yo],
             [y2 + self.width/2 * xo, x2 + self.width/2 * yo],
             [y1 + self.width/2 * xo, x1 + self.width/2 * yo],
             ]
        ).astype(np.int))

    def max_iou(self, bbs):
        self_bb = self.as_bb
        max_iou = 0
        for bb in bbs:
            iou = self_bb.iou(bb)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        self.as_bb.plot(ax, color)

    def __repr__(self):
        return '<Grasp: %s, %0.02f, %0.02f>' % (self.center, self.angle, self.value)


def detect_grasps(point_img, ang_img, width_img=None, no_grasps=1, ang_threshold=5, thresh_abs=0.5, min_distance=20):
    local_max = peak_local_max(point_img, min_distance=min_distance, threshold_abs=thresh_abs, num_peaks=no_grasps)

    grasps = []

    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]
        if ang_threshold > 0:
            if grasp_angle > 0:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                                      grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].max()
            else:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                                      grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].min()

        g = Grasp(grasp_point, grasp_angle, value=point_img[grasp_point])
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length/2

        grasps.append(g)

    return grasps


if __name__ == '__main__':
    bbs = BoundingBoxes.load_from_file('/home/douglas/dev/ae_grasp_prediction/data/grasp_data_raw/pcd0100cpos.txt')
