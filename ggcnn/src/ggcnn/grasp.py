import numpy as np

import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage.feature import peak_local_max


def _bb_text_to_no(l, offset=(0, 0)):
    """
    Convert a text to a bounding box.

    Args:
        l: (str): write your description
        offset: (int): write your description
    """
    # Get bounding box as pixel values.
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


class BoundingBoxes:
    def __init__(self, bbs=None):
        """
        Initialize bbs

        Args:
            self: (todo): write your description
            bbs: (list): write your description
        """
        if bbs:
            self.bbs = bbs
        else:
            self.bbs = []

    def __getitem__(self, item):
        """
        Return the value of item

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        return self.bbs[item]

    def __iter__(self):
        """
        Returns an iterator over the iterator.

        Args:
            self: (todo): write your description
        """
        return self.bbs.__iter__()

    def __getattr__(self, attr):
        """
        Returns a bounding attribute for the given attribute.

        Args:
            self: (todo): write your description
            attr: (str): write your description
        """
        if hasattr(BoundingBox, attr) and callable(getattr(BoundingBox, attr)):
            return lambda *args, **kwargs: list(map(lambda bb: getattr(bb, attr)(*args, **kwargs), self.bbs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        """
        Loads a bounding object from a bounding array.

        Args:
            cls: (todo): write your description
            arr: (array): write your description
        """
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
        """
        Loads a bounding box from a file.

        Args:
            cls: (todo): write your description
            fname: (str): write your description
        """
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
        """
        Append a new bounding box.

        Args:
            self: (todo): write your description
            bb: (array): write your description
        """
        self.bbs.append(bb)

    def copy(self):
        """
        Returns a copy of this bounding object.

        Args:
            self: (todo): write your description
        """
        new_bbs = BoundingBoxes()
        for bb in self.bbs:
            new_bbs.append(bb.copy())
        return new_bbs

    def show(self, ax=None, shape=None):
        """
        Plot the figure.

        Args:
            self: (todo): write your description
            ax: (todo): write your description
            shape: (int): write your description
        """
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
        """
        Draws a shapely polygon.

        Args:
            self: (todo): write your description
            shape: (int): write your description
            position: (int): write your description
            angle: (float): write your description
            width: (int): write your description
        """
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
        """
        Convert this array to a numpy array.

        Args:
            self: (todo): write your description
            pad_to: (float): write your description
        """
        a = np.stack([bb.points for bb in self.bbs])
        if pad_to:
           if pad_to > len(self.bbs):
               a = np.concatenate((a, np.zeros((pad_to - len(self.bbs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        """
        Center of the center of the bounding points.

        Args:
            self: (todo): write your description
        """
        points = [bb.points for bb in self.bbs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)


class BoundingBox:
    def __init__(self, points):
        """
        Initialize the points

        Args:
            self: (todo): write your description
            points: (todo): write your description
        """
        self.points = points

    def __str__(self):
        """
        Returns the string representation of this node.

        Args:
            self: (todo): write your description
        """
        return str(self.points)

    @property
    def angle(self):
        """
        Returns the angle between two points.

        Args:
            self: (todo): write your description
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2

    @property
    def as_grasp(self):
        """
        Returns : py : class : graspy.

        Args:
            self: (todo): write your description
        """
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        """
        Return the center of all points

        Args:
            self: (todo): write your description
        """
        return self.points.mean(axis=0).astype(np.int)

    @property
    def length(self):
        """
        Calculate the length of the camera.

        Args:
            self: (todo): write your description
        """
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        """
        The width of the rectangle.

        Args:
            self: (todo): write your description
        """
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    def polygon_coords(self, shape=None):
        """
        Return the coordinates of the coordinates of the polygon.

        Args:
            self: (todo): write your description
            shape: (int): write your description
        """
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        """
        Compute the bounding of the polygon.

        Args:
            self: (todo): write your description
            shape: (int): write your description
        """
        return Grasp(self.center, self.angle, self.length/3, self.width).as_bb.polygon_coords(shape)

    def iou(self, bb, angle_threshold=np.pi/6):
        """
        Return the angle between two points

        Args:
            self: (todo): write your description
            bb: (todo): write your description
            angle_threshold: (float): write your description
            np: (todo): write your description
            pi: (todo): write your description
        """
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
        """
        Returns a copy of this bounding box.

        Args:
            self: (todo): write your description
        """
        return BoundingBox(self.points.copy())

    def offset(self, offset):
        """
        Offset the image offset.

        Args:
            self: (todo): write your description
            offset: (int): write your description
        """
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        """
        Rotate the given angle about the given by angle.

        Args:
            self: (todo): write your description
            angle: (float): write your description
            center: (float): write your description
        """
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def plot(self, ax, color=None):
        """
        Plot a matplot.

        Args:
            self: (todo): write your description
            ax: (todo): write your description
            color: (str): write your description
        """
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        """
        Zoom in the image zoom level

        Args:
            self: (todo): write your description
            factor: (float): write your description
            center: (float): write your description
        """
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
        """
        Create a new length.

        Args:
            self: (todo): write your description
            center: (list): write your description
            angle: (float): write your description
            length: (int): write your description
            width: (int): write your description
            value: (todo): write your description
        """
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width
        self.value = value

    def line_points(self, round=True):
        """
        Returns the line

        Args:
            self: (todo): write your description
            round: (todo): write your description
        """
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
        """
        Convert the bounding box as a bounding box

        Args:
            self: (todo): write your description
        """
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
        """
        Compute the maximum of bbs.

        Args:
            self: (todo): write your description
            bbs: (todo): write your description
        """
        self_bb = self.as_bb
        max_iou = 0
        for bb in bbs:
            iou = self_bb.iou(bb)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        """
        Plot the data.

        Args:
            self: (todo): write your description
            ax: (todo): write your description
            color: (str): write your description
        """
        self.as_bb.plot(ax, color)

    def __repr__(self):
        """
        Return a human - readable representation.

        Args:
            self: (todo): write your description
        """
        return '<Grasp: %s, %0.02f, %0.02f>' % (self.center, self.angle, self.value)


def detect_grasps(point_img, ang_img, width_img=None, no_grasps=1, ang_threshold=5, thresh_abs=0.5, min_distance=20):
    """
    Detect_grasps within a point_imgaks.

    Args:
        point_img: (str): write your description
        ang_img: (int): write your description
        width_img: (int): write your description
        no_grasps: (todo): write your description
        ang_threshold: (todo): write your description
        thresh_abs: (todo): write your description
        min_distance: (float): write your description
    """
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
