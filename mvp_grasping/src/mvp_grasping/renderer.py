import numpy as np

import pybullet as pb
import pybullet_data

np.set_printoptions(precision=3, suppress=True)


class Renderer:
    def __init__(self, im_width, im_height, fov, near_plane, far_plane, DEBUG=False):
        self.im_width = im_width
        self.im_height = im_height
        self.fov = fov
        self.near_plane = near_plane
        self.far_plane = far_plane
        aspect = self.im_width/self.im_height
        self.pm = pb.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)
        self.camera_pos = np.array([0, 0, 0.5])
        self.camera_rot = self._rotation_matrix([0, np.pi, 0])

        self.objects = []

        if DEBUG:
            self.cid = pb.connect(pb.GUI)
        else:
            self.cid = pb.connect(pb.DIRECT)

        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        pb.setGravity(0, 0, -10)
        self.draw_camera_pos()

        self._rendered = None
        self._rendered_pos = None
        self._rendered_rot = None

    def load_urdf(self, urdf):
        return pb.loadURDF(urdf)

    def remove_object(self, o_id, update=True):
        pb.removeBody(o_id)
        if update:
            self.objects.remove(o_id)

    def remove_all_objects(self):
        for o_id in self.objects:
            self.remove_object(o_id, False)
        self.objects = []

    def load_mesh(self, mesh, scale=None, position=None, orientation=None, mass=1):
        if scale is None:
            scale = [1, 1, 1]
        if position is None:
            position = [0, 0, 0.1]
        if orientation is None:
            # Random orientation
            r = np.random.rand()
            orientation = [r ** 2, 0, 0, (1 - r) ** 2]
        c_id = pb.createCollisionShape(shapeType=pb.GEOM_MESH, fileName=mesh, meshScale=scale)
        o_id = pb.createMultiBody(baseMass=mass, baseCollisionShapeIndex=c_id, basePosition=position,
                                baseOrientation=orientation)
        self.objects.append(o_id)
        return o_id

    def step(self, n=1):
        for i in range(n):
            pb.stepSimulation()

    @property
    def camera_intrinsic(self):
        # Thanks http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
        return np.array([
            [self.pm[0]*self.im_width/2, 0, self.im_width/2],
            [0, self.pm[5]*self.im_height/2, self.im_height/2],
            [0, 0, 1]
        ])

    def _rotation_matrix(self, rpy):
        r, p, y = rpy

        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(r), -np.sin(r), 0],
            [0, np.sin(r), np.cos(r), 0],
            [0, 0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(p), 0, np.sin(p), 0],
            [0, 1, 0, 0],
            [-np.sin(p), 0, np.cos(p), 0],
            [0, 0, 0, 1]
        ])
        Rz = np.array([
            [np.cos(y), -np.sin(y), 0, 0],
            [np.sin(y), np.cos(y), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        return np.linalg.multi_dot([Rz, Ry, Rx])

    def draw_camera_pos(self):
        pb.removeAllUserDebugItems()
        start = self.camera_pos
        end_x = start + np.dot(self.camera_rot, np.array([0.1, 0, 0, 1.0]))[0:3]
        pb.addUserDebugLine(start, end_x, [1, 0, 0], 5)
        end_y = start + np.dot(self.camera_rot, np.array([0, 0.1, 0, 1.0]))[0:3]
        pb.addUserDebugLine(start, end_y, [0, 1, 0], 5)
        end_z = start + np.dot(self.camera_rot, np.array([0, 0, 0.1, 1.0]))[0:3]
        pb.addUserDebugLine(start, end_z, [0, 0, 1], 5)

    def render(self):
        if np.all(self._rendered_pos == self.camera_pos) and np.all(self._rendered_rot == self.camera_rot):
            return self._rendered

        target = self.camera_pos + np.dot(self.camera_rot, [0, 0, 1.0, 1.0])[0:3]
        up = np.dot(self.camera_rot, [0, -1.0, 0, 1.0])[0:3]
        vm = pb.computeViewMatrix(self.camera_pos, target, up)

        i_arr = pb.getCameraImage(self.im_width, self.im_height, vm, self.pm,
                                  shadow=0,
                                  renderer=pb.ER_TINY_RENDERER)
                                  # renderer=pb.ER_BULLET_HARDWARE_OPENGL)

        # Record the position of the camera, and don't re-render if it hasn't moved.
        self._rendered = i_arr
        self._rendered_pos = self.camera_pos.copy()
        self._rendered_rot = self.camera_rot.copy()

        return i_arr

    def get_depth(self):
        return self.render()[3]

    def get_depth_metres(self, noise=0.001):
        d = self.render()[3]
        # Linearise to metres
        return 2*self.far_plane*self.near_plane/(self.far_plane + self.near_plane - (self.far_plane - self.near_plane)*(2*d - 1)) + np.random.randn(self.im_height, self.im_width) * noise

    def px_to_xyz_metres(self, x_px, y_px, z=None):
        K = self.camera_intrinsic
        if z is None:
            z = self.get_depth_metres()[y_px, x_px]
        else:
            z = np.array([z]*x_px.shape[0])
        x = (x_px - K[0, 2]) / K[0, 0] * z
        y = (y_px - K[1, 2]) / K[1, 1] * z
        return np.dot(self.camera_rot[0:3, 0:3], np.stack((x, y, z))) + self.camera_pos.reshape((3, 1))

    def get_xyz_metres(self):
        K = self.camera_intrinsic
        z = self.get_depth_metres()
        # Pixel to physical coords in camera frame
        x = ((np.vstack((np.arange(0, self.im_width, 1, np.float), )*self.im_height) - K[0, 2])/K[0, 0] * z).flatten()
        y = ((np.vstack((np.arange(0, self.im_height, 1, np.float), )*self.im_width).T - K[1,2])/K[1, 1] * z).flatten()
        # Convert to world frame.
        return np.dot(self.camera_rot[0:3, 0:3], np.stack((x, y, z.flatten()))) + self.camera_pos.reshape((3, 1))

    def get_rgb(self):
        return self.render()[2]

    def z_rotation(self):
        a = np.arctan2(self.camera_rot[1, 0], self.camera_rot[0, 0])
        return a

    def move_to(self, T):
        self.camera_pos = np.array(T)

    def look_at(self, p):
        p = np.array(p)
        if np.all(p == self.camera_pos):
            return
        z = p - self.camera_pos
        up = np.dot(self.camera_rot, [0, 1.0, 0, 1.0])[0:3]
        #up = np.array([0, 1.0, 0])
        x = np.cross(up, z)
        y = np.cross(z, x)

        R = np.vstack((x, y, z, np.array([0, 0, 0]))).T
        R = np.vstack((R, np.array([[0, 0, 0, 1]])))

        R = R / np.linalg.norm(R, axis=0)
        self.camera_rot = R
        self.draw_camera_pos()

    def move_world(self, T):
        self.camera_pos += np.array(T)
        self.draw_camera_pos()

    def move_local(self, T):
        T = np.dot(self.camera_rot[0:3, 0:3], np.array(T))
        self.camera_pos += T
        self.draw_camera_pos()

    def rotate_world(self, rpy):
        self.camera_rot = np.dot(self._rotation_matrix(rpy), self.camera_rot)
        self.draw_camera_pos()

    def rotate_local(self, rpy):
        rpy = np.dot(self.camera_rot[0:3, 0:3], np.array(rpy))
        self.camera_rot = np.dot(self._rotation_matrix(rpy), self.camera_rot)
        self.draw_camera_pos()

    def move_camera_key(self, k):
        pos_step = 0.01
        ang_step = np.pi * 5/180
        if k == ord('q'):
            self.move_local([pos_step, 0, 0])
        elif k == ord('w'):
            self.move_local([-pos_step, 0, 0])
        elif k == ord('a'):
            self.move_local([0, pos_step, 0])
        elif k == ord('s'):
            self.move_local([0, -pos_step, 0])
        elif k == ord('z'):
            self.move_local([0, 0, pos_step])
        elif k == ord('x'):
            self.move_local([0, 0, -pos_step])

        elif k == ord('e'):
            self.rotate_local([ang_step, 0, 0])
        elif k == ord('r'):
            self.rotate_local([-ang_step, 0, 0])
        elif k == ord('d'):
            self.rotate_local([0, ang_step, 0])
        elif k == ord('f'):
            self.rotate_local([0, -ang_step, 0])
        elif k == ord('c'):
            self.rotate_local([0, 0, ang_step])
        elif k == ord('v'):
            self.rotate_local([0, 0, -ang_step])

        elif k == ord('o'):
            self.camera_pos = np.array([0, 0, 0.5])
            self.camera_rot = self._rotation_matrix([0, np.pi, 0])
            self.draw_camera_pos()


if __name__ == '__main__':
    # Testing things
    import cv2
    import matplotlib.pyplot as plt

    im_w = 640
    im_h = 480
    im_fov = 55
    nf = 0.1
    ff = 2.0
    r = Renderer(im_w, im_h, im_fov, nf, ff, DEBUG=True)

    r.load_urdf('plane.urdf')
    import glob
    objs = glob.glob('../obj/*.obj')
    print(objs)
    for obj in objs:
        r.load_mesh(obj)

    r.step(10000)

    r.look_at([0, 0, 0])

    r.z_rotation()

    yaw = -80
    pitch = -40
    dist = 0.4

    cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
    cv2.namedWindow('rgb', cv2.WINDOW_NORMAL)

    for i in range(100):
        rad = 0.2
        height = (0.5, 0.3)
        newpos = [np.cos(i%25 * np.pi * 2 / 25.0) * rad,
                  np.sin(i%25 * np.pi * 2 / 25.0) * rad,
                  i/100.0 * (height[1] - height[0]) + height[0]]
        r.move_to(newpos)
        if i == 0:
            r.look_at([0, 0, 0])
        r.rotate_world([0, 0, 1.0/25.0 * np.pi * 2])

        dep = r.get_depth_metres()
        cv2.imshow('depth', 1 - (dep - dep.min())/(dep.max() - dep.min()))
        rgb = r.get_rgb()
        cv2.circle(rgb, (75, 75), 10, (0, 0, 255))
        cv2.imshow('rgb', rgb)
        #cv2.imshow('depth', dep)
        cv2.waitKey(1000)
        #k = cv2.waitKey(0)
        # if k == ord('p'):
        #     break
        #r.move_camera_key(k)
