"""
Class for plotting a quadrotor


"""

from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt

class Quadrotor():
    def __init__(self, x=0, y=0, z=0, x1=0,y1=0,z1=0,roll=0, pitch=0, yaw=0, size=0.5, show_animation=True):
        self.p1 = np.array([size / 2, 0, 0, 1]).T
        self.p2 = np.array([-size / 2, 0, 0, 1]).T
        self.p3 = np.array([0, size / 2, 0, 1]).T
        self.p4 = np.array([0, -size / 2, 0, 1]).T
        self.p5 = np.array([size / 2, 0, 0, 1]).T
        self.p6 = np.array([-size / 2, 0, 0, 1]).T
        self.p7 = np.array([0, size / 2, 0, 1]).T
        self.p8 = np.array([0, -size / 2, 0, 1]).T

        self.x_data = []
        self.y_data = []
        self.z_data = []

        self.x1_data = []
        self.y1_data = []
        self.z1_data = []
        self.show_animation = show_animation

        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            self.ax = fig.add_subplot(111, projection='3d')

        self.update_pose(x, y, z, x1,y1,z1,roll, pitch, yaw)

    def update_pose(self, x, y, z, x1, y1, z1, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)

        self.x1 = x1
        self.y1 = y1
        self.z1 = z1

        self.x1_data.append(x)
        self.y1_data.append(y)
        self.z1_data.append(z)

        if self.show_animation:
            self.plot()

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z]
             ])
    def transformation_matrix1(self):
        x1 = self.x1
        y1 = self.y1
        z1 = self.z1
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x1],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y1],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z1]
             ])

    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()
        T1 = self.transformation_matrix1()
        p1_t = np.matmul(T, self.p1)
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        p5_t = np.matmul(T1, self.p5)
        p6_t = np.matmul(T1, self.p6)
        p7_t = np.matmul(T1, self.p7)
        p8_t = np.matmul(T1, self.p8)

        plt.cla()

        self.ax.plot([4,4,4],[0,0,0],[0,0,0],'go--',markersize=12)
        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]], 'k.')

        self.ax.plot([p1_t[0], p2_t[0]], [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        self.ax.plot([p3_t[0], p4_t[0]], [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')
####################################################################
        self.ax.plot([p5_t[0], p6_t[0], p7_t[0], p8_t[0]],
                     [p5_t[1], p6_t[1], p7_t[1], p8_t[1]],
                     [p5_t[2], p6_t[2], p7_t[2], p8_t[2]], 'k.')

        self.ax.plot([p5_t[0], p6_t[0]], [p7_t[1], p7_t[1]],
                     [p5_t[2], p6_t[2]], 'm-')
        self.ax.plot([p7_t[0], p8_t[0]], [p7_t[1], p8_t[1]],
                     [p7_t[2], p8_t[2]], 'm-')

        self.ax.plot(self.x_data, self.y_data, self.z_data,'None')
        self.ax.plot(self.x1_data, self.y1_data, self.z1_data,'None')

        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
        self.ax.set_zlim(0, 40)

        plt.pause(0.01)
