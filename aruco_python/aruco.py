from math import radians
from numpy import linalg as LA, rad2deg
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

#素材
im1 = np.array(Image.open('src/shisa1.jpg'))
im2 = np.array(Image.open('src/shisa2.jpg'))
# im3 = np.array(Image.open('src/und.bmp'))
# im3 = im3[..., :3]
# print(im1.shape, im2.shape, im3.shape) # (1920, 1080, 3)

marker_length = 0.07 # [m] ### 注意！

mtx = np.load("camera/mtx.npy")
dist = np.load("camera/dist.npy")
# print(mtx); print(dist)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

#画像のサイズ変更
# im1 = cv2.resize(im1, (360, 640))
# im2 = cv2.resize(im2, (360, 640))

#カメラの位置姿勢の計算
def estimatePose(img):
    #検出
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict)
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist) # 複数のマーカでもできるのか？
    # print(rvec) # [[[-2.58769271  0.11320459 -0.80216004]]]
    # print(tvec) # [[[0.01737477 0.05779606 0.49369714]]]

    #描画
    # img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    # img = cv2.aruco.drawAxis(img, mtx, dist, rvec, tvec, marker_length/2)
    # plt.imshow(img); plt.show()

    R = cv2.Rodrigues(rvec)[0]  # 回転ベクトル -> 回転行列
    #print(R)
    """
    [[ 0.97230479 -0.07915592 -0.21990393]
    [ 0.02508188 -0.90012641  0.43490614]
    [-0.23236673 -0.42837693 -0.87321184]]
    """
    R_T = R.T
    print(R_T)
    """
    [[ 0.97230479  0.02508188 -0.23236673]
    [-0.07915592 -0.90012641 -0.42837693]
    [-0.21990393  0.43490614 -0.87321184]]
    """
    T = tvec[0].T # 2次元に変換
    # print(T)

    xyz = np.dot(R_T, - T).squeeze() # 自分から見た相対座標？
    #print(xyz)

    rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0]) # 自分からの角度(ラジアン表記)
    # print(rpy)-[0.07760171 0.29622641 0.41642222]

    return xyz, rpy, R_T

def deriveEuler(R_T, xyz):
    pu = np.array([R_T[0,2], R_T[2,2]])    #xz成分
    pv = -xyz[::2]
    print(xyz)
    print(pu, pv)

    i = np.inner(pu, pv)
    n = LA.norm(pu) * LA.norm(pv)

    c = i / n
    a = np.arccos(np.clip(c, -1.0, 1.0))
    #print(rad2deg(a))

    return a

def eulerAnglesToRotationMatrix(euler): #角度を回転行列に変換
    R_x = np.array([[                1,                 0,                 0],
                    [                0,  np.cos(euler[0]), -np.sin(euler[0])],
                    [                0,  np.sin(euler[0]),  np.cos(euler[0])]])
    R_y = np.array([[ np.cos(euler[1]),                 0,  np.sin(euler[1])],
                    [ 0,                                1,                 0],
                    [-np.sin(euler[1]),                 0,  np.cos(euler[1])]])
    R_z = np.array([[ np.cos(euler[2]), -np.sin(euler[2]),                 0],
                    [ np.sin(euler[2]),  np.cos(euler[2]),                 0],
                    [                0,                 0,                 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def plotAxes(xyz, R_T, elev=90, azim=270):  #表示用
    x, y, z = xyz
    ux, vx, wx = R_T[:,0]
    uy, vy, wy = R_T[:,1]
    uz, vz, wz = R_T[:,2]

    fig = plt.figure(figsize=(4,3))
    ax = Axes3D(fig)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_zlim(-2, 2)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    # draw marker
    ax.scatter(0, 0, 0, color="k")
    ax.quiver(0, 0, 0, 1, 0, 0, length=1, color="r")
    ax.quiver(0, 0, 0, 0, 1, 0, length=1, color="g")
    ax.quiver(0, 0, 0, 0, 0, 1, length=1, color="b")
    ax.plot([-1,1,1,-1,-1], [-1,-1,1,1,-1], [0,0,0,0,0], color="k", linestyle=":")

    ax.quiver(x, y, z, ux, vx, wx, length=0.5, color="r")
    ax.quiver(x, y, z, uy, vy, wy, length=0.5, color="g")
    ax.quiver(x, y, z, uz, vz, wz, length=0.5, color="b")

    fig.canvas.draw()
    plt.show()

if __name__=="__main__":
    xyz, rpy, R_T = estimatePose(im2)   #z軸を原点(AR)の方に向けたい　→　yawはいじる必要ない
    plotAxes(xyz, R_T)
    
    # custom_rpy = np.array([0, 0, radians(180)])
    euler = deriveEuler(R_T, xyz)
    #print(rad2deg(euler))
    rpy[1] = euler
    # print(rpy)
    R = eulerAnglesToRotationMatrix(rpy)
    plotAxes(xyz, R)

    # diff = eulerAnglesToRotationMatrix(rpy) - R_T
    # print(diff.astype(np.float16))