import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 绘制三D图形
import os

def fromVedio(vedio_path, save_pics_dir):

    vc = cv2.VideoCapture(vedio_path)  # 读入视频文件
    fps = vc.get(cv2.CAP_PROP_FPS)  # 帧率
    timeF = int(fps) / 2

    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False

    count = 1
    frame_idx = 0
    img_data = []
    print('start getting pics')
    while rval:  # 循环读取视频帧
        rval, frame = vc.read()

        if (count % timeF == 0):  # 每隔0.5秒截取一帧
            frame_idx += 1
            img_data.append(frame)
            if not os.path.exists(save_pics_dir):       # 将截取的图片存入设置的路径中
                os.makedirs(save_pics_dir)
            cv2.imwrite(save_pics_dir + '/pics_' + str(frame_idx) + '_.jpg', frame)

            # break
        count += 1
    print(len(img_data))        # 总共截取的图片数
    vc.release()

    current_frame = img_data[-1]  # 最后一帧截取的图像为当前图像
    img_data.pop()              # 将最后一帧弹出
    return img_data, current_frame

def fromPics(prev_pics_dir, current_frame_file):
    prev_frame = []         # 新建空列表， 用来存储截取的每帧图片
    current_frame = cv2.imread(current_frame_file)
    for image in os.listdir(prev_pics_dir):
        frame = cv2.imread(prev_pics_dir + '/' + image)
        prev_frame.append(frame)

    return prev_frame, current_frame

def getPixelValue(current_frame, prev_frame, save_npy, kernel, bandwidth=100):
    # 得到图片尺寸
    height = current_frame.shape[0]
    width = current_frame.shape[1]
    channels = current_frame.shape[2]

    # 设置超参数
    # bandwidth = 100
    pi = math.pi
    N = len(prev_frame)  # N 是 最后一帧之前的截取图片的数目
    const = 15 / (8 * pi * N * (bandwidth ** 3))  # EP核函数公式中前面的系数

    if kernel == 'ep':
        count = 0  # 用于展示当前进度
        pixel_value = np.zeros((height, width))  # 用于存储计算得到的数据
        for row in range(height):  # 第 row 行
            for col in range(width):  # 第 col 列

                all_img_pv = 0
                pixel_val_temp = 0
                for i in range(N):  # 第 i 张图片
                    all_img_pv += pixel_val_temp
                    pixel_val_temp = 1
                    for channel in range(channels):  # 第 channel 个通道
                        if ((int(current_frame[row, col, channel]) - int(
                                prev_frame[i][row, col, channel])) / bandwidth) ** 2 > 1:  # 根据EP核函数 作分段判断
                            pixel_val_temp = 0
                            break
                        else:
                            pixel_val_temp *= 1 - (((int(current_frame[row, col, channel]) - int(
                                prev_frame[i][row, col, channel])) / bandwidth) ** 2)  # 根据EP核函数公式进行计算

                pixel_value[row, col] = (all_img_pv * const)
                count += 1
                print('Succeed {} / {} Pixels'.format(count, height * width))  # 展示当前进度
                # break
        np.save('ep_'+save_npy, pixel_value)  # 保存计算得到的数组为npy文件
        return pixel_value

    elif kernel == 'gauss':
        count = 0   # 用于展示当前进度
        pixel_value = np.zeros((height, width))
        for row in range(height):  # 第 row 行
            for col in range(width):  # 第 col 列

                all_img_pv = 0
                for i in range(N):  # 第 i 张图片
                    temp = 0
                    for channel in range(channels):  # 第 channel 个通道
                        v = (int(current_frame[row, col, channel]) - int(prev_frame[i][row, col, channel])) ** 2
                        temp += v

                    all_img_pv += math.exp(- temp / (2 * ((bandwidth) ** 2)))

                pixel_value[row, col] = (all_img_pv * const)
                count += 1
                print('Succeed {} / {} Pixels'.format(count, height * width))   # 展示当前进度
                # break
        np.save('gauss_'+save_npy, pixel_value)     # 存储核函数计算值
        return pixel_value

def Visualization_3d(pixel_value, save_path):
    # 三维图可视化
    height = pixel_value.shape[0]
    width = pixel_value.shape[1]

    fig = plt.figure()  # 创建一张图片
    ax3d = Axes3D(fig)
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # 显示三维图
    ax3d.plot_surface(x, y, pixel_value, rstride=1, cstride=1,
                      cmap=plt.cm.spring)
    plt.savefig(save_path)
    plt.show()

def showBinResult(pixel_value, save_path, threshold=2.5e-7):
    # 显示最终的二值图
    height = pixel_value.shape[0]
    width = pixel_value.shape[1]

    bin_value = np.zeros((height, width))
    for row in range(pixel_value.shape[0]):     # 第 row 行
        for col in range(pixel_value.shape[1]):     # 第 col 列
            if pixel_value[row, col] < threshold:
                bin_value[row, col] = 255
            else:
                bin_value[row, col] = 0

    cv2.imwrite(save_path, bin_value)       # 保存最终的二值图
    cv2.imshow('result', bin_value)         # 显示二值图
    cv2.waitKey(0)

def main(mode, kernel):
    # 设置读取和存储路径
    save_npy = 'pixel_value_road.npy'   # 核函数值存储文件
    # vedio need to set
    vedio_path = './city_cr.mp4'        # 视频路径
    save_pics_dir = 'city_cr'           # 视频截取后图片存储路径
    # pics need to set
    prev_pics_dir = r'.\train_set'      # 当前时刻之前的图片路径
    current_frame_file = r'.\test_set\frame_0777.jpg'       # 当前时刻路径

    # 经验阈值
    # threshold = 2.5e-7

    if mode == 'vedio':                 # 通过mode参数 选择从视频中截取图片 或 直接获得图像序列
        prev_frame, current_frame = fromVedio(vedio_path=vedio_path, save_pics_dir=save_pics_dir)    # 得到从视频中截取的图片
    elif mode == 'pics':
        prev_frame, current_frame = fromPics(prev_pics_dir=prev_pics_dir, current_frame_file=current_frame_file)
                                                                         # 直接得到的图片序列

    pixel_value = getPixelValue(current_frame, prev_frame, save_npy, kernel=kernel)    # 得到核函数计算后的值

    Visualization_3d(pixel_value, 'KDE_3d'+kernel+'.png')         # 得到可视化结果：三维图
    showBinResult(pixel_value, 'KDE_bin_'+kernel+'.png')           # 得到最终的结果：二值图

if __name__ == '__main__':
    main(mode='pics', kernel='gauss')              # kernel 可为 'ep' 或 'gauss', mode 可为 'vedio' 或 'pics'






