import cv2
import numpy as np
import time

# 计时开始
start_time = time.clock()
# 读取视频（路径按需替换）
capture = cv2.VideoCapture("./1.mp4")
# 读取视频的长宽
video_width = capture.get(3)
video_height = capture.get(4)
video_fps = capture.get(5)
# 读取背景图片（此处按需替换图片路径）
background = cv2.imread("./timg.jpeg")
# 注意：背景图片和视频的像素尺寸建议一样，不然后面的运算会报错，如果尺寸不一样，可以用下面的代码对背景进行放缩，但是一般不建议这样，放缩可能会导致图片信息丢失
# background = cv2.resize(background, (video_width, video_height), interpolation=cv2.INTER_CUBIC)
# 设置写入文件格式
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# 设置写入路径（按需更改）
out = cv2.VideoWriter('test_ELLIPSE.mp4', fourcc, video_fps, (int(video_width), int(video_height)), True)


# 替换背景
def replace_and_blend(frame, mask):
	size = frame.shape
	result = np.zeros(size, dtype=frame.dtype)
	mask_0 = np.where(mask == 0)
	mask_255 = np.where(mask == 255)
	mask_other = np.where((mask != 0) & (mask != 255))
	mask_other = zip(mask_other[0], mask_other[1])
	result[mask_0] = frame[mask_0]
	result[mask_255] = background[mask_255]

	for x in mask_other:
		m = mask[x]
		w = m / 255.0
		result[x] = w * background[x] + (1 - w) * frame[x]

	return result

n = 0

# 循环读取视频帧
while(capture.isOpened()):
	ret, frame = capture.read()
	if ret == True:
		print(n)
		n = n+ 1
		# 图像颜色空间转换，转换为HSV颜色，便于替换
		hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# 提取绿幕，如果是要提取蓝幕，用下面那行注释的代码
		mask = cv2.inRange(hsv_frame, np.array([35, 43, 46]), np.array([77, 255, 255]))
		# mask = cv2.inRange(hsv_frame, np.array([100, 43, 46]), np.array([124, 255, 255]))
		# 结构化元素
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
		# 下面两行均为形态学变化函数，其中第一个开运算是先腐蚀后膨胀，用来消除离散小点，平滑边界；第二个是闭运算，先膨胀后腐蚀，排除背景中的小洞
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		# 高斯滤波和中值滤波
		mask = cv2.GaussianBlur(mask, (3,3), 1)
		mask = cv2.medianBlur(mask, 5)
		# 替换背景
		result = replace_and_blend(frame, mask)
		# 将处理完的每一帧写入视频
		out.write(result)

	else: # 读到视频末尾
		break

# 释放读取的视频文件
capture.release()
# 释放写入的视频文件
out.release()
# 计时结束
end_time = time.clock()
# 打印用时
print(end_time - start_time)

#########################################################################
# 关于用时和速度
# 经测试，在8核CPU，10进程的情况下，20秒钟的视频大概处理时长是30秒（具体视视频尺寸大小、时长、计算机CPU运算能力等而定）
# 
# 整个程序中，耗时最多的部分是replace_and_blend函数，这部分主要用了numpy。如果电脑中有英伟达显卡（GT或GTX以上为佳），
# 可以在电脑上安装cudn，然后用pip安装对应cudn对应版本的cupy，用cupy来代替numpy。最佳实践是将所有视频帧的矩阵放到同一个大矩阵中，
# 再进行背景的替换，能较少数据在显存之间的交换耗时。经粗略测试，20秒的视频，用cupy能够在不到20秒的时间内处理完成。