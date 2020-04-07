import cv2
import numpy as np
import time

background = cv2.imread("../kouxiang/kouxiang/timg.jpeg")

capture = cv2.VideoCapture("../kouxiang/kouxiang/lv/1.mp4")

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter('test_ELLIPSE.avi', fourcc, 25.0, (1080, 1920), True)

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



	# 下面可以性能提升
	# w = mask / 255.0
	# part_1 = np.multiply(background * w)
	# part_2 = np.multiply(frame * (1 - w))
	# result[mask_other] = part_1[mask_other] + part_2[mask_other]

	
	# size = frame.shape
	# rows = size[0]
	# cols = size[1]

	# result = np.zeros(size, dtype=frame.dtype)
	# for x in range(rows):
	# 	for y in range(cols):
	# 		if (mask[x][y] == 255):
	# 			result[x][y] = background[x][y]
	# 		elif (mask[x][y] == 0):
	# 			result[x][y] = frame[x][y]
	# 		else:
	# 			m = mask[x][y]
	# 			w = m / 255.0
	# 			result[x][y] = w * background[x][y] + (1 - w) * frame[x][y]

	return result

n = 0

while(capture.isOpened()):
	ret, frame = capture.read()
	if ret == True:
		print(n)
		n = n + 1
		start = time.clock()
		hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		end = time.clock()
		print("hsv_frame", end - start)
		start = time.clock()
		mask = cv2.inRange(hsv_frame, np.array([35, 43, 46]), np.array([77, 255, 255]))
		# mask = cv2.inRange(hsv_frame, np.array([100, 43, 46]), np.array([124, 255, 255]))
		end = time.clock()
		print("mask", end - start)
		start = time.clock()
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
		end = time.clock()
		print("hernel", end - start)
		start = time.clock()
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		end = time.clock()
		print("mask", end - start)
		start = time.clock()
		mask = cv2.GaussianBlur(mask, (3,3), 1)
		mask = cv2.medianBlur(mask, 5)
		end = time.clock()
		print("mask", end - start)
		start = time.clock()

		result = replace_and_blend(frame, mask)
		end = time.clock()
		print("result", end - start)
		start = time.clock()
		out.write(result)
		end = time.clock()
		print("out", end - start)


	else: # 读到末尾
		break

capture.release()
out.release()



