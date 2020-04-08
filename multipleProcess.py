import multiprocessing as mul
import cv2
import numpy as np
import time

background = cv2.imread("./timg.jpeg")

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


def f(frame):
	hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_frame, np.array([35, 43, 46]), np.array([77, 255, 255]))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	mask = cv2.GaussianBlur(mask, (3,3), 1)
	mask = cv2.medianBlur(mask, 5)
	result = replace_and_blend(frame, mask)
	return result


if __name__ == '__main__':
	start_time = time.clock()
	capture = cv2.VideoCapture("./1.mp4")
	frames = []
	while(capture.isOpened()):
		ret, frame = capture.read()
		if ret == True:
			frames.append(frame)
		else:
			capture.release()
			break
	end_time = time.clock()
	print(end_time -start_time)
	pool = mul.Pool(10)
	rel = pool.map(f, frames)
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter('test_ELLIPSE.avi', fourcc, 25.0, (1080, 1920), True)
	for x in rel:
		out.write(x)
	out.release()
	end_time = time.clock()
	print(end_time - start_time)