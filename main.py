#Welkin Write - by Valar Codaeris
import numpy as np
import cv2
from collections import deque
import cv2
import numpy as np 
import copy
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import PIL
import torch.nn as nn
import torch.nn.functional as F



rect1_tl = (620,240)
rect2_tl = (620,340)
rect3_tl = (620,440)
rect4_tl = (540,370)
rect5_tl = (700,370)

height = 30
width = 30


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16,  3,padding = 1 )
        
        self.conv2 = nn.Conv2d(16, 32, 3,padding = 1)
        
        self.conv3 = nn.Conv2d(32, 64, 3,padding = 1)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64*3*3,256)
        self.fc2 = nn.Linear(256,26)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1,64*3*3)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x



def getHistogram(frame):
	roi1 = frame[rect1_tl[1]:rect1_tl[1]+width,rect1_tl[0]:rect1_tl[0]+height]
	roi2 = frame[rect2_tl[1]:rect2_tl[1]+width,rect2_tl[0]:rect2_tl[0]+height]
	roi3 = frame[rect3_tl[1]:rect3_tl[1]+width,rect3_tl[0]:rect3_tl[0]+height]
	roi4 = frame[rect4_tl[1]:rect4_tl[1]+width,rect4_tl[0]:rect4_tl[0]+height]
	roi5 = frame[rect5_tl[1]:rect5_tl[1]+width,rect5_tl[0]:rect5_tl[0]+height]
	roi = np.concatenate((roi1,roi2,roi3,roi4,roi5),axis = 0)
	roi_hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)

	return cv2.calcHist([roi_hsv],[0,1],None,[180,256],[0,180,0,256])


def drawRectangles(frame = 0):
	frame_with_rect = frame
	cv2.rectangle(frame_with_rect,rect1_tl,tuple(np.array(rect1_tl)+np.array((height,width))),(0,0,255),1)
	cv2.rectangle(frame_with_rect,rect2_tl,tuple(np.array(rect2_tl)+np.array((height,width))),(0,0,255),1)
	cv2.rectangle(frame_with_rect,rect3_tl,tuple(np.array(rect3_tl)+np.array((height,width))),(0,0,255),1)
	cv2.rectangle(frame_with_rect,rect4_tl,tuple(np.array(rect4_tl)+np.array((height,width))),(0,0,255),1)
	cv2.rectangle(frame_with_rect,rect5_tl,tuple(np.array(rect5_tl)+np.array((height,width))),(0,0,255),1)
	return frame_with_rect


def getMask(frame, histogram):
	frame_hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
	mask = cv2.calcBackProject([frame_hsv],[0,1],histogram,[0,180,0,256],1)
	_,mask = cv2.threshold(mask,10,255,cv2.THRESH_BINARY)



	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	mask = cv2.filter2D(mask,-1,kernel)

	kernel1 = np.ones((7,7),np.uint8)
	mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel1)
	mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask = cv2.bilateralFilter(mask,5,75,75)

	return mask

def getMaxContour(mask):
	_, contours, _ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	max = 0
	mi = 0
	for i in range(len(contours)):
		area = cv2.contourArea(contours[i])
		if area > max:
			max = area
			mi = i
	return contours[mi]

def drawDefects(frame_with_rect,maxContour,hull):
	defects = cv2.convexityDefects(maxContour,hull)

	for i in range(defects.shape[0]):
	    s,e,f,d = defects[i,0]
	    start = tuple(maxContour[s][0])
	    end = tuple(maxContour[e][0])
	    far = tuple(maxContour[f][0])
	    cv2.line(frame_with_rect,start,far,[255,0,0],2)
	    cv2.line(frame_with_rect,far,end,[0,255,0],2)
	    cv2.circle(frame_with_rect,far,5,[0,0,255],-1)
def getCentroid(contour):
	M = cv2.moments(contour)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	return cx,cy

def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None

def cropCharacter(canvas):
	print(canvas.shape)
	for i in range(canvas.shape[0]):
		for j in range(canvas.shape[1]):
			print('i {i}',i)
			print('j {j}',j)

			if canvas[i,j]!=255:
				canvas = canvas[i:canvas.shape[0],:]

	return canvas

def getROI(canvas):
	gray = cv2.bitwise_not(canvas)
	ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
	_, ctrs, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	areas = []
	for i in range(len(ctrs)):
		x, y, w, h = cv2.boundingRect(ctrs[i])
		areas.append((w*h,i))

	def sortSecond(val): 
		return val[0]  
		
	areas.sort(key = sortSecond,reverse = True)
	x, y, w, h = cv2.boundingRect(ctrs[areas[1][1]])
	cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 0), 1)
	roi = gray[y:y + h, x:x + w]
	#cv2.imshow('ROI', roi)
	return roi

def predictCharacter(roi,model):
	img = cv2.resize(roi, (28, 28)) 
	img = cv2.GaussianBlur(img,(3,3),0)
	img = Image.fromarray(img)
	#img = PIL.ImageOps.invert(img)

	normalize = transforms.Normalize(
	   mean=[0.5,0.5,0.5],
	   std=[0.5,0.5,0.5]
	)
	preprocess = transforms.Compose([
	    transforms.Resize((28,28)),
	    transforms.ToTensor(),
	    normalize
	])

	p_img = preprocess(img)

	model.eval()
	p_img = p_img.reshape([1,1,28,28]).float()
	output = model(torch.transpose(p_img,2,3))
	_, preds_tensor = torch.max(output, 1)
	preds = np.squeeze(preds_tensor.numpy())
	return preds


def histSide():
	cap = cv2.VideoCapture(0)

	canvas = np.zeros((720,1280), np.uint8)

	far_points = []

	pressed = False
	isDrawing = False
	madePrediction = False
	model = Net()

	model.load_state_dict(torch.load('model_emnist.pt',map_location='cpu'))
	while True:
		_ , frame = cap.read()
		frame = cv2.flip(frame,flipCode = 1)
		originalFrame = copy.deepcopy(frame)
		originalFrame = drawRectangles(originalFrame)
		canvas[:,:] = 255

		
def dist(a,b):
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

def fingerCursor(device):
    cap = cv2.VideoCapture(device)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    gesture2 = cv2.imread('gesture2.png')
    gesture2 = cv2.cvtColor(gesture2, cv2.COLOR_BGR2GRAY)
    _, gesture2 , _ = cv2.findContours(gesture2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    skin_min = np.array([0,0,0],np.uint8)
    skin_max = np.array([0,0, 255],np.uint8)

    topmost_last = (200,100)
    traj = np.array([], np.uint16)
    traj = np.append( traj, topmost_last)
    dist_pts = 0
    dist_records = [dist_pts]

    low_filter_size = 5
    low_filter = deque([topmost_last,topmost_last,topmost_last,topmost_last,topmost_last],low_filter_size )

    gesture_filter_size = 5
    gesture_matching_filter = deque([0.0,0.0,0.0,0.0,0.0], gesture_filter_size )
    gesture_index_thres = 0.8

    orange = (0,97,255)
    blue = (255,0,0)
    green = (0,255,0)
    kernel_size = 5
    kernel1 = np.ones((kernel_size,kernel_size),np.float32)/kernel_size/kernel_size
    kernel2 = np.ones((10,10), np.uint8)/100

    while(cap.isOpened()):
        ret, frame_raw = cap.read()
        while not ret:
            ret,frame_raw = cap.read()
        frame_raw = cv2.flip(frame_raw,1)
        frame = frame_raw[:round(cap_height),:round(cap_width)]
        cv2.imshow('raw_frame',frame)


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, skin_min, skin_max)
        res = cv2.bitwise_and(hsv, hsv, mask= mask)
        res = cv2.erode(res, kernel1, iterations=1)
        res = cv2.dilate(res, kernel1, iterations=1)
        rgb = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        cv2.imshow('rgb_2',rgb)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray',gray)

        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        cv2.imshow('gray',gray)
 
        im2, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) !=0:
            c = max(contours, key = cv2.contourArea)
            if cv2.contourArea(c) > 1000:
                topmost = tuple(c[c[:,:,1].argmin()][0]) 
                gesture_index = cv2.matchShapes(c,gesture2[0],2,0.0)
                gesture_matching_filter.append(gesture_index)
                sum_gesture = 0
                for i in gesture_matching_filter:
                    sum_gesture += i
                gesture_index = sum_gesture/gesture_filter_size
                print(gesture_index)

                dist_pts = dist(topmost,topmost_last) 
                if dist_pts < 150:
                    try:
                        cv2.drawContours(rgb, [c], 0 , (0, 255, 0),5)
                        low_filter.append(topmost)
                        sum_x = 0
                        sum_y = 0
                        for i in low_filter:
                            sum_x += i[0]
                            sum_y += i[1]
                        topmost = (sum_x//low_filter_size, sum_y//low_filter_size)

                        if gesture_index > gesture_index_thres:
                            traj = np.append( traj, topmost)
                            dist_records.append(dist_pts)
                  
                        else:
                            traj = np.array([], np.uint16)
                            traj = np.append( traj, topmost_last)
                            dist_pts = 0
                            dist_records = [dist_pts]
                            pass
                        topmost_last = topmost 
                    except:
                        print('error')
                        pass
             
        for i in range(1, len(dist_records)):
	            thickness = int(-0.072 * dist_records[i] + 13)
	            cv2.line(frame, (traj[i*2-2],traj[i*2-1]), (traj[i*2],traj[i*2+1]), orange , thickness)
	            cv2.line(rgb, (traj[i*2-2],traj[i*2-1]), (traj[i*2],traj[i*2+1]), orange , thickness)

        j = cv2.waitKey(1)%256

        if j==ord('w'):
        	orange = (255,255,255)

        if j == ord('b'):
        	orange = (205,0,0)

        if j == ord('r'):
        	orange = (0,0,255)

        cv2.circle(frame, topmost_last, 10, blue , 3)
        cv2.circle(rgb, topmost_last, 10, blue , 3)
        cv2.imshow('rgb', rgb)
        cv2.imshow('frame', frame_raw)
        


        if j == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    device = 0
    fingerCursor(device)
