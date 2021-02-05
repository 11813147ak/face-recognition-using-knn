import cv2
import numpy as np


cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt.xml')

skip=0

filename=input("Enter the name")
face_data=[]
dataset_path='C:\\Users\\HP\\Desktop\\facere\\data\\'
face_section=0

while True:


	ret,frame=cap.read()

	if ret==False:
		continue
	# key_pressed=cv2.waitKey(1) & 0xFF
	# if key_pressed==ord('q'):
	# 	breakq


	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	cv2.imshow("Frame",gray_frame)


	faces=face_cascade.detectMultiScale(frame,1.3,5)
	faces=sorted(faces,key=lambda f:f[2]*f[3])

	#pick the last face because it is largest face according to area
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#extract 

		offset=10

		face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section=cv2.resize(face_section,(100,100))

		skip+=1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))



	cv2.imshow("Frame",frame)
	cv2.imshow("Face section",face_section)

	#store every 10th face




	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break


face_data=np.array(face_data)

face_data=face_data.reshape((face_data.shape[0],-1))


print(face_data.shape)


np.save(dataset_path+filename+'.npy',face_data)

print("Data saved")

cap.release()
# cv2.waitKey(0)
cv2.destroyAllWindows()














































# import cv2

# cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
# face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades +'C:/Users/HP/Desktop/facere/harcascade.xml')
# while True:
# 	ret,frame=cap.read()
# 	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# 	#gray_frame = np.array(gray_frame, dtype='uint8')
# 	if ret==False:
# 		continue

# 	faces=face_cascade.detectMultiScale(gray_frame,1.3,5)


# 	for (x,y,w,h) in faces:
# 		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


# 	cv2.imshow("Video Frame",frame)


# 	key_pressed=cv2.waitKey(1) & 0xFF

# 	if key_pressed==ord('q'):
# 		break

# cap.release()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

















# img=cv2.imread("WIN_20200815_17_37_50_Pro (2).jpg")

# cv2.imshow("yb",img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
