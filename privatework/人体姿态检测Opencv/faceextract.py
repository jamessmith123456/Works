import cv2
image = cv2.imread('./example.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #灰度转换的作用就是：转换成灰度的图片的计算强度得以降低。

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
faces = face_cascade.detectMultiScale(
   gray,
   scaleFactor = 1.2,
   minNeighbors = 3,
   minSize = (64,64),
   flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
if len(faces)>0:
    for(x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
#cv2.imwrite('output.jpg',image)
cv2.imshow("Find Faces!",image)
cv2.waitKey(0)