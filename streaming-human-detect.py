#import libraries
import cv2

counterPeople = 0
#resize image function
def resize_img(img, width):
  height = int(img.shape[0]/img.shape[1]*width)
  img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
  return img

#Human detected using Haar-like cascade feature - xml
human_detect = cv2.CascadeClassifier('xml/haarcascade_fullbody.xml')

#Open streaming video
cam = cv2.VideoCapture(0)

while True:
  #read() function, return 1 - when success and 2 - self frame
  (success, frame) = cam.read()
  if not success:
    break
  #reduce size of frame
  frame = resize_img(frame, 320)
  #bluring
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  full_bodys = human_detect.detectMultiScale(gray_frame, scaleFactor = 1.1,
minNeighbors = 3, minSize = (20, 20), flags = cv2.CASCADE_SCALE_IMAGE)
  temp_frame = frame.copy() #temporary frame
  for (x, y, w, h) in full_bodys:
    cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    counterPeople += 1

  print("Counting..."+str(counterPeople)) 

  #Show particular frame
  cv2.imshow('Searching for humans...', resize_img(temp_frame, 640))
  #Waiting for any key to exit
  if cv2.waitKey(1) & 0xFF == ord("s"):
    break

cam.release()
cv2.destroyAllWindows()
