#Prueba Reproducir Video

##
import numpy as np
import cv2
##

video = cv2.VideoCapture("ambas emg.mp4")

##
while True:
    ret, frame = video.read()

    cv2.imshow("video", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()

