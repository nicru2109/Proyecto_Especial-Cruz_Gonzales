#Prueba Reproducir Video

##
import numpy as np
import cv2
##
#
# video = cv2.VideoCapture(r'video.mp4')
#
# while True:
#     ret, frame = video.read()
#
#     cv2.imshow("video", frame)
#
#     if cv2.waitKey(27) == 27:
#         break
#
# video.release()
# cv2.destroyAllWindows()

##
video = cv2.VideoCapture(r'ambas.mp4')

scale = .25

while video.isOpened():
    ret, frame = video.read()
    # if frame is read correctly ret is True
    if not ret:
        break
    dsize = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    frame = cv2.resize(frame, dsize)
    cv2.imshow('frame', frame)
    if cv2.waitKey(27) == 27:
        break
video.release()
cv2.destroyAllWindows()


##

