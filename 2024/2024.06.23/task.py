import cv2
import numpy as np

cap = cv2.VideoCapture(0)
printed = False

while True:
    success, frame = cap.read()
    if not success:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("Arif", frame)

    if not printed:
        print("This message is printed only once.")
        printed = True

    key = cv2.waitKey(1)
    if key == 27:  # Escape key
        break

cap.release()
cv2.destroyAllWindows()