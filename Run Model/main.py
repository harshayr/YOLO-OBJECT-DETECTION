import cv2
import yaml
import numpy as np
from yaml.loader import SafeLoader
from yolo_prediction import yolo_pred

yaml_path = '/Users/harshalrajput/Desktop/mlops_yolo/data.yaml'
model_path = '/Users/harshalrajput/Desktop/mlops_yolo/Model/weights/best.onnx'

yolo = yolo_pred(model_path,yaml_path)

cap = cv2.VideoCapture(0)
# yolo = yolo_pred(Model_path,data_yaml)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if ret == False:
        print('unable to read video')
        break
    pred_img = yolo.predictions(frame)

    cv2.imshow("YOLO", pred_img)
    #     cv2.imshow('orignal',img_org)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()








