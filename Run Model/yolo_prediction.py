
import cv2
import yaml
import numpy as np
from yaml.loader import SafeLoader


class yolo_pred():
    def __init__(self, onnox_model, data_yaml):
        # load yaml

        with open(data_yaml,mode = 'r')as f:
            data_yaml = yaml.load(f,Loader = SafeLoader)

        self.lables = data_yaml['names']
        self.nc = data_yaml['nc']


        # load model

        self.yolo = cv2.dnn.readNetFromONNX(onnox_model)

        # model train on gpu and we are testing on cpu
        self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    
    def predictions(self, image):

        rows, column , d = image.shape #  i.e (height, width, colur density(r,g,b))

        # step -1 convert image into square matrix

        max_rc = max(rows,column)
        input_image  = np.zeros((max_rc,max_rc,3),dtype =np.uint8) #A uint8 data type contains all whole numbers from 0 to 255.
        input_image[0:rows,0:column] = image # on this black image we have to overlay our image

        # step - 2 get prediction from square array

        INPUT_WH_YOLO = 640 # we train aur model on this image size
        blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WH_YOLO,INPUT_WH_YOLO),swapRB = True,crop = False)
        self.yolo.setInput(blob)
        pred = self.yolo.forward()  # prediction from yolo

        # NON MAX SUPRESSION

        # step1 filter detection based on confidence score (0.4) and probablity(0.25) 
        #and for that we have to convert [xc,yc,w,h] into [xmin,xmax,ymin,ymax]

        detections = pred[0]

        boxes = []
        confidences = []
        classes = []

        # width and height of the input_image
        image_w, image_h = input_image.shape[:2]
        x_factor = image_w/INPUT_WH_YOLO
        y_factor = image_h/INPUT_WH_YOLO

        for i in range(len(detections)):  # now we are itering for each row in detections
            row = detections[i]
            confidence = row[4]   # confidence for detecting object

            if confidence>0.4:
                clas = row[5:].max()  # maximum probablity of obj
                clas_id = row[5:].argmax()   # position of maximum probablity of obj
                if clas>0.25:
                    cx,cy,w,h = row[0:4]
                    # constract bounding box
                    # left top width height

                    left = int((cx - 0.5*w)*x_factor)
                    top = int((cy - 0.5*h)*y_factor)
                    width = int(w*x_factor) 
                    height  = int(h*y_factor)

                    box = np.array([left,top,width,height])

                    # append all values to list

                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(clas_id)

        # still there are some dublicate values are there whose confidence is same

        boxes_np = np.array(boxes).tolist()
        confidences_np = np.array(confidences).tolist()

        # NON MAX SUPRESSION

        index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25, 0.45)
        

        # draw bounding boxes

        for i in index:
            x,y,w,h = boxes_np[i]
            bb_conf = int(confidences_np[i]*100)
            class_id = classes[i]
            class_name = self.lables[class_id]
            colour = self.generate_colour(class_id)

            text = f'{class_name}: {bb_conf}%'

            #draw given  bounding boxes to obj
            cv2.rectangle(image,(x,y),(x+w,y+h), colour ,5)
            cv2.rectangle(image,(x,y-50),(x+w,y),colour,-1)

            cv2.putText(image,text,(x,y-20),cv2.FONT_HERSHEY_PLAIN, 2,(0,0,0),2)
            
            
        return image



    def generate_colour(self,ID):
        
        np.random.seed(10)   # to get the same colour every time you execute for same id
        colours = np.random.randint(100,255,size = (self.nc,3)).tolist()
        
        return tuple(colours[ID])
        
        
        




