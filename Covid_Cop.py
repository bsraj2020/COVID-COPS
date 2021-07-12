import cv2
import numpy as np
import argparse
from scipy.spatial.distance import cdist as Dis


def Detect_people(frame,yolo,min_confi=0.7 , min_thres=0.3,person_id=0):

    # create Binary large object(BLOB) of image
    blob = cv2.dnn.blobFromImage(frame,1/255.0,(320,320),swapRB=True,crop=False)
    ln = yolo.getLayerNames()
    ln = [ ln[i[0]-1] for i in yolo.getUnconnectedOutLayers() ]
    yolo.setInput(blob)
    outputs = yolo.forward(ln)

    Boxes,Confidance,ClassId,Centroid=[],[],[],[]
    H,W= frame.shape[0:2]

    for output_layer in outputs:
        for detection in output_layer:
            cx,cy,w,h=detection[:4] * np.array([W,H,W,H])
            x,y=int(cx-w/2) ,int(cy-h/2) 
            scores = detection[5:]
            id_ = np.argmax(scores)
            probability = float(scores[id_])
            center = (cx,cy)

            if(probability > min_confi and person_id==id_):
                Boxes.append([x,y,int(w),int(h)]) 
                Confidance.append(probability)
                ClassId.append(id_)
                Centroid.append(center)

    # apply NMS(non-max Suppression) on Bounding Boxes
    Final_Boxes,Final_Confidance,Final_ClassId,Final_Centroid=[],[],[],[]
    indeces = cv2.dnn.NMSBoxes(Boxes,Confidance,MIN_CONFIDANCE,MIN_THRESHOLD) # output is 2d array

    if(len(indeces)>0):
        for i in indeces.flatten():
            Final_Boxes.append(Boxes[i])
            Final_Confidance.append(Confidance[i])
            Final_ClassId.append(ClassId[i])
            Final_Centroid.append(Centroid[i])
    
    return [ Final_Boxes,Final_Confidance,Final_ClassId,Final_Centroid]

   # -----------------begin----------------- 

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input",type=str, default="",help="Path to input(Optional) Video file")
parser.add_argument("-c","--confidance",type=float, default=0.7,help="Min Confidance to Detect object")
parser.add_argument("-t","--Threshold",type=float, default=0.3,help="Threshold For Non-max-supression")
parser.add_argument("-d","--Distance",type=int, default=5, help="Min Social Distance to maintain")

arg= vars(parser.parse_args())

# load YOLO model
yolo = cv2.dnn.readNetFromDarknet( "./yolo-coco/yolov3.cfg" , "./yolo-coco/yolov3.weights" )

classes = []
with open ('yolo-coco/coco.names') as file:
    classes=[ i for i in file.read().splitlines() ]



video = cv2.VideoCapture(arg["input"] if(arg["input"]) else 0)

MIN_CONFIDANCE = arg["confidance"]
MIN_THRESHOLD = arg["Threshold"]
MIN_DISTANCE = 50

target_obj_id = 0

while(1):
    is_cap,frame = video.read()

    if(cv2.waitKey(1)==ord('q')):
        break

    Final_Boxes,Final_Confidance,Final_ClassId,Final_Centroid = Detect_people(frame,yolo,MIN_CONFIDANCE,MIN_THRESHOLD,target_obj_id)
    # now we have Bounding Boxes and their Centroids , now we can measure distance between all bounding Boxes

    DIS = Dis(Final_Centroid,Final_Centroid,metric='euclidean') # output is 2d-metrix if distance between i, j
     
    # find victm person based on distance
    voilationSet = set()

    for i in range(0,(DIS.shape[0])):
        for j in range(i+1,(DIS.shape[1])):
            if(DIS[i,j] < MIN_DISTANCE ):
                voilationSet.add(i)
                voilationSet.add(j)
    print(f"No of Victims people :{len(voilationSet)}")

    for i in range(0,len(Final_Boxes)):
        Color = (0,255,0)
        if(i in voilationSet): # if Victim change Color 
            Color=(0,0,255)

        x,y,w,h = Final_Boxes[i]
        cv2.rectangle(frame,(x,y),(x+w,y+h),Color,2)
        cv2.imshow("Covid-Cop",frame)

video.release()
cv2.destroyAllWindows()

            
            

       
    







