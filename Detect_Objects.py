# Importing Libraries
import argparse
import numpy as np
import cv2 



def Detect_Objects( frame,yolo,MIN_confi ,MIN_Threshold ):

    ln=yolo.getLayerNames()
    ln = [ ln[i[0]-1]  for i in yolo.getUnconnectedOutLayers()]
    
    # Change frame into Binary Large OBject(blob format)
    blob = cv2.dnn.blobFromImage(frame,1/255.0,(320,320),swapRB=True,crop=False)
    yolo.setInput(blob)
    Outputs = yolo.forward(ln) #(exa- 3*X*85) got Outputs as detected objects 
    H,W = frame.shape[:2]
    Boxes,Confidance,ClassId,Cetroid=[],[],[],[]
    
    for Layer in Outputs:
        for detection in Layer:
            cx,cy,w,h=detection[:4] * np.array([W,H,W,H]) 
            x,y = [int(cx-w/2),int(cy-h/2)] #(x,y) is left-butoom corner point
            scores = detection[5:]
            id_ = np.argmax(scores)
            proability =  scores[id_] # probabilty/Confidance of having id_th Object
            center = (cx,cy)

            if(proability > MIN_confi):
                Boxes.append([x,y,int(w),int(h)])
                Confidance.append(float(proability))
                ClassId.append(id_)
                Cetroid.append(center)
    # After Non-max-Supression Index values of Resuting Boxes (2d array Format ,So need to flatten that)
    idxs = cv2.dnn.NMSBoxes( Boxes,Confidance,MIN_confi, MIN_Threshold )   
     
    final_Boxes,final_Confidance,final_ClassId,final_Cetroid=[],[],[],[] 
    if(len(idxs)>0):
        for i in idxs.flatten():
            final_Boxes.append(Boxes[i])
            final_Confidance.append((Confidance[i]))
            final_ClassId.append(ClassId[i])
            final_Cetroid.append(Cetroid[i])
    
    return [final_Boxes, final_Confidance, final_ClassId, final_Cetroid]


parser = argparse.ArgumentParser(description="Pass the Arguments")
parser.add_argument("-i","--input",type=str, default="",help="Path to input(Optional) Video file")
parser.add_argument("-c","--confidance",type=str, default=0.7,help="Min Confidance to Detect object")
parser.add_argument("-t","--Threshold",type=str, default=0.3,help="Threshold For Non-max-supression")

arg= vars(parser.parse_args())


yolo = cv2.dnn.readNetFromDarknet(  './yolo-coco/yolov3.cfg' ,'./yolo-coco/yolov3.weights')
classes =[]
Colors=[]
with open('./yolo-coco/coco.names') as file:
    classes =  file.read().splitlines() 


Colors= [ ( np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255 )) for i in range(0,80) ] 

MIN_confidance=arg["confidance"]
MIN_Threshold = arg["Threshold"]


custom = arg["input"] if(arg["input"]) else 0 
   
video = cv2.VideoCapture(custom)

while(1):

    is_cap,frame = video.read()
    if(cv2.waitKey(1)==ord('q')): # press 'q' key for Quit
        break
    
    final_Boxes, final_Confidance, final_ClassId, final_Cetroid = Detect_Objects(frame,yolo,MIN_confidance,MIN_Threshold)

    # add a rectangle and text on frame to show

    for i in range(0,len(final_Boxes)):
        x,y,w,h = final_Boxes[i]

        text = f"{classes[final_ClassId[i]]} : {round(final_Confidance[i],2)}"
        color = Colors[final_ClassId[i]] 
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2,cv2.LINE_AA)# p1 , p2 are opposite points
        cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2) 

        cv2.imshow("Object Detection",frame)
    

video.release()
cv2.destroyAllWindows()


