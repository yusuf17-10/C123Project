import numpy as np
import cv2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os,ssl,time
from PIL import Image
import PIL.ImageOps

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

df  = pd.read_csv("labels.csv")

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","X","Y","Z"]
nclasses = len(classes)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size =660 ,random_state = 0)
X_train_scaled = X_train/255
X_test_scaled = X_test/255

clf = LogisticRegression(solver="saga",multi_class="multinomial").fit(X_train_scaled,y_train)


y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test,y_pred)

print("Accuracy :",accuracy)

# Here we are Starting the camera
cap = cv2.VideoCapture(0)

while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #Drawing A Box In The Center Of  The Video
        height,width = gray.shape

        upper_left = (int(width/2 - 56),int(height/2-56))
        bottom_right = (int(width/2 + 56),int(height/2+56))
        cv2.rectangle(gray,upper_left,bottom_right,(0,255,0),2)

        #Considering Only the area inside the box i.e,roi = regionofintrest
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]

        #Converting the cv2 image into pil format
        im_pil = Image.fromarray(roi)
        im_bw = im_pil.convert('L')
        image_bw = im_bw.resize((28,28),Image.ANTIALIAS)
        image_bw_inverted = PIL.ImageOps.invert(image_bw)

        pixel_filter = 20
        min_pixel=np.percentile(image_bw_inverted,pixel_filter)
        image_bw_inverted_scale = np.clip(image_bw_inverted-min_pixel,0,255)
        max_pixel = np.max(image_bw_inverted)

        image_bw_inverted_scale = np.asarray(image_bw_inverted_scale)/max_pixel
        test_sample = np.array(image_bw_inverted_scale).reshape(1,784)
        test_pred = clf.predict(test_sample)
        print("Predicted Class Is :- ",test_pred)
        cv2.imshow("frame",gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()


