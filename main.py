import cv2
import matplotlib.pyplot as plt
#import mediapipe as mp
import numpy as np
import math
import tkinter.messagebox as tkMessageBox
import time
#import pyautogui




# import tempfile  
from tkinter import font, ttk
from PIL import Image,ImageTk
from tkinter import *
#from pycaw.pycaw import AudioUtilities , IAudioEndpointVolume
#from comtypes import  CLSCTX_ALL
from ctypes import cast , POINTER




root = Tk()


##### start AI_control class


class AI_control():


#### start object detection
    def object_detect(self):

        global camera_detect
        camera_detect = cv2.VideoCapture(0)
        camera_detect.set(3,550)
        camera_detect.set(4,250)


        className = []
        classFile = 'coco.names'

        with open(classFile , 'rt') as f:
            className = f.read().rstrip('\n').split('\n') 

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightspaht = 'frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightspaht, configPath)
        net.setInputSize(320, 230)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        while True:
            success, img = camera_detect.read()

            classIds, confs, bbox = net.detect(img, confThreshold = 0.5)
            print(classIds)

            if len(classIds) != 0 :
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox) :
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness= 4)
                    cv2.putText(img, className[classId-1], (box[0]+10, box[1]+20),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0), thickness=2)

            cv2.imshow('Output', img)
            cv2.waitKey(1)   
#### end object detection

#### start Stack plot

    def StackPlot(self):

        global stack

        stack = plt

        mo = np.array(["Jan", "Fan", "Mar", "Apr", "May", "Jun"])

        channel1 = np.array([20, 25, 30, 35, 40, 45])
        channel2 = np.array([15, 20, 25, 30, 35, 40])
        channel3 = np.array([10, 15, 30, 35, 40, 45])
        channel4 = np.array([15, 20, 25, 30, 40, 50])

        stack.stackplot(mo, channel1, channel2, channel3, channel4)
        stack.title("Number of viewers for four TV channels over six months")
        stack.xlabel("Months")
        stack.ylabel("Viewers (in millions)")
        stack.legend(["Channel 1", "Channel 2", "Channel 3", "Channel 4"])

        stack.show()

#### end Stack plot

### start BFS Breadth-Frist Search

    def BFS(graph, start):

        global search

        queue = []
        queue.append(start)

        visited = set()
        while queue:
            node = queue.pop(0)
            print(node, end=" ")

            visited.add(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)


    graph = {
        "A": ["B", "C"],
        "B": ["A", "D", "E"],
        "C": ["A", "F"],
        "D": ["B"],
        "E": ["B", "F"],
        "F": ["C", "E"]
    }

    BFS(graph, "A")

#### end BFS

### start Histogram

    def Histogram(self):

        global hist
        hist = plt
        x = np.random.normal(0, 1, 100)

        hist.hist(x, bins=10)

        hist.show()

### end Histogram

### start

    def faces(self):

        global image

        import cv2


        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')
        mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()


            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

                mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=11)

                for (mx, my, mw, mh) in mouths:
                    cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)

                noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)

                for (nx, ny, nw, nh) in noses:
                    cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 255, 255), 2)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def MultiLP(self):
        global multillpp;
        class MLP:
            def __init__(self, input_size, hidden_size, output_size):
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size


                self.weights_hidden = np.random.randn(self.input_size, self.hidden_size)
                self.weights_output = np.random.randn(self.hidden_size, self.output_size)

                self.bias_hidden = np.random.randn(self.hidden_size)
                self.bias_output = np.random.randn(self.output_size)

            def forward(self, X):

                self.hidden_layer = np.dot(X, self.weights_hidden) + self.bias_hidden
                self.hidden_activation = self.sigmoid(self.hidden_layer)
                self.output_layer = np.dot(self.hidden_activation, self.weights_output) + self.bias_output
                self.output_activation = self.sigmoid(self.output_layer)
                return self.output_activation

            def backward(self, X, y, learning_rate):

                self.error = y - self.output_activation
                self.output_gradient = self.error * self.sigmoid_derivative(self.output_layer)
                self.hidden_gradient = np.dot(self.output_gradient, self.weights_output.T) * self.sigmoid_derivative(
                    self.hidden_layer)

                # تحديث الوزن والانحراف (biases)
                self.weights_output += learning_rate * np.dot(self.hidden_activation.T, self.output_gradient)
                self.bias_output += learning_rate * np.sum(self.output_gradient, axis=0)
                self.weights_hidden += learning_rate * np.dot(X.T, self.hidden_gradient)
                self.bias_hidden += learning_rate * np.sum(self.hidden_gradient, axis=0)

            def train(self, X, y, epochs, learning_rate):
                for epoch in range(epochs):
                    output = self.forward(X)
                    self.backward(X, y, learning_rate)

            def predict(self, X):
                return self.forward(X)

            def sigmoid(self, x):
                return 1 / (1 + np.exp(-x))

            def sigmoid_derivative(self, x):
                return self.sigmoid(x) * (1 - self.sigmoid(x))



        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        mlp = MLP(input_size=2, hidden_size=4, output_size=1)

        mlp.train(X, y, epochs=10000, learning_rate=0.1)

        predictions = mlp.predict(X)
        print("Predictions:", predictions)
'''
#### start mouse controle ####
    def mouse_control(self): # هذه الدالة تقوم بالتحكم بالماوس بواسطة الكاميرا

        global camera_mous

        camera_mous = cv2.VideoCapture(0)
        camera_mous.set(3,350) #هنا تقوم بتحديد حجم الفورم التي تعرض الكاميرا
        camera_mous.set(4,150)#هنا تقوم بتحديد حجم الفورم التي تعرض الكاميرا

        hand_detector = mp.solutions.hands.Hands() # هنا تقوم بتحديد اليد المراد التحكم بها
        drawing_utils = mp.solutions.drawing_utils  # هنا الرسومات تقوم بتحديد اليد والاصابع
        screen_width, screen_height = pyautogui.size()
        index_y = 0
       

        while True:
            _, frame = camera_mous.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width, _ = frame.shape
            output = hand_detector.process(rgb_frame)
            hands = output.multi_hand_landmarks
            # print(hands)
            if hands:
                for hand in hands:
                    drawing_utils.draw_landmarks(frame, hand)
                    landmarks = hand.landmark
                    for id ,landmark in enumerate(landmarks):
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)
                        if id == 8:
                            cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                            index_x = screen_width / frame_width * x
                            index_y = screen_height / frame_height * y
                            pyautogui.moveTo(index_x, index_y)

                        if id == 4:
                            cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 155, 255))
                            thumb_x = screen_width / frame_width * x
                            thumb_y = screen_height / frame_height * y
                            if abs(index_y - thumb_y) < 20: 
                                pyautogui.click()
                                pyautogui.sleep(0)
                                


            cv2.imshow('virtual mouse', frame)
            cv2.waitKey(1)   
#### end mouse controle ####


#### start volume controle ###
    def volume_control(self):   #هنا الدالة تقوم بالتحكم بالصوت بواسطة اليد 
        global camera_volum
        camera_volum = cv2.VideoCapture(0)
        camera_volum.set(3,550)
        camera_volum.set(4,150)
        
        mpHands = mp.solutions.hands
        hands =  mpHands.Hands(min_detection_confidence = 0.7)
        mpDraw = mp.solutions.drawing_utils
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volrange = volume.GetVolumeRange()
        minvol = volrange[0]
        maxvol = volrange[1]
        while True:
            success, img = camera_volum.read()   
            img = cv2.flip(img, 2)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            lmList = []

    

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])

                        # mpDraw.draw_land marks(img, handLms, mpHands.HAND_CONNECTIONS)

                        if len(lmList) == 21:

                            x1, y1 = lmList[4][1],lmList[4][2]
                            x2, y2 = lmList[8][1],lmList[8][2]
                            cx, cy  = (x1 + x2)// 2, (y1 + y2)// 2

                            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
                            cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
                            cv2.line(img, (x1, y1), (x2, y2), (0 ,0 ,0), 3)
                            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                            length = math.hypot(x2 - x1 , y2 - y1)

                            if length > 50 : #هنا يتم تحديد المسافة بين الاصبعيين الخنصر والبنصر 
                                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                            if length > 200 :
                                cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                            vol = np.interp(length, [50, 200], [minvol, maxvol])
                            
                            volume.SetMasterVolumeLevel(vol, None)

            cv2.imshow("Img", img)      
            if cv2.waitKey(39) & 0xFF == 27: 
                break
        
###### end voulume controle


##### to exit from programm
    def Exit(self):  # هذه الدالة تقوم بالخروج من البرنامج
        question = tkMessageBox.askquestion('معلومات البرنامج', 'هل تريد الخروج بالفعل ؟', icon='warning')
        if question == 'yes':
            cv2.destroyAllWindows()
            root.destroy()
            exit()   
##### end exit from programm
'''




  





def Main(): # هنا تم تعريف دالة رئيسية من اجل تنفيذ البرنامج



    global camera_detect
    global camera_mous  
    global camera_volum
    global root
    global stack
    global search
    global hist
    global image
    global multillpp


    root.title(" HomeWork The Doctor: Mohand") #هنا عنوان الوالجهة
    # root.overrideredirect(True)
    
    ai_control = AI_control() # هنا تم تعريف متغير يحمل اسم الكلاس للوصول الى جميع الدوال اللتي بداخلة
    


    width = 1230
    height = 500
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root.geometry("%dx%d+%d+%d" % (width, height, x, y))

    
    # these variables for colors  هنا تم تعريف متغيرات تقوم بحمل القيمة  وهي اسم اللون
    p ="pink"
    r ="red"
    w ="white"
    b ="black"
    y ="yellow"

    # vid =cv2.VideoCapture()
    # video = Label(root)
    # video.pack()
    
    root.config(bg=b) #هنا خلفية الواجهة 





     #Button(text=" get camer   ",command=ai_control.Camer, font=("Cambira math", 20,), fg=p, bg=w).place(x=1070,y=20)

    Button(text='خروج',anchor="n" ,borderwidth=20,font=("Cambira math", 15,'bold'),bg=r,fg=w, width=8, height=1).place(x=10,y=8)

      #  هذا الزر لجلب دالة التحكم بالماوس 
    Button(text='التحكم بالماوس',borderwidth=10, font=("Cambira math", 15,'bold'),bg=p, width=30, height=2).place(x=820,y=170)
      #  هذا الزر لجلب دالة التحكم بالصوت
    Button(text='التحكم بصوت الجهاز',borderwidth=10, font=("Cambira math", 15,'bold'),bg=p, width=30, height=2).place(x=820,y=260)
    # هذا الزر يجلب الدالة للتعرف على الاشياء
    Button(text='ابدأ بالتعرف على الأشياء',borderwidth=10, command=ai_control.object_detect,font=("Cambira math", 15,'bold'),bg=p, width=30, height=2).place(x=820,y=350)

    Button(text='خوارمية Stack Plot',borderwidth=10, command=ai_control.StackPlot,font=("Cambira math", 15,'bold'),bg=p, width=30, height=2).place(x=420,y=350)

    Button(text=' خوارزمية البحث Breadth-Frist Search',borderwidth=10, command=ai_control.BFS,font=("Cambira math", 15,'bold'),bg=p, width=30, height=2).place(x=420,y=260)

    Button(text='خوارزمية Histogram',borderwidth=10, command=ai_control.Histogram,font=("Cambira math", 15,'bold'),bg=p, width=30, height=2).place(x=420,y=170)

    Button(text='التعرف على تفاصيل الوجة',borderwidth=10, command=ai_control.faces,font=("Cambira math", 15,'bold'),bg=p, width=30, height=2).place(x=20,y=170)

    Button(text='خوارزمية MLP ',borderwidth=10, command=ai_control.MultiLP,font=("Cambira math", 15,'bold'),bg=p, width=30, height=2).place(x=20,y=260)

    # l = Label(width=55,height=25)
    # l.pack(fill='y',pady=120,padx=0)
    
    
    
    
    # s = cv2.VideoCapture(0)
    # while True:

    #     d = s.read()
    #     a = cv2.imshow('sss',d)
    #     LabelFrame(width=400,height=50).place(x=20,y=300)
    #     Label(a)
    # Screen(s)

    # Label(s,width=5000,height=600)

    root.mainloop()     


if __name__ == '__main__':
    Main()



