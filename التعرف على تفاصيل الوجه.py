import cv2

# تحميل ملفات haarcascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_nose.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

# تشغيل الكاميرا
video_capture = cv2.VideoCapture(0)

while True:
    # قراءة الإطار من الكاميرا
    ret, frame = video_capture.read()

    # تحويل الإطار إلى اللون الرمادي
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # التعرف على الوجه
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # رسم مربع حول الوجوه المكتشفة
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # قص منطقة الوجه لتطبيق التعرف على العين والفم والأنف والأذن
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # التعرف على العين داخل منطقة الوجه
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)

        # رسم مربع حول العين
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        # التعرف على الفم داخل منطقة الوجه
        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=11)

        # رسم مربع حول الفم
        for (mx, my, mw, mh) in mouths:
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)

        # التعرف على الأنف داخل منطقة الوجه
        noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4)

        # رسم مربع حول الأنف
        for (nx, ny, nw, nh) in noses:
            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 255, 255), 2)

       

    # عرض الإطار المعالج
    cv2.imshow('Video', frame)

    # الخروج عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# إيقاف الكاميرا وإغلاق النوافذ
video_capture.release()
cv2.destroyAllWindows()