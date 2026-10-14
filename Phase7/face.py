import cv2

face_cascade = cv2.CascadeClassifier('Phase7/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    '''
    detectMultiScale() - scan & detect objects

    1.1 - balance, not too slow, blind

    minNeighbours = 5

    3 - means (loose checking)
    5 - means (safe checking)
    6 or more - means (strict checking)
    '''

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    '''
    x,y = top = left corner of face

    (x+w, y+h) = get the bottom right corner of face

    faces = [
        (100, 150, 80, 80) face1
        (250, 120, 90, 90) face2
    ]

    x - how far from left
    y - how far from top
    w - width of face
    h - height of face
    '''

    cv2.imshow("Webcam Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
