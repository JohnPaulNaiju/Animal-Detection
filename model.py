import cv2 #OpenCV
#import winsound #Only works on Windows
from ultralytics import YOLO #YOLOv8 model

model = YOLO("weights.pt")

cap = cv2.VideoCapture(0)

dangerous_animals = ['Elephant', 'Lion', 'Tiger', 'Bear', 'Wild Boar', 'Buffalo'] #Wild board and bear doesn't work

def alert(animal, confidence):
    #winsound.Beep(1000, 200) #Play Beep sound (only works on Windows)
    print(f'{animal} detected with confidence {confidence}')
    #Put your code here of what do you want to do when dangerous animal is detected

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)

            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            label = model.names[class_id]

            print(label)

            if label in dangerous_animals and confidence > 0.75:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                alert(label, confidence)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()