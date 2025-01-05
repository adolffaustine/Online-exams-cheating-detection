import cv2
import os
import numpy as np
from PIL import Image
from uuid import UUID
from FaceDetection.settings import BASE_DIR

detector = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'Face_Detection', 'haarcascade_frontalface_default.xml'))
recognizer = cv2.face.LBPHFaceRecognizer_create()

class FaceRecognition:

    def faceDetect(self, Entry1):
        face_id = Entry1
        cam = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1
                filename = f"{face_id}.{count}.jpg"
                filepath = os.path.join(BASE_DIR, 'Face_Detection', 'dataset', filename)
                cv2.imwrite(filepath, gray[y:y+h, x:x+w])
                cv2.imshow('Register Face', img)

            k = cv2.waitKey(100) & 0xff
            if k == 27 or count >= 30:
                break

        cam.release()
        cv2.destroyAllWindows()

    def trainFace(self):
        path = os.path.join(BASE_DIR, 'Face_Detection', 'dataset')

        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faceSamples = []
            ids = []
            uuid_to_int = {}
            current_id = 0

            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')

                filename = os.path.split(imagePath)[-1]
                face_id_str = filename.split(".")[0]
                try:
                    face_id = UUID(face_id_str)
                except ValueError:
                    print(f"Skipping file with invalid UUID: {imagePath}")
                    continue

                if face_id not in uuid_to_int:
                    uuid_to_int[face_id] = current_id
                    current_id += 1

                faces = detector.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(uuid_to_int[face_id])

            return faceSamples, ids

        print("\n Training faces. It will take a few seconds. Wait ...")
        faces, ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        trainer_path = os.path.join(BASE_DIR, 'Face_Detection', 'trainer', 'trainer.yml')
        recognizer.save(trainer_path)

        print(f"\n {len(np.unique(ids))} faces trained. Exiting Program")

    def recognizeFace(self):
        recognizer.read(os.path.join(BASE_DIR, 'Face_Detection', 'trainer', 'trainer.yml'))
        faceCascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'Face_Detection', 'haarcascade_frontalface_default.xml'))

        font = cv2.FONT_HERSHEY_SIMPLEX
        cam = cv2.VideoCapture(0)
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 100:
                    name = 'Detected'
                else:
                    name = "Unknown"
                
                cv2.putText(img, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

            cv2.imshow('Detect Face', img)
            k = cv2.waitKey(10) & 0xff
            if k == 27 or confidence > 50:
                break

        print("\n Exiting Program")
        cam.release()
        cv2.destroyAllWindows()
        print(face_id)
        return face_id

def detect_and_recognize_face(frame):
    recognizer.read(os.path.join(BASE_DIR, 'Face_Detection', 'trainer', 'trainer.yml'))
    faceCascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'Face_Detection', 'haarcascade_frontalface_default.xml'))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(0.1 * frame.shape[1]), int(0.1 * frame.shape[0])),
    )

    for (x, y, w, h) in faces:
        face_id, confidence = recognizer.predict(gray[y:y + h, x: x + w])
        if confidence < 100:
            return face_id, confidence
    return None, None
