from django.shortcuts import render, redirect
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .detection import FaceRecognition
from .forms import ResgistrationForm
from .models import UserProfile
from .serializers import UserProfileSerializer,SuccessMessageSerializer
from .main import process_gaze
from threading import Thread
import queue
from .models import GazeData
from .serializers import GazeDataSerializer
output_queue = queue.Queue()
import datetime
from celery import shared_task
faceRecognition = FaceRecognition()

# Start the gaze processing thread
# output_queue = queue.Queue()
# gaze_thread = Thread(target=process_gaze, args=(output_queue,))
# gaze_thread.daemon = True
# gaze_thread.start()

from django.shortcuts import render,redirect
from Face_Detection.detection import FaceRecognition
from .forms import *
from django.contrib import messages
from threading import Thread
import queue
from rest_framework import status, generics
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from .models import UserProfile
from .serializers import UserProfileSerializer
import cv2 as cv
import mediapipe as mp
import time
from .utils import *
import math
import numpy as np
import pygame
from pygame import mixer
from django.http import JsonResponse
# Import the necessary function from detection.py
from Face_Detection.detection import detect_and_recognize_face
faceRecognition = FaceRecognition()


gaze_processing_started = False
# variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
start_voice = False
counter_right = 0
counter_left = 0
counter_center = 0

# constants
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# initialize mixer
mixer.init()
# loading in the voices/sounds
voice_left = mixer.Sound('Face_Detection/Voice/left.wav')
voice_right = mixer.Sound('Face_Detection/Voice/Right.wav')
voice_center = mixer.Sound('Face_Detection/Voice/center.wav')

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord

# Euclidean distance
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # LEFT_EYE
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

# Eyes Extractor function
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155

    r_max_x = max(right_eye_coords, key=lambda item: item[0])[0]
    r_min_x = min(right_eye_coords, key=lambda item: item[0])[0]
    r_max_y = max(right_eye_coords, key=lambda item: item[1])[1]
    r_min_y = min(right_eye_coords, key=lambda item: item[1])[1]
    l_max_x = max(left_eye_coords, key=lambda item: item[0])[0]
    l_min_x = min(left_eye_coords, key=lambda item: item[0])[0]
    l_max_y = max(left_eye_coords, key=lambda item: item[1])[1]
    l_min_y = min(left_eye_coords, key=lambda item: item[1])[1]

    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    return cropped_right, cropped_left

# Eyes Position Estimator
def positionEstimator(cropped_eye):
    if cropped_eye is None or cropped_eye.size == 0:
        print("Error: cropped_eye is empty or None")
        return None, (255, 255, 255)
 
    h, w = cropped_eye.shape[:2]
    if h == 0 or w == 0:
        print("Error: cropped_eye has invalid dimensions")
        return None, (255, 255, 255)

    try:
        gaussain_blur = cv.GaussianBlur(cropped_eye, (5, 5), 0)
        median_blur = cv.medianBlur(gaussain_blur, 3)
    except cv.error as e:
        print(f"Error during blurring: {e}")
        return None, (255, 255, 255)

    threshed_eye = cv.adaptiveThreshold(median_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    piece = int(w / 3)
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)
    return eye_position, color

# creating pixel counter function
def pixelCounter(first_piece, second_piece, third_piece):
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    eye_parts = [right_part, center_part, left_part]

    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''
    if max_index == 0:
        pos_eye = "RIGHT"
        color = [BLACK, GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [YELLOW, PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [GRAY, YELLOW]
    else:
        pos_eye = "Closed"
        color = [GRAY, YELLOW]
    return pos_eye, color

def process_gaze(output_queue):
    camera = cv.VideoCapture(0)
    _, frame = camera.read()
    img = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
    img_height, img_width = img.shape[:2]

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('Face_Detection/output21.mp4', fourcc, 30.0, (img_width, img_height))

    map_face_mesh = mp.solutions.face_mesh

    frame_counter = 0
    CEF_COUNTER = 0
    TOTAL_BLINKS = 0
    counter_right = 0
    counter_left = 0
    counter_center = 0

    initial_face_id = None

    eye_position_right = "UNKNOWN"
    eye_position_left = "UNKNOWN"

    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        start_time = time.time()
        while True:
            frame_counter += 1
            ret, frame = camera.read()
            if not ret:
                break
            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)

            # Face detection and recognition check
            current_face_id, confidence = detect_and_recognize_face(frame)
            if current_face_id is None:
                face_status = status.HTTP_100_CONTINUE
                message = 'Exam will be submitted automatically due to wrong gaze position'
            elif initial_face_id is None:
                initial_face_id = current_face_id
                face_status = status.HTTP_100_CONTINUE
                message = ''
            elif current_face_id != initial_face_id:
                face_status = status.HTTP_400_BAD_REQUEST
                message = ''
                break
            else:
                face_status = status.HTTP_200_OK
                message = ''

            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                colorBackgroundText(frame, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, PINK, YELLOW)

                if ratio > 5.5:
                    CEF_COUNTER += 1
                    colorBackgroundText(frame, f'Blink', FONTS, 1.7, (int(frame_height / 2), 100), 2, YELLOW, pad_x=6, pad_y=6)
                else:
                    if CEF_COUNTER > CLOSED_EYES_FRAME:
                        TOTAL_BLINKS += 1
                        CEF_COUNTER = 0
                colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)

                cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, GREEN, 1, cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, GREEN, 1, cv.LINE_AA)

                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
                eye_position_right, color = positionEstimator(crop_right)
                colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
                eye_position_left, color = positionEstimator(crop_left)
                colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)

                if eye_position_right == "RIGHT" and pygame.mixer.get_busy() == 0 and counter_right < 2:
                    counter_right += 1
                    counter_center = 0
                    counter_left = 0
                    voice_right.play()

                if eye_position_right == "CENTER" and pygame.mixer.get_busy() == 0 and counter_center < 2:
                    counter_center += 1
                    counter_right = 0
                    counter_left = 0
                    voice_center.play()

                if eye_position_right == "LEFT" and pygame.mixer.get_busy() == 0 and counter_left < 2:
                    counter_left += 1
                    counter_center = 0
                    counter_right = 0
                    voice_left.play()

            end_time = time.time() - start_time
            fps = frame_counter / end_time

            frame = textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
            last_action=None
            if eye_position_right != last_action: 
               result = {
                'timestamp': datetime.datetime.now().isoformat(),
                # 'total_blinks': TOTAL_BLINKS,
                'eye_position_right': eye_position_right,
                'face_status': face_status,
                'message': message
                # 'eye_position_left': eye_position_left,
                # 'fps': round(fps, 1)
            }
            output_queue.put(result)
            last_action = eye_position_right
            # Save to database
            GazeData.objects.create(
                  timestamp=result['timestamp'],
                # total_blinks=TOTAL_BLINKS,
                  eye_position_right=eye_position_right,
                  face_status= face_status, 
                  message=message
                # eye_position_left=eye_position_left,
                # fps=round(fps, 1)
            )
            out.write(frame)
           
            cv.imshow('frame', frame)
            key = cv.waitKey(2)
            if key == ord('q') or key == ord('Q'):
                break
        cv.destroyAllWindows()
        camera.release()
   
def home(request):
    return render(request,'faceDetection/home.html')



def addFace(face_id):
    face_id = face_id
    faceRecognition.faceDetect(face_id)
    faceRecognition.trainFace()
    return redirect('/')

class GazeDataAPIView(APIView):
    def get(self, request, format=None):
        global gaze_processing_started

        if not gaze_processing_started:
            output_queue = queue.Queue()
            gaze_thread = Thread(target=process_gaze, args=(output_queue,))
            gaze_thread.daemon = True
            gaze_thread.start()
            gaze_processing_started = True
        
        # Fetch the latest GazeData record from the database
        latest_gaze_data = GazeData.objects.order_by('-timestamp').first()        
        if latest_gaze_data:
            serializer = GazeDataSerializer(latest_gaze_data)
            return Response(serializer.data, status=status.HTTP_200_OK)
        else:
            return JsonResponse(
                {
                    "message": "No gaze detected"
                }, status=status.HTTP_400_BAD_REQUEST
            )

@shared_task
def handle_face_recognition(face_id):
    user = UserProfile.objects.get(face_id=face_id)
    faceRecognition.faceDetect(user.face_id)
    faceRecognition.trainFace()         


class RegisterUserAPIView(APIView):
    def post(self, request, format=None):
        form = ResgistrationForm(request.data)
        if form.is_valid():
            user = form.save()
            # Trigger the asynchronous task
            handle_face_recognition.delay(user.face_id)
            serializer = UserProfileSerializer(user)
            messages.success(request, 'User account successfully created')
            request.session['registration_success'] = True
            return JsonResponse(
                {
                    "message": "Success"
                }, status=status.HTTP_200_OK
            )
        else:
            return JsonResponse(
                {
                    "error": "Invalid"
                }, status=status.HTTP_400_BAD_REQUEST
            )
# class RegisterUserAPIView(APIView):
#     def post(self, request, format=None):
#         form = ResgistrationForm(request.data)
#         if form.is_valid():
#             user = form.save()
#             faceRecognition.faceDetect(user.face_id)
#             faceRecognition.trainFace()
#             serializer = UserProfileSerializer(user)
#             messages.success(request, 'User account successfully created')
#             request.session['registration_success'] = True
#             return JsonResponse(
#                 {
#                     "message":"Success"
#                 }, status=200
#             )
#         else:
#             return JsonResponse(
#                 {
#                     "error":"Invalid"
#                 }, status=400
#             )
        
    # def get(self, request, format=None):
    #     if request.session.get('registration_success', False):
    #         message = {"message": "User account successfully created"}
    #         request.session['registration_success'] = False
    #         return Response(message, status=status.HTTP_400_BAD_REQUEST)
    #     # elif request.session.get('login_success', False):
    #     #     message = {"message": "User logged in successfully"}
    #     #     request.session['login_success'] = False
    #     #     return Response(message, status=status.HTTP_200_OK)
    #     else:
    #         return Response({"message": "Please register or log in first!"}, status=status.HTTP_400_BAD_REQUEST)
    
    # def post(self, request, format=None):
    #     form = ResgistrationForm(request.data)
    #     if form.is_valid():
    #         user = form.save()
    #         faceRecognition.faceDetect(user.face_id)
    #         faceRecognition.trainFace()
    #         serializer = UserProfileSerializer(user)
    #         messages.success(request, 'user account succesful created')
    #         return Response(serializer.data, status=status.HTTP_201_CREATED)
            
    #     else: 
    #         context = {'form':form}
    #         return Response(form.errors, status=status.HTTP_400_BAD_REQUEST)
      

# Login User API
# class LoginUserAPIView(APIView):
#     def post(self, request, format=None):
#         face_id = faceRecognition.recognizeFace()
#         try:
#             user = UserProfile.objects.get(face_id=face_id)
#             serializer = UserProfileSerializer(user)
#             messages.success(request, 'User logged in successfully')
#             request.session['login_success'] = True
#             return Response(serializer.data, status=status.HTTP_200_OK)
#         except UserProfile.DoesNotExist:
#             return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
#     def get(self, request, format=None):
#         if request.session.get('login_success', False):
#             message = {"message": "User logged in successfully"}
#             request.session['login_success'] = False
#             return Response(message, status=status.HTTP_200_OK)
#         else:
#             return Response({"message": "Please register or log in first!"}, status=status.HTTP_400_BAD_REQUEST)
    

    # def post(self, request, format=None):
    #     face_id = faceRecognition.recognizeFace()
    #     try:
    #         user = UserProfile.objects.get(face_id=face_id)
    #         serializer = UserProfileSerializer(user)
    #         return Response(serializer.data)
    #     except UserProfile.DoesNotExist:
    #         return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)
        

        # form = LoginForm(request.POST)
        
        # if form.is_valid():
        #     username = form.cleaned_data['username']
        #     password = form.cleaned_data['password']
        #     user = authenticate(request,username=username,password=password)
        #     if user:
        #         login(request, user)
        #         # messages.success(request,f'Hi {username.title()}, welcome back!')
        #         return redirect('index')
        
        # # form is not valid or user is not authenticated
        # messages.error(request,f'Invalid username or password')
        # return render(request,'login.html',{'form': form})
# Success Message API
# class SuccessMessageAPIView(APIView):
#     def get(self, request, format=None):
#         success_message = messages.get_messages(request)
#         message_text = None
#         for message in success_message:
#             message_text = message
#         if message_text:
#             serializer = SuccessMessageSerializer({'success': True, 'message': str(message_text)})  # Serialize the data
#             return Response(serializer.data)
#         else:
#             serializer = SuccessMessageSerializer({'success': False, 'message': 'No success message available'})  # Serialize the data
#             return Response(serializer.data)