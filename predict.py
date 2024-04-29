import cv2
import mediapipe as mp
import numpy as np
import csv
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

def predict_kantuk(model, scaler_model,  mar, opennes):
    with open(model, 'wb') as file:
        model = pickle.load(file)
    
    with open(scaler_model, 'wb') as file:
        pickle.dump(scaler_model, file)
    
    # Prepare the data for prediction
    data = np.array([[mar, opennes]])
    data_scaled = scaler_model.transform(data)
    
    # Predict using the SVM model
    prediction = model.predict(data_scaled)
    
    return prediction


def eye_openness(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, binary_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
    white_pixels = cv2.countNonZero(binary_eye)

   #Normalisasi dengan membagi total pixel dibagian mata
    openness = white_pixels / (binary_eye.shape[0] * binary_eye.shape[1])
    return openness, binary_eye

# ekstrak fitur mata 
def get_eye_region(image, landmarks, eye_indices):
    #mengambil titik koordinat landmark
    eye_points = np.array([(int(landmarks.landmark[i].x * image.shape[1]), int(landmarks.landmark[i].y * image.shape[0])) for i in eye_indices])
    eye_points = eye_points.reshape(-1, 1, 2)
    # menghitung bonding box
    x, y, w, h = cv2.boundingRect(eye_points)
    eye_region = image[y:y+h, x:x+w]
    return eye_region

def mouth_aspect_ratio(landmarks):
    # Indices for landmarks of the outer corners of the mouth
    left_mouth_corner = np.array([landmarks[61].x, landmarks[61].y])  # Left corner
    right_mouth_corner = np.array([landmarks[291].x, landmarks[291].y])  # Right corner

    # Indices for landmarks of the upper and lower inner lip
    upper_inner_lip = np.array([landmarks[13].x, landmarks[13].y])  # Upper inner lip
    lower_inner_lip = np.array([landmarks[14].x, landmarks[14].y])  # Lower inner lip

    # Calculate the distances
    horiz_dist = np.linalg.norm(left_mouth_corner - right_mouth_corner)
    vert_dist = np.linalg.norm(upper_inner_lip - lower_inner_lip)

    # Calculate MAR
    mar = vert_dist / horiz_dist * 1.5
    return mar


#masih percobaan
def draw_facial_features(image, landmarks, eye_indices, mouth_indices):
# #     # Get the points for the eyes
      left_eye = np.array([(landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in eye_indices[0]], dtype=np.int32)
      right_eye = np.array([(landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in eye_indices[1]], dtype=np.int32)

#      # Draw the polylines for the eyes
      cv2.polylines(image, [left_eye], True, (0, 255, 0), 2)
      cv2.polylines(image, [right_eye], True, (0, 255, 0), 2)

#      # Get the points for the mouth
      mouth_outer = np.array([(landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in mouth_indices], dtype=np.int32)

# #     # Draw the lines for the mouth
      cv2.polylines(image, [mouth_outer], True, (0, 255, 0), 2)

# #


# indeks mata di mp 468 titik
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
LipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LipsLowerOuter = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]


video_path = r"F:\SKRIPSI\TA\DROZY\videos_i8\6-1.mp4"  # Use the correct path or 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get the base name of the video file without the extension
video_basename = os.path.basename(video_path)
video_title, _ = os.path.splitext(video_basename)

# Define the CSV file name based on the video title
csv_filename = f"{video_title}_data_analysis.csv"

fps = cap.get(cv2.CAP_PROP_FPS)
delay_between_frames = int(1000 / fps)
data_list = []

model = 'model.pkl'
scaler_model = 'scaler_model.pkl'


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video or the video cannot be read.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Frame number
    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            # mengambil bagian mata
            left_eye_region = get_eye_region(frame, face_landmarks, LEFT_EYE_INDICES)
            right_eye_region = get_eye_region(frame, face_landmarks, RIGHT_EYE_INDICES)
            #draw_facial_features(frame, face_landmarks.landmark, (LEFT_EYE_INDICES, RIGHT_EYE_INDICES), MOUTH_INDICES)

            # Calculate eye openness
            left_openness, left_binary_eye = eye_openness(left_eye_region)
            right_openness, right_binary_eye = eye_openness(right_eye_region)
            openness = (left_openness + right_openness)/2
 
            mar = mouth_aspect_ratio(landmarks)

            data_list.append([cap.get(cv2.CAP_PROP_POS_FRAMES), openness, mar])  

            y = predict_kantuk(model, scaler_model, mar, openness) 
            # Display the openness and the binary eye image
            cv2.imshow('Left Eye Binary', left_binary_eye)
            cv2.imshow('Right Eye Binary', right_binary_eye)
            #cv2.putText(image, f'left Eye Openness: {left_openness:.2f}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
           # cv2.putText(image, f'Right Eye Openness: {right_openness:.2f}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Eye Openness: {openness:.2f}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),1)
            cv2.putText(frame, f'MAR: {mar:.2f}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    else:
        # No landmarks detected, so set values to None or np.nan
        left_openness, right_openness, mar = None, None, None 
    # Show the image
    cv2.imshow('MediaPipe Face Mesh', frame)
    if cv2.waitKey(delay_between_frames) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['Frame', 'Openness', 'MAR'])
    # Write the data
    csvwriter.writerows(data_list)
    print(f"Data saved to {csv_filename}")

print(f"Processed {frame_number} frames from {video_title}")