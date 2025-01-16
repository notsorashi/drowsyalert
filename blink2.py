import cv2
import mediapipe as mp
import numpy as np
import os
import winsound  


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

drawing_utils = mp.solutions.drawing_utils

# Eye Aspect Ratio (EAR)
def calculate_ear(eye_points, landmarks):
    p2 = np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y])
    p6 = np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y])
    p3 = np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y])
    p5 = np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y])
    p1 = np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y])
    p4 = np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y])

    # Calculate distances
    vertical_distance1 = np.linalg.norm(p2 - p6)
    vertical_distance2 = np.linalg.norm(p3 - p5)
    horizontal_distance = np.linalg.norm(p1 - p4)

    ear = (vertical_distance1 + vertical_distance2) / (2.0 * horizontal_distance)
    return ear

# Eye landmarks based on Mediapipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Parameters
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 2
ALERT_THRESHOLD = 5  

# Variables
blink_count = 0
frame_counter = 0
drowsy_alert_played = False  

# Video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark

        left_ear = calculate_ear(LEFT_EYE, landmarks)
        right_ear = calculate_ear(RIGHT_EYE, landmarks)
        avg_ear = (left_ear + right_ear) / 2.0

        print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Avg EAR: {avg_ear:.2f}")

        # Blink detection 
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= CONSEC_FRAMES:
                blink_count += 1
                print(f"Blink detected! Count: {blink_count}")
            frame_counter = 0

        # Drowsiness detection 
        if frame_counter >= ALERT_THRESHOLD and not drowsy_alert_played:
            winsound.Beep(1000, 3000)  
            drowsy_alert_played = True  
            cv2.putText(frame, "Drowsy! Please Wake Up!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
          
        if frame_counter < ALERT_THRESHOLD:
            drowsy_alert_played = False

        
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    
    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()