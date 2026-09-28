import cv2
import mediapipe as mp
import csv
import time

gesture_name = input("Введи назву жесту: ")
samples_to_collect = 1000 

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

collected = 0
start_time = None
counting_down = False

print(f"Автостарт для: {gesture_name}. Збираємо {samples_to_collect} кадрів.")

with open('gestures_data.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    
    while collected < samples_to_collect:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            if not counting_down and collected == 0:
                start_time = time.time()
                counting_down = True

            if counting_down and collected == 0:
                elapsed = time.time() - start_time
                if elapsed < 5.0:  
                    cv2.putText(frame, f"GET READY: {5.0 - elapsed:.1f}s", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    counting_down = False 

            if not counting_down:

                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                base_z = hand_landmarks.landmark[0].z
                
                row = [gesture_name]
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                
                writer.writerow(row)
                collected += 1
                
                cv2.putText(frame, f"RECORDING: {collected}/{samples_to_collect}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        else:
            cv2.putText(frame, "SHOW YOUR HAND", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Data Collection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print(f"Готово! Додано {collected} нових рядків для '{gesture_name}'.")
