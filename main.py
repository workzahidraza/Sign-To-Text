from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp

app = Flask(__name__)

translated_text = ""  # Global variable to hold translated sign text

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


# ======= Default webcam setup =======
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Error: Cannot open DroidCam stream. Check IP and port.")

# ======= Hand gesture detection functions =======
def fingers_open(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    return (landmarks[8][1] < landmarks[6][1] and 
            landmarks[12][1] < landmarks[10][1] and  
            landmarks[16][1] < landmarks[14][1] and  
            landmarks[20][1] < landmarks[18][1])

def is_thumbs_up(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    thumb_up = landmarks[4][1] < landmarks[5][1]
    index_folded = landmarks[8][1] > landmarks[5][1]
    middle_folded = landmarks[12][1] > landmarks[9][1]
    ring_folded = landmarks[16][1] > landmarks[13][1]
    pinky_folded = landmarks[20][1] > landmarks[17][1]
    return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded

def is_victory(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    index_up = landmarks[8][1] < landmarks[6][1]     
    middle_up = landmarks[12][1] < landmarks[10][1]  
    ring_folded = landmarks[16][1] > landmarks[14][1]  
    pinky_folded = landmarks[20][1] > landmarks[18][1]  
    return index_up and middle_up and ring_folded and pinky_folded

def is_call_me(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    thumb_up = landmarks[4][1] < landmarks[3][1]
    pinky_up = landmarks[20][1] < landmarks[19][1]
    index_folded = landmarks[8][1] > landmarks[7][1]
    middle_folded = landmarks[12][1] > landmarks[11][1]
    ring_folded = landmarks[16][1] > landmarks[15][1]
    return thumb_up and pinky_up and index_folded and middle_folded and ring_folded

def is_i_love_you(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    thumb_up = landmarks[4][1] < landmarks[3][1]
    index_up = landmarks[8][1] < landmarks[7][1]
    pinky_up = landmarks[20][1] < landmarks[19][1]
    middle_folded = landmarks[12][1] > landmarks[11][1]
    ring_folded = landmarks[16][1] > landmarks[15][1]
    return thumb_up and index_up and pinky_up and middle_folded and ring_folded

def is_good_luck(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    index_crossed = landmarks[8][0] < landmarks[12][0]  # just an example
    return index_crossed  # adjust based on your test

def is_rock_on(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    index_up = landmarks[8][1] < landmarks[7][1]
    pinky_up = landmarks[20][1] < landmarks[19][1]
    middle_folded = landmarks[12][1] > landmarks[11][1]
    ring_folded = landmarks[16][1] > landmarks[15][1]
    return index_up and pinky_up and middle_folded and ring_folded

def is_i_want_to_talk(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    thumb_folded = landmarks[4][1] > landmarks[3][1]
    index_up = landmarks[8][1] < landmarks[7][1]
    return thumb_folded and index_up

def is_care(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    palms_open = all([landmarks[i][1] < landmarks[i-1][1] for i in [4, 8, 12, 16, 20]])
    return palms_open

def is_pain(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    fist = all([landmarks[i][1] > landmarks[i-1][1] for i in [4, 8, 12, 16, 20]])
    return fist

def is_loser(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    index_up = landmarks[8][1] < landmarks[7][1]
    thumb_out = landmarks[4][0] < landmarks[3][0]  # left hand L
    return index_up and thumb_out

def is_hurts_a_lot(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    fist = all([landmarks[i][1] > landmarks[i-1][1] for i in [4, 8, 12, 16, 20]])
    thumb_cross = landmarks[4][0] > landmarks[3][0]
    return fist and thumb_cross

def is_ok(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = ((thumb_tip[0]-index_tip[0])**2 + (thumb_tip[1]-index_tip[1])**2)**0.5
    fingers_folded = all([landmarks[i][1] > landmarks[i-1][1] for i in [12,16,20]])
    return distance < 0.05 and fingers_folded

def is_good_job(hand_landmarks):
    landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    thumb_up = landmarks[4][1] < landmarks[3][1]
    fingers_folded = all([landmarks[i][1] > landmarks[i-1][1] for i in [8,12,16,20]])
    return thumb_up and fingers_folded


# ======= Video frame generator =======
def gen_frames():
    global translated_text  
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame from DroidCam.")
                continue  # retry if frame not captured

            frame = cv2.resize(frame, (640, 480))  # optional resize

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                if fingers_open(results.right_hand_landmarks):
                    translated_text = "Hello"
                elif is_thumbs_up(results.right_hand_landmarks):
                    translated_text = "OK"
                elif is_victory(results.right_hand_landmarks):
                    translated_text = "Victory"
                elif is_call_me(results.right_hand_landmarks):
                    translated_text = "Call me"
                elif is_i_love_you(results.right_hand_landmarks):
                    translated_text = "I love you"
                elif is_good_luck(results.right_hand_landmarks):
                    translated_text = "Good luck"
                elif is_rock_on(results.right_hand_landmarks):
                    translated_text = "Rock on"
                elif is_i_want_to_talk(results.right_hand_landmarks):
                    translated_text = "I want to talk"
                elif is_care(results.right_hand_landmarks):
                    translated_text = "Care"
                elif is_pain(results.right_hand_landmarks):
                    translated_text = "Pain"
                elif is_victory(results.right_hand_landmarks):
                    translated_text = "Victory"
                elif is_loser(results.right_hand_landmarks):
                    translated_text = "Loser"
                elif is_hurts_a_lot(results.right_hand_landmarks):
                    translated_text = "Hurts a lot"
                elif is_ok(results.right_hand_landmarks):
                    translated_text = "OK"
                elif is_good_job(results.right_hand_landmarks):
                    translated_text = "Good job"
            
                else:
                    translated_text = "Waiting for Sign..."
            else:
                translated_text = "Waiting for Sign..."

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ======= Flask routes =======
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    global translated_text
    return jsonify({'text': translated_text})

if __name__ == "__main__":
    app.run(debug=True)
