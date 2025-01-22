import cv2
import mediapipe as mp

# Initialisation de Mediapipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

# Configure la fenêtre pour l'affichage en plein écran
cv2.namedWindow("Detection des mains", cv2.WINDOW_NORMAL)



def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]

    thumb_folded = thumb_tip.x < hand_landmarks.landmark[3].x
    index_folded = index_tip.y > hand_landmarks.landmark[6].y
    middle_folded = middle_tip.y > hand_landmarks.landmark[10].y
    ring_folded = ring_tip.y > hand_landmarks.landmark[14].y
    pinky_folded = pinky_tip.y > hand_landmarks.landmark[18].y

    if index_folded and middle_folded and ring_folded and pinky_folded:
        return "Pierre"
    elif not index_folded and not middle_folded and not ring_folded and not pinky_folded:
        return "Feuille"
    elif not index_folded and not middle_folded and ring_folded and pinky_folded:
        return "Ciseaux"
    else:
        return "Geste inconnu"


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Conversion de l'image en RGB pour Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des mains
    results = hands.process(rgb_frame)

    # Dessiner les landmarks des mains détectées
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Afficher la vidéo avec OpenCV
    cv2.imshow("Detection des mains", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(hand_landmarks)
    
    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()