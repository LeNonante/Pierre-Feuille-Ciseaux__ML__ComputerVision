import cv2
import mediapipe as mp
import csv




#On recupere le type a collecter
type=str(input("Quel signe allez vous apprendre (p : Pierre, f : Feuille, c : Ciseaux) ? "))
dicoType = {"p":"Pierre", "f":"Feuille", "c":"Ciseaux"}
if type not in dicoType.keys():
    print("Type inconnu")
    exit()
else :
    type = dicoType[type]

#Fonction pour sauvegarder les donnees
def saveDonnees(donnees):
    print("Données sauvegardées pour le signe : "+ type)
    with open("donnees.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [type]
        for lm in donnees.landmark:
            row.extend([lm.x, lm.y, lm.z])  # Ajoute les coordonnées X, Y, Z
        writer.writerow(row)
    
    

# Initialisation de Mediapipe pour la détection des mains
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialisation de la capture vidéo
cap = cv2.VideoCapture(0)

# Configure la fenêtre pour l'affichage en plein écran
cv2.namedWindow("Detection des mains", cv2.WINDOW_NORMAL)

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

    # Afficher la vidéo avec OpenCV
    cv2.imshow("Detection des mains", frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        saveDonnees(hand_landmarks)
    
    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()