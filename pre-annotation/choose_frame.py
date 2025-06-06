"""
Interactive frame viewer and saver

Défilez les frames d'une vidéo et cliquez pour sauvegarder la frame courante
"""

import cv2
import os
import argparse

# Variables globales pour la frame courante
current_frame = None
current_frame_number = 0
output_dir = None


def save_frame(event, x, y, flags, param):
    """
    Callback souris : enregistre la frame courante dans le dossier output_dir
    """
    global current_frame, current_frame_number, output_dir
    if event == cv2.EVENT_LBUTTONDOWN and current_frame is not None:
        # Génération du nom de fichier avec zéro-padded sur 6 chiffres
        filename = os.path.join(output_dir, f"frame_{current_frame_number:06d}.jpg")
        cv2.imwrite(filename, current_frame)
        print(f"Saved {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame viewer and saver")
    parser.add_argument('video_path', type=str, help="Path to the video file")
    args = parser.parse_args()

    video_path = args.video_path
    # Création du dossier de sortie à partir du nom de la vidéo
    basename = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = basename
    os.makedirs(output_dir, exist_ok=True)

    # Ouverture de la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_number = 0

    cv2.namedWindow("Frame Viewer")
    cv2.setMouseCallback("Frame Viewer", save_frame)

    while True:
        # Positionnement sur la frame souhaitée
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Copie de la frame courante pour la sauvegarde
        current_frame = frame.copy()

        # Affichage du numéro de frame
        display = frame.copy()
        cv2.putText(display,
                    f"Frame: {current_frame_number}/{total_frames-1}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Frame Viewer", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            # Quitter
            break
        elif key in [ord('n'), 83]:  # 'n' ou flèche droite
            current_frame_number = min(current_frame_number + 1, total_frames - 1)
        elif key in [ord('b'), 81]:  # 'b' ou flèche gauche
            current_frame_number = max(current_frame_number - 1, 0)
        elif key == ord('N'):  # +10 frames
            current_frame_number = min(current_frame_number + 10, total_frames - 1)
        elif key == ord('B'):  # -10 frames
            current_frame_number = max(current_frame_number - 10, 0)

    cap.release()
    cv2.destroyAllWindows()
