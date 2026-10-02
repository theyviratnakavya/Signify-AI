
import cv2
import numpy as np
from hand_detector import HandDetector


def main():

    # Camera
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    detector = HandDetector()

    # Persistent drawing canvas
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    previous_x = None
    previous_y = None

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        hand = detector.detect(frame)

        if hand:

            # Index fingertip
            index_tip = hand[8]

            h, w, _ = frame.shape

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            # Green fingertip
            cv2.circle(
                frame,
                (x, y),
                10,
                (0, 255, 0),
                -1
            )

            # Draw on persistent canvas
            if previous_x is not None:

                cv2.line(
                    canvas,
                    (previous_x, previous_y),
                    (x, y),
                    (0, 0, 255),
                    8
                )

            previous_x = x
            previous_y = y

            cv2.putText(
                frame,
                "DRAWING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            # Stop connecting lines when hand disappears
            previous_x = None
            previous_y = None

            cv2.putText(
                frame,
                "SHOW YOUR HAND",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # Combine camera + canvas
        output = cv2.addWeighted(
            frame,
            1.0,
            canvas,
            1.0,
            0
        )

        cv2.imshow(
            "Signify AI - Air Drawing",
            output
        )

        key = cv2.waitKey(1) & 0xFF

        # C = Clear
        if key == ord("c"):

            canvas = np.zeros(
                (480, 640, 3),
                dtype=np.uint8
            )

            previous_x = None
            previous_y = None

        # Q = Quit
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()





