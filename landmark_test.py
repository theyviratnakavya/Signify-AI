
import cv2
from hand_detector import HandDetector


def main():

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    detector = HandDetector()

    while True:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        hand = detector.detect(frame)

        if hand:

            h, w, _ = frame.shape

            # Draw all 21 landmarks
            for i, landmark in enumerate(hand):

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    str(i),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        cv2.imshow(
            "Signify AI - 21 Hand Landmarks",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


