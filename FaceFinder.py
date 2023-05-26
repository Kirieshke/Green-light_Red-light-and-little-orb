import cv2
import numpy as np
import time
from datetime import datetime
import threading
import os
import signal

class face_finder:
    def __init__(self):
        self.classifier = cv2.CascadeClassifier('files_for_learning/haarcascade_frontalface_default.xml')
        self.video_captured = cv2.VideoCapture(0)
        self.orb = cv2.ORB_create()
        self.old_frame = None
        self.disp_frame = None
        self.words = ["Red Light", "Green Light"]  # List of words to display
        self.text = "TutorialsPoint"
        self.coordinates = (100, 100)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255, 0, 255)
        self.thickness = 2
        self.update_interval = 5  # Update interval in seconds

    def open_initial_window(self):
        cv2.namedWindow("Frame with Keypoints")

    def update_text(self):
        while True:
            index = 0
            start_time = time.time()

            while time.time() - start_time < self.update_interval:
                text = self.words[index % len(self.words)]
                index += 1
                time.sleep(2)  # Adjust the delay to control the text update speed

                # Acquire a lock to update the shared text variable
                with threading.Lock():
                    self.text = text

    def process_frames(self):
        text_thread = threading.Thread(target=self.update_text)
        text_thread.start()

        while True:
            ret, frame = self.video_captured.read()
            frame = cv2.resize(frame, (640, 360))

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            detected_faces = self.classifier.detectMultiScale(gray_frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.old_frame is not None:
                diff_frame = gray - self.old_frame
                diff_frame -= diff_frame.min()
                self.disp_frame = np.uint8(255.0 * diff_frame / float(diff_frame.max()))
            self.old_frame = gray

            for (x, y, w, h) in detected_faces:
                if self.disp_frame is not None and np.mean(self.disp_frame[y:y+h, x:x+w]) > 135:
                    self.close_current_window()
                    self.open_lose_window()
                    break
                else:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                keypoints, descriptors = self.orb.detectAndCompute(gray, None)
                frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color)

                # Acquire a lock to read the shared text variable
                with threading.Lock():
                    text = self.text

                cv2.putText(frame_with_keypoints, text, self.coordinates, self.font, self.fontScale, self.color, self.thickness)
                cv2.imshow("Frame with Keypoints", frame_with_keypoints)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        text_thread.join()  # Wait for the text thread to finish before exiting

    def close_current_window(self):
        if cv2.getWindowProperty("Frame with Keypoints", cv2.WND_PROP_VISIBLE):
            cv2.destroyWindow("Frame with Keypoints")

    def open_lose_window(self):
        self.close_current_window()  # Close the current window before opening the "lose" window
        lose_screen = cv2.imread('img/lose.png')
        cv2.imshow("Lose Screen", lose_screen)
        cv2.waitKey(3)

    def cleanup(self):
        cv2.destroyAllWindows()
        self.video_captured.release()

    def find_body(self):
        self.open_initial_window()
        self.process_frames()
        self.cleanup()
