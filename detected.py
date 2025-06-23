import cv2
import threading
import numpy as np
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import mediapipe as mp

class MotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Detection")

        # Initialize the camera
        self.cap = cv2.VideoCapture(0)  # Change to 0 if using a built-in webcam

        # Create a label to display the camera feed
        self.label = Label(root)
        self.label.pack()

        # Start the video thread
        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.start()

        # Store the previous frame for motion detection
        self.previous_frame = None

        # Initialize MediaPipe pose model
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        # Close the camera when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def detect_motion(self, frame, gray):
        motion_detected = False
        motion_regions = []

        if self.previous_frame is None:
            self.previous_frame = gray
        else:
            delta_frame = cv2.absdiff(self.previous_frame, gray)
            threshold_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
            threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

            # Find contours of the detected motion
            contours, _ = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 1000:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                motion_detected = True
                motion_regions.append((x, y, w, h))

            # Update previous frame
            self.previous_frame = gray

        return motion_detected, motion_regions

    def detect_body_parts(self, frame):
        body_parts = []

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe pose model
        result = self.pose.process(rgb_frame)

        if result.pose_landmarks:
            for landmark in result.pose_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                body_parts.append((cx, cy))

                # Draw the landmarks on the frame
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        return body_parts

    def map_motion_to_body_part(self, motion_regions, body_parts):
        motion_part_map = []

        for (x, y, w, h) in motion_regions:
            for (px, py) in body_parts:
                if x < px < x + w and y < py < y + h:
                    motion_part_map.append((px, py))

        return motion_part_map

    def video_loop(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                # Convert the frame to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                # Detect motion
                motion_detected, motion_regions = self.detect_motion(frame, gray)

                # Detect body parts using pose estimation
                body_parts = self.detect_body_parts(frame)

                # Map detected motion to body parts
                moving_body_parts = self.map_motion_to_body_part(motion_regions, body_parts)

                # Annotate the frame with motion and body parts
                for (x, y) in moving_body_parts:
                    cv2.putText(frame, "Moving part", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Add text "Motion detected" if motion is detected
                if motion_detected:
                    cv2.putText(frame, "Motion detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Convert the frame to an image format Tkinter can use
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image = ImageTk.PhotoImage(image)

                # Update the label with the new frame
                self.label.config(image=image)
                self.label.image = image

    def on_close(self):
        # Release the camera and destroy the window
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.mainloop()
