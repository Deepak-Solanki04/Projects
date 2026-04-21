import cv2
import pandas as pd
import os
from datetime import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

ATTENDANCE_FILE = "attendance.csv"

def mark_attendance(name="Student"):
    """Log attendance entry to CSV."""
    now = datetime.now()
    entry = {
        "Name": name,
        "Date": now.strftime("%Y-%m-%d"),
        "Time": now.strftime("%H:%M:%S"),
        "Status": "Present"
    }

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        # Avoid duplicate entry for same person same day
        already_marked = ((df["Name"] == name) & (df["Date"] == entry["Date"])).any()
        if already_marked:
            return False
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(ATTENDANCE_FILE, index=False)
    return True

def run_attendance():
    cap = cv2.VideoCapture(0)
    print("Smart Attendance System Running... Press 'q' to quit, 's' to mark attendance.")

    marked_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 100), 2)
            cv2.putText(frame, "Face Detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 100), 2)

        # Status bar
        cv2.putText(frame, f"Marked Today: {marked_count} | Press 's' to mark | 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Smart Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            if len(faces) > 0:
                name = input("Enter student name: ")
                success = mark_attendance(name)
                if success:
                    print(f"Attendance marked for {name}")
                    marked_count += 1
                else:
                    print(f"{name} already marked today.")
            else:
                print("No face detected. Please position your face in frame.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. Total marked: {marked_count}")
    print(f"Attendance saved to {ATTENDANCE_FILE}")

if __name__ == "__main__":
    run_attendance()