# Smart Attendance System

A facial recognition-based attendance system that automates student record-keeping using real-time webcam feed and OpenCV.

## What it does
- Detects faces in real-time using webcam
- Marks attendance with name, date, and time
- Saves records to a CSV file
- Prevents duplicate entries for the same day

## How to Run
```bash
pip install -r requirements.txt
python attendance.py
```

## Controls
- Press `s` to mark attendance when face is detected
- Press `q` to quit

## Tech Stack
- Python, OpenCV, Pandas