import datetime
import random
from ultralytics import YOLO
import cv2
from addon import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import json
import numpy as np

CONFIDENCE_THRESHOLD = 0.75
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLACK = (0, 0, 0)

# Initialize the video capture object
video_cap = cv2.VideoCapture("C:\\Users\\binor\\Desktop\\MACV AI\\macv-obj-tracking-video.mp4")
# Initialize the video writer object
writer = create_video_writer(video_cap, "output.mp4")

# Load the pre-trained YOLOv8n model
model = YOLO("yolov8s.pt")
tracker = DeepSort(max_age=43, n_init=5, max_iou_distance=0.8)

# Initialize metrics storage
object_times = {}  
id_mapping = {}  
next_custom_id = 1  
gone_ids = set()  

# Dictionary to store object colors and trails
object_colors = {}
object_trails = {}  

# Create a blank overlay for fading trails
trail_overlay = None

while True:
    start = datetime.datetime.now()

    ret, frame = video_cap.read()

    if not ret:
        break

    if trail_overlay is None:
        trail_overlay = np.zeros_like(frame, dtype=np.uint8)  # Initialize the trail overlay

    # Run the YOLO model on the frame
    detections = model(frame)[0]

    # Initialize the list of bounding boxes and confidences
    results = []

    

    # Loop over the detections
    for data in detections.boxes.data.tolist():
        confidence = data[4]

        # Filter out weak detections by ensuring the confidence is greater than the minimum threshold
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue

        xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
        class_id = int(data[5])
        results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

    

    # Update the tracker with the new detections
    tracks = tracker.update_tracks(results, frame=frame)

    # Loop over the tracks
    for track in tracks:
        if not track.is_confirmed():
            continue

        deepsort_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

        # If the object has gone permanently, don't reuse the ID
        if deepsort_id in gone_ids:
            continue

        # Assign a custom ID based on first-come-first-serve rule
        if deepsort_id not in id_mapping:
            id_mapping[deepsort_id] = next_custom_id
            next_custom_id += 1

        custom_id = id_mapping[deepsort_id]

        # Assign a random color if not already assigned
        if custom_id not in object_colors:
            random_color = tuple(random.randint(0, 255) for _ in range(3))  # Random color
            object_colors[custom_id] = random_color

        color = object_colors[custom_id]

        # Draw bounding box and centroid
        centroid = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)  # Draw the bounding box with random color
        cv2.circle(frame, centroid, 5, color, -1)  # Draw the centroid as a small colored point

        # Draw ID with random color and black background
        text = f"ID {custom_id}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x, text_y = xmin + 5, ymin - 8
        cv2.rectangle(frame, (text_x - 2, text_y - text_size[1] - 2),
                      (text_x + text_size[0] + 2, text_y + 2), BLACK, -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Update object times
        if custom_id not in object_times:
            object_times[custom_id] = 0  # Initialize object time

        # Record the time spent in the video (in seconds)
        object_times[custom_id] += (end - start).total_seconds()

        # Display the time spent by the object in seconds
        time_text = f"Time: {object_times[custom_id]:.2f}s"
        time_size = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        time_x, time_y = xmin, ymin - 40
        cv2.rectangle(frame, (time_x - 2, time_y - time_size[1] - 2),
                      (time_x + time_size[0] + 2, time_y + 2), BLACK, -1)
        cv2.putText(frame, time_text, (time_x, time_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Update the trail
        if custom_id not in object_trails:
            object_trails[custom_id] = []

        object_trails[custom_id].append(centroid)

        # Limit the trail length to avoid excessive memory usage
        if len(object_trails[custom_id]) > 30:
            object_trails[custom_id].pop(0)

    # Draw fading trails on the overlay
    trail_overlay = cv2.addWeighted(trail_overlay, 0.9, np.zeros_like(trail_overlay), 0, 0)  # Fade old trails
    for custom_id, trail in object_trails.items():
        for i in range(1, len(trail)):
            cv2.line(trail_overlay, trail[i - 1], trail[i], object_colors[custom_id], 2)

    # Blend the trail overlay with the current frame
    frame = cv2.addWeighted(frame, 1, trail_overlay, 0.6, 0)

    # End time to compute FPS
    end = datetime.datetime.now()

    # Show the time it took to process one frame
    print(f"Time to process 1 frame: {(end - start).total_seconds() * 1000:.0f} milliseconds")

    # Calculate the FPS and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 8)

    # Show the frame on the screen
    cv2.imshow("Frame", frame)
    writer.write(frame)

    if cv2.waitKey(1) == ord("q"):
        break

video_cap.release()
writer.release()
cv2.destroyAllWindows()

