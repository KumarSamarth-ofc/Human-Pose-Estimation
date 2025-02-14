import cv2
import numpy as np

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width = 368
height = 368
inWidth = width
inHeight = height

# Load the model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
if net.empty():
    print("Error: Model did not load correctly!")
    exit()
else:
    print("Model loaded successfully!")

# Confidence threshold
thres = 0.2

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)  # Change this to 0 for webcam
if not cap.isOpened():
    print("Error: Webcam could not be opened!")
    exit()
else:
    print("Webcam loaded successfully!")

def pose_estimation(cap):
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame")
            break

        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        # Prepare input for the model
        net.setInput(cv2.dnn.blobFromImage(frame, 2.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))

        out = net.forward()
        out = out[:, :19, :, :]

        # Check the output shape and print debugging information
        print(f"Output shape: {out.shape}")

        points = []
        for i in range(len(BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            # Get the body part name using the index i
            body_part_name = list(BODY_PARTS.keys())[i]
            print(f"Body part: {body_part_name}, Confidence: {conf}, Point: {point}")
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > thres else None)

        # Draw points for debugging
        for i in range(len(BODY_PARTS)):
            if points[i]:
                cv2.circle(frame, points[i], 5, (0, 0, 255), -1)  # Red dot for detected keypoints

        # Draw pose connections
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]

            if partFrom in BODY_PARTS and partTo in BODY_PARTS:
                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]

                if points[idFrom] and points[idTo]:
                    cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                    cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        # Display the resulting frame
        cv2.imshow('Pose Estimation', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the pose estimation
pose_estimation(cap)
