import cv2
import mediapipe as mp
import math

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = r"D:\python\term 3\mediapipe_project 1\pose_landmarker.task"

count = 0
stage = "UP"


def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c.y - b.y, c.x - b.x)
        - math.atan2(a.y - b.y, a.x - b.x)
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=
VisionRunningMode.VIDEO
,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


with PoseLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Webcam started.")
    print("Press Q to exit.")

    timestamp_ms = 0

    while True:

        success, frame = cap.read()

        if not success:
            print("Error: Could not read frame.")
            break

        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms += 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:

            landmarks = result.pose_landmarks[0]

            left_shoulder = landmarks[11]
            left_elbow = landmarks[13]
            left_wrist = landmarks[15]

            right_shoulder = landmarks[12]
            right_elbow = landmarks[14]
            right_wrist = landmarks[16]

            left_angle = calculate_angle(
                left_shoulder,
                left_elbow,
                left_wrist
            )

            right_angle = calculate_angle(
                right_shoulder,
                right_elbow,
                right_wrist
            )

            elbow_angle = (
                left_angle + right_angle
            ) / 2

            left_hip = landmarks[23]
            right_hip = landmarks[24]

            left_ankle = landmarks[27]
            right_ankle = landmarks[28]

            body_angle_left = calculate_angle(
                left_shoulder,
                left_hip,
                left_ankle
            )

            body_angle_right = calculate_angle(
                right_shoulder,
                right_hip,
                right_ankle
            )

            body_angle = (
                body_angle_left + body_angle_right
            ) / 2

            if elbow_angle > 160 and body_angle > 150:
                if stage == "DOWN":
                    count += 1

                stage = "UP"

            elif elbow_angle < 90 and body_angle > 140:
                stage = "DOWN"

            for landmark in landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                if (
                    0 <= x < width
                    and 0 <= y < height
                ):
                    
                    cv2.circle
                    (
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            connections = [
                (11, 12),

                (11, 13),
                (13, 15),

                (12, 14),
                (14, 16),

                (11, 23),
                (12, 24),

                (23, 24),

                (23, 25),
                (25, 27),

                (24, 26),
                (26, 28)
            ]

            for start_idx, end_idx in connections:

                p1 = landmarks[start_idx]
                p2 = landmarks[end_idx]

                x1 = int(p1.x * width)
                y1 = int(p1.y * height)
                x2 = int(p2.x * width)
                y2 = int(p2.y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

            cv2.putText(
                frame,
                f"Push-ups: {count}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Stage: {stage}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Elbow: {int(elbow_angle)}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Body: {int(body_angle)}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.imshow(
            "Push-Up Counter",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()