import cv2
import mediapipe as mp
import sounddevice as sd
from scipy.io.wavfile import write
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import os
from datetime import datetime


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


def activate_microphone():

    devices = AudioUtilities.GetAllDevices()

    for device in devices:

        name = device.FriendlyName

        if name is None:
            continue

        if "microphone" in name.lower() or "mic" in name.lower():

            try:

                volume = device._dev.Activate(
                    IAudioEndpointVolume._iid_,
                    CLSCTX_ALL,
                    None
                )

                volume = cast(
                    volume,
                    POINTER(IAudioEndpointVolume)
                )

                volume.SetMute(0, None)

                print("Microphone activated")
                return True

            except Exception as error:

                print("Microphone error:", error)

    print("Microphone not found")
    return False


def count_fingers(hand):

    fingers = 0

    if hand.landmark[8].y < hand.landmark[6].y:
        fingers += 1

    if hand.landmark[12].y < hand.landmark[10].y:
        fingers += 1

    if hand.landmark[16].y < hand.landmark[14].y:
        fingers += 1

    if hand.landmark[20].y < hand.landmark[18].y:
        fingers += 1

    if hand.landmark[4].x > hand.landmark[3].x:
        fingers += 1

    return fingers


SAMPLE_RATE = 44100
CHANNELS = 1

recording = False
audio_data = []


# Create recordings folder
os.makedirs("recordings", exist_ok=True)


def start_recording():

    global recording
    global audio_data

    print("Recording started!")

    recording = True

    audio_data = sd.rec(
        int(60 * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16"
    )


def stop_recording():

    global recording
    global audio_data

    print("Stopping recording...")

    sd.stop()

    recording = False

    filename = datetime.now().strftime(
        "recordings/recording_%Y-%m-%d_%H-%M-%S.wav"
    )

    write(
        filename,
        SAMPLE_RATE,
        audio_data
    )

    print("Recording saved to:", filename)

    audio_data = []


camera = cv2.VideoCapture(0)

numbers = []


while True:

    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)


    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        finger_count = count_fingers(hand)


        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )


        cv2.putText(
            frame,
            "Fingers: " + str(finger_count),
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )


        if finger_count == 3:

            if len(numbers) == 0:

                numbers.append(3)

                print("3")


        elif finger_count == 2:

            if len(numbers) == 1:

                numbers.append(2)

                print("2")


        elif finger_count == 1:

            if len(numbers) == 2:

                numbers.append(1)

                print("1")


        cv2.putText(
            frame,
            "Sequence: " + str(numbers),
            (30, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        if numbers == [3, 2, 1]:

            print("3 2 1 detected!")

            activate_microphone()


            if not recording:

                start_recording()

            else:

                stop_recording()


            numbers = []


    if recording:

        cv2.putText(
            frame,
            "RECORDING",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )


    cv2.imshow(
        "Microphone Control",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        if recording:
            stop_recording()

        break


camera.release()

cv2.destroyAllWindows()

hands.close()