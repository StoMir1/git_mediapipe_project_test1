import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

mp_hands = mp.solutions.hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
finger_list = [8,12,16,20]
counter = 0

sequence = [3, 2, 1]
current_step = 0
mic_active = False


devices = AudioUtilities.GetAllDevices()

mic = None

for device in devices:
    print(device.FriendlyName)

    if device.FriendlyName and "Microphone Array" in device.FriendlyName:
        mic = device


# DEBUGGING — keep this for now
for device in devices:
    if device.FriendlyName:
        print(device.FriendlyName)
        print(type(device))
        print(dir(device))
        print("----------------")


if mic:
    print("Microphone found!")
else:
    print("Microphone not found!")


mic = AudioUtilities.GetMicrophone()

interface = mic.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))

print("Microphone ready!")

while True:

    _, frame = cap.read()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(frameRGB)

    counter = 0

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            landmark = hand.landmark

            for tip in finger_list:
                if landmark[tip].y < landmark[tip - 2].y:
                    counter += 1

            if landmark[4].x > landmark[3].x:
                counter += 1

    # Check 3 -> 2 -> 1
    if not mic_active and counter == sequence[current_step]:

        current_step += 1

        print("Detected:", counter)

        if current_step == len(sequence):

            mic_active = True

            volume.SetMute(0, None)

            print("MICROPHONE ACTIVATED!")

    cv2.putText(
    frame,
    str(counter),
    (50, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    3,
    (0, 255, 0),
    3
)

    if mic_active:
        cv2.putText(
            frame,
            "MIC ACTIVATED",
            (50, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()