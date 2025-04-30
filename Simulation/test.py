import pybullet as p 
import pybullet_data
import numpy as np
import cv2
import time
import google.generativeai as genai
from PIL import Image
import speech_recognition as sr
import pyttsx3
import threading
import sys
from queue import Queue

# ======================
# 1. PyBullet Setup
# ======================
def init_simulation(use_gui=True):
    if use_gui:
        physics_client = p.connect(p.GUI)
    else:
        physics_client = p.connect(p.DIRECT)

    p.setGravity(0, 0, -9.8, physicsClientId=physics_client)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)

    p.loadURDF("plane.urdf", physicsClientId=physics_client)
    p.loadURDF("table/table.urdf", [0, 2, 0], physicsClientId=physics_client)
    p.loadURDF("cube_small.urdf", [0.5, 0, 0.5], physicsClientId=physics_client)

    stand_id = p.loadURDF("rotating_glasses_stand.urdf", [0, 0, 0], useFixedBase=True)

    texture_id = p.loadTexture(r"C:\\Users\\DELL\\OneDrive\\Desktop\\Simulation\\Intro.png")
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.75, 0.01, 1.0],
        rgbaColor=[1, 1, 1, 1]
    )
    board_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, 2, 1.0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi])  # Flipped 180 degrees to fix inverted image
    )
    p.changeVisualShape(board_id, -1, textureUniqueId=texture_id)

    return stand_id, physics_client

# ======================
# 2. AI + Voice Setup
# ======================
COMMANDS = {
    "read text": "Reading the image text now.",
    "any obstacle": "Obstacle detected: A table is in front of you at 2 meters.",
    "thank you": "I'm glad to help!"
}

def init_ai_services():
    genai.configure(api_key="YOUR_GEMINI_KEY")
    recognizer = sr.Recognizer()
    return genai.GenerativeModel('gemini-1.5-flash'), recognizer

def speak(text_queue):
    engine = pyttsx3.init()
    while True:
        text = text_queue.get()
        if text == "__exit__":
            break
        print(f"[GLASSES]: {text}")
        engine.say(text)
        engine.runAndWait()
        text_queue.task_done()

# ======================
# 3. Camera View
# ======================
def get_glasses_view(stand_id, physics_client):
    camera_link_index = 1
    link_state = p.getLinkState(stand_id, camera_link_index, computeForwardKinematics=True, physicsClientId=physics_client)
    if link_state is None:
        raise ValueError("getLinkState returned None. Check the link index.")

    camera_pos = link_state[0]
    camera_ori = link_state[1]

    rot_matrix = p.getMatrixFromQuaternion(camera_ori)
    forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
    up = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]
    target_pos = [camera_pos[i] + forward[i] for i in range(3)]

    view_matrix = p.computeViewMatrix(camera_pos, target_pos, up)
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=10)

    width, height, rgb_img, _, _ = p.getCameraImage(
        width=320,
        height=240,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        physicsClientId=physics_client
    )

    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3].astype(np.uint8)
    return rgb_array

# ======================
# 4. Threads
# ======================
def physics_loop(running_flag, stand_id, physics_client):
    print("[THREAD] Physics thread starting...")
    try:
        while running_flag.is_set():
            p.stepSimulation(physicsClientId=physics_client)
            time.sleep(1. / 240.)
    finally:
        print("[THREAD] Physics thread exiting.")

def vision_loop(running_flag, stand_id, physics_client, text_queue):
    print("[THREAD] Vision thread starting...")
    try:
        model, recognizer = init_ai_services()
        spoken = False
        start_time = time.time()

        while running_flag.is_set():
            frame = get_glasses_view(stand_id, physics_client)
            if frame is not None:
                cv2.imshow("Glasses View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if not spoken and time.time() - start_time > 10:
                spoken = True
                simulated_text = (
                    "Welcome to the Shenovation Hackathon 2025. Presented by Team VisionX. "
                    "Our project introduces Smart Glasses designed for the visually impaired, "
                    "featuring real-time environment analysis, obstacle detection, and voice guidance. "
                    "By combining PyBullet simulation and Gemini AI, we enable a safer, more independent "
                    "navigation experience through intelligent assistive technology. This is more than a "
                    "hackathon project – it's a step towards inclusive innovation."
                )
                text_queue.put(simulated_text)
    finally:
        print("[THREAD] Vision thread exiting.")
        cv2.destroyAllWindows()

def handle_command(command, text_queue):
    if "read" in command and "text" in command:
        text_queue.put(COMMANDS["read text"])
        text_queue.put(
            "Welcome to the Shenovation Hackathon 2025. Presented by Team VisionX. "
            "Our project introduces Smart Glasses designed for the visually impaired, "
            "featuring real-time environment analysis, obstacle detection, and voice guidance. "
            "By combining PyBullet simulation and Gemini AI, we enable a safer, more independent "
            "navigation experience through intelligent assistive technology. This is more than a "
            "hackathon project – it's a step towards inclusive innovation."
        )
    elif "obstacle" in command:
        text_queue.put(COMMANDS["any obstacle"])
    elif "thank you" in command:
        text_queue.put(COMMANDS["thank you"])
    else:
        text_queue.put("Sorry, I didn't understand that command.")

def voice_command_loop(running_flag, text_queue, recognizer):
    print("[THREAD] Voice Command thread starting...")
    try:
        mic_index = None

        print("\nAvailable microphones:")
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"{i}: {name}")

        with sr.Microphone(device_index=mic_index) as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=1)

            while running_flag.is_set():
                try:
                    print("[VOICE] Listening for commands...")
                    audio = recognizer.listen(mic, timeout=10, phrase_time_limit=6)
                    command = recognizer.recognize_google(audio).lower()
                    print(f"[VOICE] You said: {command}")
                    handle_command(command, text_queue)

                except sr.WaitTimeoutError:
                    print("[VOICE] Timeout: No speech detected.")
                except sr.UnknownValueError:
                    print("[VOICE] Could not understand audio.")
                except sr.RequestError as e:
                    print(f"[VOICE] Google API error: {e}")
                except Exception as e:
                    print(f"[VOICE] Unexpected error: {e}")
    finally:
        print("[THREAD] Voice Command thread exiting.")

# ======================
# 5. Main
# ======================
if __name__ == "__main__":
    print("[MAIN] Starting application...")
    running_flag = threading.Event()
    running_flag.set()

    try:
        stand_id, physics_client = init_simulation(use_gui=True)
        text_queue = Queue()

        physics_thread = threading.Thread(target=physics_loop, args=(running_flag, stand_id, physics_client))
        vision_thread = threading.Thread(target=vision_loop, args=(running_flag, stand_id, physics_client, text_queue))
        model, recognizer = init_ai_services()
        voice_thread = threading.Thread(target=voice_command_loop, args=(running_flag, text_queue, recognizer))
        tts_thread = threading.Thread(target=speak, args=(text_queue,), daemon=True)

        physics_thread.start()
        vision_thread.start()
        voice_thread.start()
        tts_thread.start()

        while running_flag.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt received. Shutting down...")
    finally:
        print("[MAIN] Cleaning up...")
        running_flag.clear()
        text_queue.put("__exit__")
        physics_thread.join(timeout=1)
        vision_thread.join(timeout=1)
        voice_thread.join(timeout=1)
        tts_thread.join(timeout=1)
        p.disconnect(physicsClientId=physics_client)
        print("[MAIN] Exit complete.")
        sys.exit()
