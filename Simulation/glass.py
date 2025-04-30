import pybullet as p  
import pybullet_data
import numpy as np
import cv2
import time
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
import threading
import sys
from queue import Queue
import os

# ======================
# Configuration
# ======================
class Config:
    GEMINI_API_KEY = "YOUR_GEMINI_KEY"  # Replace with your actual key
    PHYSICS_TIMESTEP = 1./240.
    CAMERA_RESOLUTION = (320, 240)
    OBJECTS_ON_TABLE = [
        {"name": "cup", "position": [0.5, 2.0, 0.7], "color": [1, 0, 0, 1], "shape": "cylinder", "size": [0.05, 0.1]},
        {"name": "book", "position": [-0.3, 2.0, 0.7], "color": [0, 1, 0, 1], "shape": "box", "size": [0.15, 0.1, 0.02]},
        {"name": "apple", "position": [0.1, 2.0, 0.7], "color": [0, 0, 1, 1], "shape": "sphere", "size": [0.05]}
    ]

# ======================
# 1. PyBullet Simulation Setup
# ======================
def init_simulation(use_gui=True):
    try:
        physics_client = p.connect(p.GUI if use_gui else p.DIRECT)
        p.setGravity(0, 0, -9.8, physicsClientId=physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=physics_client)

        # Load environment
        p.loadURDF("plane.urdf", physicsClientId=physics_client)
        table_id = p.loadURDF("table/table.urdf", [0, 2, 0], physicsClientId=physics_client)
        
        # Load objects on table using primitive shapes
        object_ids = []
        for obj in Config.OBJECTS_ON_TABLE:
            if obj["shape"] == "box":
                col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=obj["size"])
                vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=obj["size"], rgbaColor=obj["color"])
            elif obj["shape"] == "sphere":
                col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=obj["size"][0])
                vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=obj["size"][0], rgbaColor=obj["color"])
            elif obj["shape"] == "cylinder":
                col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=obj["size"][0], height=obj["size"][1])
                vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=obj["size"][0], length=obj["size"][1], rgbaColor=obj["color"])
            
            obj_id = p.createMultiBody(baseMass=0.5, 
                                     baseCollisionShapeIndex=col_shape,
                                     baseVisualShapeIndex=vis_shape,
                                     basePosition=obj["position"])
            object_ids.append({"id": obj_id, "name": obj["name"]})
        
        # Load glasses stand
        stand_id = p.loadURDF("rotating_glasses_stand.urdf", [0, 0, 0], useFixedBase=True, physicsClientId=physics_client)
        
        return stand_id, table_id, object_ids, physics_client
    
    except Exception as e:
        print(f"Error initializing simulation: {e}")
        raise

# ======================
# 2. Camera View
# ======================
def get_glasses_view(stand_id, physics_client):
    # Camera is attached to the first link (index 1) of the stand
    link_state = p.getLinkState(stand_id, 1, computeForwardKinematics=True, physicsClientId=physics_client)
    camera_pos = link_state[0]
    camera_ori = link_state[1]
    
    # Calculate view matrix
    rot_matrix = p.getMatrixFromQuaternion(camera_ori)
    forward = [rot_matrix[0], rot_matrix[3], rot_matrix[6]]
    up = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]
    target_pos = [camera_pos[i] + forward[i] for i in range(3)]
    
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=target_pos,
        cameraUpVector=up
    )
    
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=60, aspect=1.0,
        nearVal=0.1, farVal=100.0
    )
    
    # Get camera image
    width, height, rgb_img, _, _ = p.getCameraImage(
        width=Config.CAMERA_RESOLUTION[0],
        height=Config.CAMERA_RESOLUTION[1],
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        physicsClientId=physics_client
    )
    
    # Process image for display
    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
    return rgb_array

# ======================
# 3. AI + Voice Services
# ======================
def init_ai_services():
    genai.configure(api_key=Config.GEMINI_API_KEY)
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
# 4. Object Detection
# ======================
def detect_objects(table_id, physics_client):
    objects_on_table = []
    for i in range(p.getNumBodies(physicsClientId=physics_client)):
        pos, _ = p.getBasePositionAndOrientation(i, physicsClientId=physics_client)
        if pos[2] > 0.5:  # Objects above ground level
            objects_on_table.append(i)
    return objects_on_table

def count_objects(table_id, physics_client):
    return len(detect_objects(table_id, physics_client))

def describe_scene(table_id, object_ids, physics_client):
    object_names = [obj["name"] for obj in object_ids]
    
    if not object_names:
        return "There are no objects on the table."
    elif len(object_names) == 1:
        return f"There is a {object_names[0]} on the table."
    else:
        return "There are several objects on the table: " + ", ".join(object_names[:-1]) + f" and {object_names[-1]}."

# ======================
# 5. Command Handling
# ======================
def handle_command(command, text_queue, table_id, object_ids, physics_client):
    command = command.lower()
    
    if "read" in command and "text" in command:
        text_queue.put("Reading text: Welcome to the Shenovation Hackathon 2025. Presented by Team VisionX. Our project introduces Smart Glasses designed for the Visually Impaired, featuring real-time environment analysis, obstacle detection, and voice guidance. By combining PyBullet simulation and Gemini AI, we enable a safer, more independent navigation experience through intelligent assistive technology. This is more than a hackathon project — it's a step towards inclusive innovation.")
    
    elif "obstacle" in command or "object" in command:
        count = count_objects(table_id, physics_client)
        if count > 0:
            text_queue.put(f"I detect {count} objects in front of you.")
            text_queue.put(describe_scene(table_id, object_ids, physics_client))
            text_queue.put(f"The objects are on a table 2 meters away from you.")
        else:
            text_queue.put("No obstacles detected in your immediate area.")
    
    elif "describe" in command or "what is" in command:
        text_queue.put(describe_scene(table_id, object_ids, physics_client))
    
    elif "count" in command and ("object" in command or "item" in command):
        text_queue.put(f"There are {count_objects(table_id, physics_client)} objects on the table.")
    
    elif "thank you" in command:
        text_queue.put("You're welcome! Is there anything else I can help with?")
    
    elif "help" in command:
        text_queue.put("Available commands are: read text, any obstacles, describe objects, count objects, thank you")
    
    else:
        text_queue.put("I didn't understand that command. Try saying 'help' for available commands.")

# ======================
# 6. Thread Functions
# ======================
def physics_loop(running_flag, stand_id, physics_client):
    print("[THREAD] Physics thread starting...")
    try:
        while running_flag.is_set():
            p.stepSimulation(physicsClientId=physics_client)
            time.sleep(Config.PHYSICS_TIMESTEP)
    finally:
        print("[THREAD] Physics thread exiting.")

def vision_loop(running_flag, stand_id, physics_client, text_queue):
    print("[THREAD] Vision thread starting...")
    try:
        while running_flag.is_set():
            frame = get_glasses_view(stand_id, physics_client)
            if frame is not None:
                cv2.imshow("Glasses View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cv2.destroyAllWindows()
        print("[THREAD] Vision thread exiting.")

def voice_command_loop_continuous(running_flag, text_queue, recognizer, table_id, object_ids, physics_client):
    print("[THREAD] Voice Command (Continuous) thread starting...")

    def callback(recognizer, audio):
        if not running_flag.is_set():
            return
        try:
            command = recognizer.recognize_google(audio)
            print(f"[USER]: {command}")
            text_queue.put(f"Processing: '{command}'")
            handle_command(command, text_queue, table_id, object_ids, physics_client)
            time.sleep(2)  # Give user time to prepare next command
        except sr.UnknownValueError:
            print("[VOICE] Could not understand audio")
            text_queue.put("Sorry, I didn't catch that.")
            time.sleep(1.5)
        except sr.RequestError as e:
            print(f"[VOICE] API error: {e}")
            text_queue.put(f"Speech recognition service error: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[VOICE] Unexpected error: {e}")
            text_queue.put("An error occurred while processing your voice command.")
            time.sleep(2)


    try:
        mic = sr.Microphone()
        with mic as source:
            print("[VOICE] Calibrating microphone...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("[VOICE] Microphone calibrated.")

        stop_listening = recognizer.listen_in_background(mic, callback, phrase_time_limit=10)
        text_queue.put("Voice command system is active and listening...")

        while running_flag.is_set():
            time.sleep(0.1)

        stop_listening(wait_for_stop=False)

    except Exception as e:
        print(f"[VOICE THREAD ERROR]: {e}")
        text_queue.put("Voice system failed to start.")
    finally:
        print("[THREAD] Voice Command (Continuous) thread exiting.")



# ======================
# 7. Main Application
# ======================
def main():
    print("[MAIN] Starting Smart Glasses Simulation...")
    running_flag = threading.Event()
    running_flag.set()
    text_queue = Queue()
    
    try:
        # Initialize simulation and AI
        stand_id, table_id, object_ids, physics_client = init_simulation()
        model, recognizer = init_ai_services()
        
        # Start threads
        threads = [
            threading.Thread(target=physics_loop, args=(running_flag, stand_id, physics_client)),
            threading.Thread(target=vision_loop, args=(running_flag, stand_id, physics_client, text_queue)),
            threading.Thread(target=voice_command_loop_continuous, args=(running_flag, text_queue, recognizer, table_id, object_ids, physics_client)),
            threading.Thread(target=speak, args=(text_queue,), daemon=True)
        ]
        
        for t in threads:
            t.start()
        
        # Initial greeting
        text_queue.put("VisionX Smart Glasses initialized. How can I assist you today?")
        
        # Main loop
        while running_flag.is_set():
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
    except Exception as e:
        print(f"[MAIN] Error: {e}")
    finally:
        running_flag.clear()
        text_queue.put("__exit__")
        for t in threads:
            t.join(timeout=1)
        p.disconnect(physicsClientId=physics_client)
        print("[MAIN] Simulation ended.")

if __name__ == "__main__":
    main()