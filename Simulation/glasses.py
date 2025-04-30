import pybullet as p 
import pybullet_data
import threading
import time
import cv2
import numpy as np
import sys

def init_simulation(use_gui=True):
    if use_gui:
        physics_client = p.connect(p.GUI)
    else:
        physics_client = p.connect(p.DIRECT)
    
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0, 0, 0])
    p.loadURDF("cube_small.urdf", [0.5, 0, 0.5])

    glasses = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.05, 0.01]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.05, 0.01], rgbaColor=[0, 0, 1, 0.5]),
        basePosition=[0, 0, 1.2]
    )

    return glasses

def init_ai_services():
    print("[INIT] Initializing AI services...")
    # Placeholder for your actual models
    model = None
    engine = None
    recognizer = None
    print("[INIT] AI services initialized successfully.")
    return model, engine, recognizer

def get_glasses_view(glasses):
    glasses_pos, _ = p.getBasePositionAndOrientation(glasses)
    cam_pos = [glasses_pos[0], glasses_pos[1] - 0.3, glasses_pos[2] + 0.05]
    target_pos = [glasses_pos[0], glasses_pos[1], glasses_pos[2]]

    view = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(60, 1, 0.1, 10)
    width, height, rgb_img, _, _ = p.getCameraImage(320, 240, view, proj)

    # Convert to proper numpy array and cast to uint8
    rgb_array = np.reshape(rgb_img, (height, width, 4))[:, :, :3].astype(np.uint8)

    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def vision_loop(running_flag):
    print("[THREAD] Vision thread starting...")
    try:
        model, engine, recognizer = init_ai_services()
        glasses = init_simulation(use_gui=False)

        while running_flag.is_set():
            try:
                frame = get_glasses_view(glasses)
                # Insert your vision logic here
                cv2.imshow("Glasses View", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(f"[VISION ERROR] {e}")
                break
    finally:
        print("[THREAD] Vision thread exiting.")
        cv2.destroyAllWindows()
        p.disconnect()

def physics_loop(running_flag):
    print("[THREAD] Physics thread starting...")
    try:
        glasses = init_simulation(use_gui=True)
        print("[THREAD] Physics simulation running...")
        while running_flag.is_set():
            p.stepSimulation()
            time.sleep(1. / 240.)
    finally:
        print("[THREAD] Physics thread exiting.")
        p.disconnect()

if __name__ == "__main__":
    print("[MAIN] Starting application...")
    running_flag = threading.Event()
    running_flag.set()

    try:
        physics_thread = threading.Thread(target=physics_loop, args=(running_flag,))
        vision_thread = threading.Thread(target=vision_loop, args=(running_flag,))

        physics_thread.start()
        vision_thread.start()

        while running_flag.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt received. Shutting down...")

    finally:
        print("[MAIN] Cleaning up...")
        running_flag.clear()
        physics_thread.join(timeout=1)
        vision_thread.join(timeout=1)
        print("[MAIN] Exit complete.")
        sys.exit()
