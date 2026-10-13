from pynput import keyboard
import mujoco
import mujoco_viewer
import time

model = mujoco.MjModel.from_xml_path('Long_Manipulator.xml')
data = mujoco.MjData(model)
ctrl = data.ctrl
viewer = mujoco_viewer.MujocoViewer(model, data)

activation = [0.0, 0.0]

key_state = {'k': False, 'l': False}

def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        pass

def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()


segment_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"segment_{i+1}") for i in range(28)]

while True:
    for i, keys in enumerate([['k'], ['l']]):
        if any(key_state[k] for k in keys):
            activation[i] = min(activation[i] + 0.1, 1.0)
        else:
            activation[i] = max(activation[i] - 0.1, 0.0)

    ctrl[0], ctrl[1] = activation
    mujoco.mj_step(model, data)
    viewer.render()

    # Positions and orientations for ALL segments:
    segment_positions = [data.xpos[i].copy() for i in segment_ids]
    segment_orientations = [data.xmat[i].reshape(3, 3) for i in segment_ids]

    for idx, (pos, ori) in enumerate(zip(segment_positions, segment_orientations)):
        print(f"Segment {idx+1}: Pos = {pos}, Orientation =\n{ori}")



    # Shaft diameter in meters (75 mm = 0.075 m)
    D = 0.075  

    # Tendon lengths
    t1_length = data.ten_length[0]
    t2_length = data.ten_length[1]

    print(f"Tendon t1 length: {t1_length:.4f}, Tendon t2 length: {t2_length:.4f}")

    # Shaft angles (from each tendon independently)
    theta1_rad = (2 * t1_length) / D
    theta1_deg = theta1_rad * 180.0 / 3.14159

    theta2_rad = (2 * t2_length) / D
    theta2_deg = theta2_rad * 180.0 / 3.14159

    print(f"Shaft angle from t1: {theta1_rad:.4f} rad ({theta1_deg:.2f} deg)")
    print(f"Shaft angle from t2: {theta2_rad:.4f} rad ({theta2_deg:.2f} deg)")



    time.sleep(0.001)


