from pynput import keyboard
import mujoco
import mujoco_viewer
import time

model = mujoco.MjModel.from_xml_path('Long_Manipulator+Gripper.xml')
data = mujoco.MjData(model)
ctrl = data.ctrl
viewer = mujoco_viewer.MujocoViewer(model, data)

activation = [0.0, 0.0, 0.0, 0.0]  

key_state = {'k': False, 'l': False, 'x': False, 'z': False, 'c': False}
grasp_auto = False  # for tracking x-triggered grasp
grasp_auto_level = 0.0

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

a3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_3")
a4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_4")

while True:
    print(activation)

    # Control base actuators (A_1 and A_2) with 'k' and 'l'
    for i, keys in enumerate([['k'], ['l']]):
        if any(key_state[k] for k in keys):
            activation[i] = min(activation[i] + 0.1, 1.0)
        else:
            activation[i] = max(activation[i] - 0.1, 0.0)

    # Handle 'x' grasp pulse trigger
    if key_state['x']:
        grasp_auto = True
        grasp_auto_level = 1.0
        key_state['x'] = False  # avoid repeat while held

    if grasp_auto:
        grasp_auto_level = max(grasp_auto_level - 0.01, 0.0)
        activation[2] = grasp_auto_level
        activation[3] = grasp_auto_level
        if grasp_auto_level == 0.0:
            grasp_auto = False
    else:
        # Manual z/c control
        if key_state['z']:
            activation[2] = min(activation[2] + 0.001, 1.0)
            activation[3] = min(activation[3] + 0.001, 1.0)
        elif key_state['c']:
            activation[2] = max(activation[2] - 0.001, 0.0)
            activation[3] = max(activation[3] - 0.001, 0.0)

    # Apply control
    ctrl[0], ctrl[1], ctrl[2], ctrl[3] = activation

    print("Activation:", activation)
    print("Grip1 Length:", data.actuator_length[a3_id])
    print("Grip2 Length:", data.actuator_length[a4_id])


    mujoco.mj_step(model, data)
    viewer.render()
    time.sleep(0.0001)
    
