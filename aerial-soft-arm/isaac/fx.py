"""渲染特效共享库，影棚、材质、相机朝向、实例清理、视频流，全部纯函数供各渲染脚本复用。"""
import subprocess

import numpy as np


class StreamVideo:
    """逐帧直推 ffmpeg，不落盘存中间帧。"""

    def __init__(self, out_path, w, h, fps):
        self.w = w
        self.h = h
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "%dx%d" % (w, h), "-r", "%g" % fps, "-i", "-",
             "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
             str(out_path)], stdin=subprocess.PIPE)

    def write(self, rgb):
        try:
            self.proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
        except (BrokenPipeError, OSError):
            pass

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        self.proc.wait()


def dir_quat(eye, target):
    a = np.array([0.0, 0.0, -1.0])
    f = np.array(target, dtype=float) - np.array(eye, dtype=float)
    f = f / (np.linalg.norm(f) + 1e-12)
    c = float(np.dot(a, f))
    ax = np.cross(a, f)
    if c < -0.99999:
        return (1.0, 0.0, 0.0, 0.0)
    q = np.array([ax[0], ax[1], ax[2], 1.0 + c])
    q = q / np.linalg.norm(q)
    return tuple(q.tolist())


def build_studio(sim_utils, dome=300.0, key=5.0e4, fill=1.0e4, rim=2.2e4):
    world = sim_utils.DomeLightCfg(intensity=dome, color=(0.40, 0.43, 0.50))
    world.func("/World/EnvLight", world)
    floor = sim_utils.CuboidCfg(size=(40.0, 40.0, 0.1),
                                collision_props=sim_utils.CollisionPropertiesCfg())
    floor.func("/World/floor", floor, translation=(0.0, 0.0, -0.05))
    k = sim_utils.SphereLightCfg(intensity=key, radius=0.8,
                                 color=(1.0, 0.97, 0.92), enable_color_temperature=False)
    k.func("/World/key", k, translation=(2.0, -1.8, 3.6))
    f = sim_utils.SphereLightCfg(intensity=fill, radius=1.4,
                                 color=(0.75, 0.82, 1.0), enable_color_temperature=False)
    f.func("/World/fill", f, translation=(-2.8, -1.8, 1.9))
    r = sim_utils.SphereLightCfg(intensity=rim, radius=0.6,
                                 color=(0.85, 0.90, 1.0), enable_color_temperature=False)
    r.func("/World/rim", r, translation=(-1.5, 2.8, 3.0))


def clear_instancing(root_path):
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for _ in range(5):
        changed = 0
        p = stage.GetPrimAtPath(root_path)
        if p and p.IsInstance():
            p.SetInstanceable(False)
            changed += 1
        for prim in stage.Traverse():
            if prim.IsInstance():
                try:
                    prim.SetInstanceable(False)
                    changed += 1
                except Exception:
                    pass
        if changed == 0:
            return True
    return False


def bind_all(verbose=True):
    from pxr import Sdf, Usd, UsdShade

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    def make(key, color, metallic, roughness):
        path = "/World/Looks/" + key
        m = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(color))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        m.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
        return m

    lib = {
        "alu": make("alu", (0.68, 0.70, 0.72), 0.45, 0.38),
        "steel": make("steel", (0.60, 0.61, 0.64), 0.6, 0.30),
        "brass": make("brass", (0.72, 0.55, 0.24), 0.8, 0.34),
        "black": make("black", (0.045, 0.045, 0.05), 0.15, 0.58),
        "rubber": make("rubber", (0.035, 0.035, 0.038), 0.0, 0.88),
        "blue": make("blue", (0.10, 0.24, 0.62), 0.25, 0.38),
        "red": make("red", (0.66, 0.06, 0.06), 0.25, 0.38),
        "carbon": make("carbon", (0.06, 0.062, 0.068), 0.1, 0.45),
        "pa6": make("pa6", (0.10, 0.10, 0.11), 0.1, 0.62),
        "silicone": make("silicone", (0.80, 0.82, 0.84), 0.0, 0.55),
        "orange": make("orange", (0.88, 0.36, 0.02), 0.0, 0.38),
        "floor": make("floor", (0.055, 0.058, 0.065), 0.0, 0.30),
    }

    def pick(p):
        if "lattice" in p:
            return lib["alu"]
        if "sheath" in p or "bundle" in p:
            return lib["rubber"]
        if "truss_clamp" in p:
            return lib["steel"]
        if "truss_tensioners" in p:
            return lib["brass"]
        if "truss_base_ring" in p or "base_flange" in p or "standoff" in p or "mast" in p or "gs_" in p:
            return lib["alu"]
        if "finger_prox" in p and "finger_0" in p:
            return lib["blue"]
        if "finger_prox" in p:
            return lib["red"]
        if "finger_dist" in p and "finger_0" in p:
            return lib["red"]
        if "finger_dist" in p:
            return lib["blue"]
        if "finger_soft" in p:
            return lib["silicone"]
        if "finger_knuckle" in p:
            return lib["steel"]
        if "q_plate" in p or "q_arm_boom" in p or "q_landing" in p:
            return lib["carbon"]
        if "q_prop" in p:
            return lib["pa6"]
        if "q_motor" in p:
            return lib["steel"]
        if "q_gnss" in p:
            return lib["silicone"]
        if "grip_base" in p or "grip_links" in p or "q_battery" in p or "q_pdb" in p \
                or "q_fc" in p or "q_obc" in p or "q_vtx" in p or "q_rx" in p or "q_cam" in p \
                or "q_tray" in p:
            return lib["black"]
        if "p_body" in p or "/World/part" in p:
            return lib["orange"]
        if "/World/floor" in p or "World/ground" in p:
            return lib["floor"]
        return None

    n = 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() not in ("Mesh", "Cylinder", "Cube", "Sphere", "Capsule"):
            continue
        m = pick(str(prim.GetPath()))
        if m is None:
            continue
        try:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(m)
            n += 1
        except Exception:
            pass
    if verbose:
        print("[fx] 材质绑定 %d" % n, flush=True)
    return n
