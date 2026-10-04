import os
import sys
import math
import shutil
import arcpy
import arcpy.mapping
import numpy as np
from collections import deque, defaultdict
import common as C

SNAP_RADIUS = 2.0
SPLIT_TOL = 1.0
VALVE_TOL = 2.0
ARROW_LEN = 12.0
PIPE_GREY = (0.73, 0.73, 0.73)
PIPE_LIGHT = (0.80, 0.80, 0.80)
FLOW_GREEN = (0.18, 0.545, 0.34)
SOURCE_GREEN = (0.0, 0.39, 0.0)
FAULT_RED = (0.5, 0.0, 0.0)
HOT_RED = (0.86, 0.08, 0.24)
VALVE_ORANGE = (1.0, 0.647, 0.0)


def read_pipes(fc):
    out = []
    with arcpy.da.SearchCursor(fc, ["SHAPE@"]) as cur:
        for row in cur:
            g = row[0]
            for i in range(g.partCount):
                arr = g.getPart(i)
                pts = [(p.X, p.Y) for p in arr if p]
                if len(pts) >= 2:
                    out.append(pts)
    return out


def read_points(fc):
    out = []
    with arcpy.da.SearchCursor(fc, ["SHAPE@"]) as cur:
        for row in cur:
            g = row[0]
            out.append((g.centroid.X, g.centroid.Y))
    return out


def build_nodes(conn_pts):
    grid = {}
    node_xy = []

    def get_node(x, y):
        gx = int(math.floor(x / 2.0))
        gy = int(math.floor(y / 2.0))
        best = None
        bestd = 1e18
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for nid in grid.get((gx + dx, gy + dy), []):
                    nx, ny = node_xy[nid]
                    d = (nx - x) ** 2 + (ny - y) ** 2
                    if d < bestd:
                        bestd = d
                        best = nid
        if best is not None and bestd <= SNAP_RADIUS ** 2:
            return best
        nid = len(node_xy)
        node_xy.append((x, y))
        grid.setdefault((gx, gy), []).append(nid)
        return nid

    for (x, y) in conn_pts:
        get_node(x, y)
    return node_xy, get_node


def dist_pt_seg(px, py, a, b):
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(qx - px, qy - py), (qx, qy), t


def build_edges(pipes, conn_pts, get_node):
    edges = []
    for pts in pipes:
        n0 = get_node(pts[0][0], pts[0][1])
        n1 = get_node(pts[-1][0], pts[-1][1])
        splits = []
        for (cx, cy) in conn_pts:
            nid = get_node(cx, cy)
            if nid == n0 or nid == n1:
                continue
            for si in range(1, len(pts)):
                d, q, t = dist_pt_seg(cx, cy, pts[si - 1], pts[si])
                if d <= SPLIT_TOL and 0.001 < t < 0.999:
                    splits.append((si, t, q[0], q[1], nid))
                    break
        if not splits:
            edges.append((n0, n1, pts))
            continue
        splits.sort(key=lambda s: (s[0], s[1]))
        uniq = []
        for s in splits:
            if uniq and uniq[-1][4] == s[4] and uniq[-1][0] == s[0] and abs(uniq[-1][1] - s[1]) < 1e-9:
                continue
            uniq.append(s)
        cur_pts = [pts[0]]
        cur_node = n0
        for (si, t, sx, sy, nid) in uniq:
            sub = cur_pts + [p for p in pts[1:si] if p not in cur_pts] + [(sx, sy)]
            edges.append((cur_node, nid, sub))
            cur_pts = [(sx, sy)]
            cur_node = nid
        tail = [p for p in pts[uniq[-1][0]:] if p not in cur_pts]
        sub = cur_pts + tail
        if len(sub) >= 2:
            edges.append((cur_node, n1, sub))
    return edges


def build_tree(edges, src_node):
    adj = defaultdict(list)
    for ei, (n1, n2, pts) in enumerate(edges):
        adj[n1].append(ei)
        adj[n2].append(ei)
    parent = {src_node: None}
    order = [src_node]
    q = deque([src_node])
    while q:
        n = q.popleft()
        for ei in adj[n]:
            n1, n2, pts = edges[ei]
            other = n2 if n1 == n else n1
            if other not in parent:
                parent[other] = n
                q.append(other)
                order.append(other)
    depth = {src_node: 0}
    for n in order[1:]:
        depth[n] = depth[parent[n]] + 1
    return adj, parent, depth


def ancestors(parent, n):
    out = []
    while n is not None:
        out.append(n)
        n = parent[n]
    return out


def valve_nodes(junc_pts, node_xy):
    valves = set()
    for (x, y, typ) in junc_pts:
        if typ != "famen":
            continue
        for nid, (nx, ny) in enumerate(node_xy):
            if math.hypot(nx - x, ny - y) <= VALVE_TOL:
                valves.add(nid)
    return valves


def write_point(path, x, y, sr):
    arcpy.CreateFeatureclass_management(os.path.dirname(path), os.path.basename(path), "POINT", spatial_reference=sr)
    cur = arcpy.da.InsertCursor(path, ["SHAPE@"])
    cur.insertRow([arcpy.Point(x, y)])
    del cur


def write_polylines(path, parts, sr):
    arcpy.CreateFeatureclass_management(os.path.dirname(path), os.path.basename(path), "POLYLINE", spatial_reference=sr)
    cur = arcpy.da.InsertCursor(path, ["SHAPE@"])
    for pts in parts:
        arr = arcpy.Array([arcpy.Point(x, y) for (x, y) in pts])
        cur.insertRow([arcpy.Polyline(arr, sr)])
    del cur


def write_points(path, pts, sr):
    arcpy.CreateFeatureclass_management(os.path.dirname(path), os.path.basename(path), "POINT", spatial_reference=sr)
    cur = arcpy.da.InsertCursor(path, ["SHAPE@"])
    for (x, y) in pts:
        cur.insertRow([arcpy.Point(x, y)])
    del cur


def export_figures(node_xy, edges, parent, src, lca, valve, down_edges, down_nodes, gz_pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=10)

    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    for (n1, n2, pts) in edges:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PIPE_GREY, lw=0.7, zorder=1)
    for (n1, n2, pts) in edges:
        if n1 in parent and parent.get(n2) == n1:
            a = node_xy[n1]
            b = node_xy[n2]
        elif n2 in parent and parent.get(n1) == n2:
            a = node_xy[n2]
            b = node_xy[n1]
        else:
            continue
        mx = (a[0] + b[0]) / 2.0
        my = (a[1] + b[1]) / 2.0
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            continue
        ux = dx / length * ARROW_LEN
        uy = dy / length * ARROW_LEN
        ax.arrow(mx - ux / 2, my - uy / 2, ux, uy, head_width=4, head_length=6,
                 fc=FLOW_GREEN, ec=FLOW_GREEN, lw=0.3, zorder=3)
    ax.plot([node_xy[src][0]], [node_xy[src][1]], "s", color=SOURCE_GREEN, ms=9, zorder=6)
    for (x, y) in gz_pts:
        ax.plot([x], [y], "o", color=HOT_RED, ms=5, zorder=6)
    ax.plot([node_xy[lca][0]], [node_xy[lca][1]], "*", color=HOT_RED, ms=16, zorder=7)
    if valve is not None:
        ax.plot([node_xy[valve][0]], [node_xy[valve][1]], "D", color=VALVE_ORANGE, ms=7, zorder=7)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color=PIPE_GREY, lw=1),
               plt.Line2D([0], [0], marker=">", color="w", markerfacecolor=FLOW_GREEN, ms=6),
               plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=SOURCE_GREEN, ms=7),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=HOT_RED, ms=6),
               plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=HOT_RED, ms=12),
               plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=VALVE_ORANGE, ms=6)]
    labels = [C.LBL_PIPE, C.LBL_FLOW, C.LBL_SOURCE, C.LBL_FAULT, C.LBL_BURST, C.LBL_VALVE]
    ax.legend(handles, labels, loc="lower right", prop=font)
    fig.savefig(C.FIG6_1, dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 10), dpi=150)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    for (n1, n2, pts) in edges:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PIPE_LIGHT, lw=0.6, zorder=1)
    for ei in down_edges:
        n1, n2, pts = edges[ei]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=HOT_RED, lw=1.8, zorder=3)
    for n in down_nodes:
        ax.plot([node_xy[n][0]], [node_xy[n][1]], "o", color=HOT_RED, ms=2.5, zorder=4)
    ax.plot([node_xy[src][0]], [node_xy[src][1]], "s", color=SOURCE_GREEN, ms=9, zorder=6)
    for (x, y) in gz_pts:
        ax.plot([x], [y], "o", color=FAULT_RED, ms=5, zorder=6)
    ax.plot([node_xy[lca][0]], [node_xy[lca][1]], "*", color=HOT_RED, ms=16, zorder=7)
    if valve is not None:
        ax.plot([node_xy[valve][0]], [node_xy[valve][1]], "D", color=VALVE_ORANGE, ms=8, zorder=7)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color=PIPE_LIGHT, lw=1),
               plt.Line2D([0], [0], color=HOT_RED, lw=2),
               plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=SOURCE_GREEN, ms=7),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=FAULT_RED, ms=6),
               plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=HOT_RED, ms=12),
               plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=VALVE_ORANGE, ms=6)]
    labels = [C.LBL_PIPE, C.LBL_AFFECTED, C.LBL_SOURCE, C.LBL_FAULT, C.LBL_BURST2, C.LBL_VALVE2]
    ax.legend(handles, labels, loc="lower right", prop=font)
    fig.savefig(C.FIG6_2, dpi=150)
    plt.close(fig)


def export_arcmap():
    mxd = arcpy.mapping.MapDocument(C.MXD_GAS)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    for shp in [C.GAS_BURST, C.GAS_VALVE, C.GAS_PIPES, C.GAS_NODES]:
        arcpy.mapping.AddLayer(df, arcpy.mapping.Layer(shp), "TOP")
    arcpy.mapping.ExportToPNG(mxd, C.FIG6_3, resolution=150)
    del mxd
    if os.path.exists(C.MXD_GAS_OUT):
        os.remove(C.MXD_GAS_OUT)
    shutil.copyfile(C.MXD_GAS, C.MXD_GAS_OUT)
    mxd = arcpy.mapping.MapDocument(C.MXD_GAS_OUT)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    for shp in [C.GAS_BURST, C.GAS_VALVE, C.GAS_PIPES, C.GAS_NODES]:
        arcpy.mapping.AddLayer(df, arcpy.mapping.Layer(shp), "TOP")
    mxd.save()
    del mxd


def main():
    arcpy.env.overwriteOutput = True
    ds = C.MDB_GAS + "/gas_network"
    sys.stderr.write("exp5 read data\n")
    pipes = read_pipes(ds + "/line") + read_pipes(ds + "/diyaline")
    junc_pts = []
    for fc, typ in [("santong", "santong"), ("sitong", "sitong"), ("famen", "famen"),
                    ("tiaoyaqi", "tiaoyaqi"), ("qizhan", "qizhan")]:
        for (x, y) in read_points(ds + "/" + fc):
            junc_pts.append((x, y, typ))
    gz_pts = read_points(ds + "/guzhangdian")
    src_pt = read_points(ds + "/qizhan")[0]
    conn_pts = []
    for pts in pipes:
        conn_pts.append(pts[0])
        conn_pts.append(pts[-1])
    for (x, y, typ) in junc_pts:
        conn_pts.append((x, y))
    node_xy, get_node = build_nodes(conn_pts)
    edges = build_edges(pipes, conn_pts, get_node)
    src_node = get_node(src_pt[0], src_pt[1])
    adj, parent, depth = build_tree(edges, src_node)
    sys.stderr.write("exp5 nodes=%d edges=%d reached=%d\n" % (len(node_xy), len(edges), len(parent)))
    fault_nodes = sorted({get_node(x, y) for (x, y) in gz_pts if get_node(x, y) in parent},
                         key=lambda n: -depth.get(n, 0))
    common_set = set(ancestors(parent, fault_nodes[0]))
    for n in fault_nodes[1:]:
        common_set &= set(ancestors(parent, n))
    common_sorted = sorted(common_set, key=lambda n: -depth.get(n, 0))
    lca = common_sorted[0]
    valves = valve_nodes(junc_pts, node_xy)
    valve = None
    n = lca
    while n is not None:
        if n in valves:
            valve = n
            break
        n = parent[n]
    down_nodes = set()
    down_edges = set()
    q = deque([valve])
    down_nodes.add(valve)
    while q:
        n = q.popleft()
        for ei in adj[n]:
            n1, n2, pts = edges[ei]
            other = n2 if n1 == n else n1
            if other in parent and parent.get(other) == n:
                down_edges.add(ei)
                if other not in down_nodes:
                    down_nodes.add(other)
                    q.append(other)
    total = 0.0
    for ei in down_edges:
        n1, n2, pts = edges[ei]
        for k in range(1, len(pts)):
            total += math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1])
    sys.stderr.write("exp5 burst=%d valve=%d affected edges=%d nodes=%d length=%.1f\n" % (
        lca, valve, len(down_edges), len(down_nodes), total))
    sr = arcpy.Describe(ds + "/line").spatialReference
    write_point(C.GAS_BURST, node_xy[lca][0], node_xy[lca][1], sr)
    write_point(C.GAS_VALVE, node_xy[valve][0], node_xy[valve][1], sr)
    write_polylines(C.GAS_PIPES, [edges[ei][2] for ei in sorted(down_edges)], sr)
    write_points(C.GAS_NODES, [node_xy[n] for n in sorted(down_nodes)], sr)
    export_figures(node_xy, edges, parent, src_node, lca, valve, down_edges, down_nodes, gz_pts)
    export_arcmap()
    sys.stderr.write("exp5 done\n")


if __name__ == "__main__":
    main()
