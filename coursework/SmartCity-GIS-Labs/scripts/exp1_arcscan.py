import os
import sys
import math
import shutil
import arcpy
import arcpy.mapping
import numpy as np
from collections import deque, defaultdict
import common as C

RDP_EPS = 0.30
NOISE_LEN = 2.0
CLEAN_BOX = 42
CLEAN_AREA = 8


def label_components(fg):
    h, w = fg.shape
    parent = []
    row_runs = []
    run_id = 0
    for r in range(h):
        row = fg[r].astype(np.uint8)
        d = np.diff(np.concatenate(([0], row, [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0] - 1
        cur = []
        for i in range(len(starts)):
            c0 = int(starts[i])
            c1 = int(ends[i])
            parent.append(run_id)
            cur.append((c0, c1, run_id))
            run_id += 1
        row_runs.append(cur)
    for r in range(1, h):
        prev = row_runs[r - 1]
        cur = row_runs[r]
        j = 0
        for (c0, c1, rid) in cur:
            while j < len(prev) and prev[j][1] < c0 - 1:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] <= c1 + 1:
                ra = rid
                while parent[ra] != ra:
                    parent[ra] = parent[parent[ra]]
                    ra = parent[ra]
                rb = prev[k][2]
                while parent[rb] != rb:
                    parent[rb] = parent[parent[rb]]
                    rb = parent[rb]
                if ra != rb:
                    parent[rb] = ra
                k += 1
    labels = np.zeros((h, w), dtype=np.int32)
    roots = {}
    for r in range(h):
        for (c0, c1, rid) in row_runs[r]:
            root = rid
            while parent[root] != root:
                parent[root] = parent[parent[root]]
                root = parent[root]
            if root not in roots:
                roots[root] = len(roots) + 1
            labels[r, c0:c1 + 1] = roots[root]
    return labels, len(roots)


def clean_components(fg, labels, ncomp):
    ys, xs = np.where(fg)
    lab_fg = labels[ys, xs]
    order = np.argsort(lab_fg)
    lab_sorted = lab_fg[order]
    ys_sorted = ys[order]
    xs_sorted = xs[order]
    starts = np.searchsorted(lab_sorted, np.arange(1, ncomp + 2))
    keep = np.ones(ncomp + 1, dtype=np.uint8)
    removed = []
    for cid in range(1, ncomp + 1):
        i0 = int(starts[cid - 1])
        i1 = int(starts[cid])
        if i1 <= i0:
            keep[cid] = 0
            continue
        h = int(ys_sorted[i1 - 1] - ys_sorted[i0] + 1)
        ww = int(xs_sorted[i0:i1].max() - xs_sorted[i0:i1].min() + 1)
        area = i1 - i0
        if (h <= CLEAN_BOX and ww <= CLEAN_BOX) or area < CLEAN_AREA:
            keep[cid] = 0
            removed.append((int(ys_sorted[i0]), int(ys_sorted[i1 - 1]),
                            int(xs_sorted[i0:i1].min()), int(xs_sorted[i0:i1].max()), area))
    cleaned = fg & (keep[labels] > 0)
    return cleaned, removed


def neighbors8(P):
    p = np.zeros((P.shape[0] + 2, P.shape[1] + 2), dtype=np.uint8)
    p[1:-1, 1:-1] = P.astype(np.uint8)
    return (p[0:-2, 1:-1], p[0:-2, 2:], p[1:-1, 2:], p[2:, 2:],
            p[2:, 1:-1], p[2:, 0:-2], p[1:-1, 0:-2], p[0:-2, 0:-2])


def cross_number(seq):
    a = np.zeros(seq[0].shape, dtype=np.uint8)
    for i in range(8):
        a += (seq[i] == 0) & (seq[i + 1] == 1)
    return a


def thin(P):
    P = P.astype(np.uint8)
    while True:
        P2, P3, P4, P5, P6, P7, P8, P9 = neighbors8(P)
        b = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        a = cross_number([P2, P3, P4, P5, P6, P7, P8, P9, P2])
        m1 = (P == 1) & (b >= 2) & (b <= 6) & (a == 1) & (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
        P[m1] = 0
        P2, P3, P4, P5, P6, P7, P8, P9 = neighbors8(P)
        b = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        a = cross_number([P2, P3, P4, P5, P6, P7, P8, P9, P2])
        m2 = (P == 1) & (b >= 2) & (b <= 6) & (a == 1) & (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
        P[m2] = 0
        if not m1.any() and not m2.any():
            return P.astype(bool)


def reduce_simple_points(S):
    h, w = S.shape
    while True:
        P2, P3, P4, P5, P6, P7, P8, P9 = neighbors8(S)
        b = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9
        a = cross_number([P2, P3, P4, P5, P6, P7, P8, P9, P2])
        cand = S & (a == 1) & (b >= 2)
        if not cand.any():
            return S
        ys, xs = np.where(cand)
        blocked = np.zeros((h, w), dtype=bool)
        for r, c in zip(ys.tolist(), xs.tolist()):
            if blocked[r, c]:
                continue
            S[r, c] = False
            r0 = max(0, r - 2)
            r1 = min(h, r + 3)
            c0 = max(0, c - 2)
            c1 = min(w, c + 3)
            blocked[r0:r1, c0:c1] = True


def trace_paths(S):
    h, w = S.shape
    P2, P3, P4, P5, P6, P7, P8, P9 = neighbors8(S)
    a = cross_number([P2, P3, P4, P5, P6, P7, P8, P9, P2])
    a = a * S.astype(np.uint8)
    node_mask = S & ((a >= 3) | (a == 1))
    visited = np.zeros_like(S, dtype=bool)
    lab = np.zeros((h, w), dtype=np.int32) - 1
    clusters = []
    ys, xs = np.where(node_mask)
    for r0, c0 in zip(ys.tolist(), xs.tolist()):
        if visited[r0, c0]:
            continue
        cid = len(clusters)
        q = deque([(r0, c0)])
        visited[r0, c0] = True
        pix = []
        while q:
            r, c = q.popleft()
            pix.append((r, c))
            lab[r, c] = cid
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and node_mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
        clusters.append(pix)
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    used = set()

    def ekey(a1, b1):
        return (a1, b1) if a1 <= b1 else (b1, a1)

    def walk(start, nxt):
        path = [start, nxt]
        used.add(ekey(start, nxt))
        prev = start
        cur = nxt
        while True:
            r, c = cur
            if lab[r, c] >= 0:
                break
            cands = []
            for dr, dc in nbrs:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and S[nr, nc] and (nr, nc) != prev and ekey((r, c), (nr, nc)) not in used:
                    cands.append((nr, nc))
            if not cands:
                break
            if len(cands) > 1:
                pr, pc = prev
                inx = r - pr
                iny = c - pc
                best = None
                bestdot = -9
                for (nr, nc) in cands:
                    dy = nr - r
                    dx = nc - c
                    dot = (inx * dx + iny * dy) / (math.hypot(inx, iny) * math.hypot(dx, dy) + 1e-9)
                    if dot > bestdot:
                        bestdot = dot
                        best = (nr, nc)
                nxt2 = best
            else:
                nxt2 = cands[0]
            used.add(ekey((r, c), nxt2))
            path.append(nxt2)
            prev = cur
            cur = nxt2
        return path

    paths = []
    for cid, pix in enumerate(clusters):
        for (r, c) in pix:
            for dr, dc in nbrs:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and S[nr, nc] and lab[nr, nc] != cid and ekey((r, c), (nr, nc)) not in used:
                    paths.append(walk((r, c), (nr, nc)))
    ys2, xs2 = np.where(S)
    for r0, c0 in zip(ys2.tolist(), xs2.tolist()):
        for dr, dc in nbrs:
            nr = r0 + dr
            nc = c0 + dc
            if 0 <= nr < h and 0 <= nc < w and S[nr, nc] and ekey((r0, c0), (nr, nc)) not in used:
                p = walk((r0, c0), (nr, nc))
                if len(p) >= 2:
                    paths.append(p)
    return paths


def chain_paths(paths):
    node_pix = {}

    def get_node(p):
        if p not in node_pix:
            node_pix[p] = len(node_pix)
        return node_pix[p]

    for p in paths:
        get_node(p[0])
        get_node(p[-1])
    edges = []
    for p in paths:
        edges.append((get_node(p[0]), get_node(p[-1]), p))
    inc = defaultdict(list)
    for i, (a1, b1, p) in enumerate(edges):
        inc[a1].append(i)
        if b1 != a1:
            inc[b1].append(i)
    used = set()
    chains = []

    def edge_pixels(i, from_node):
        a1, b1, p = edges[i]
        if a1 == from_node and b1 != from_node:
            return list(p), b1
        if b1 == from_node and a1 != from_node:
            return list(reversed(p)), a1
        return list(p), from_node

    def walk_chain(start_node, first_edge):
        coords = []
        node = start_node
        ei = first_edge
        used.add(ei)
        while True:
            px, other = edge_pixels(ei, node)
            if coords:
                coords.extend(px[1:])
            else:
                coords.extend(px)
            remaining = [j for j in inc.get(other, []) if j not in used]
            if len(inc.get(other, [])) == 2 and len(remaining) == 1:
                node = other
                ei = remaining[0]
                used.add(ei)
                continue
            return coords

    starts = [n for n, eis in inc.items() if len(eis) != 2]
    for n in starts:
        for ei in inc[n]:
            if ei not in used:
                chains.append(walk_chain(n, ei))
    for n, eis in inc.items():
        for ei in eis:
            if ei not in used:
                chains.append(walk_chain(n, ei))
    kept = []
    for p in chains:
        length = 0.0
        for i in range(1, len(p)):
            length += math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1])
        deg0 = len(inc.get(node_pix.get(p[0], -1), []))
        deg1 = len(inc.get(node_pix.get(p[-1], -1), []))
        if length < NOISE_LEN and deg0 <= 1 and deg1 <= 1:
            continue
        kept.append(p)
    return kept


def rdp(points, eps):
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        ax, ay = points[i]
        bx, by = points[j]
        dx = bx - ax
        dy = by - ay
        l2 = dx * dx + dy * dy
        dmax = -1.0
        idx = -1
        for k in range(i + 1, j):
            px, py = points[k]
            if l2 == 0:
                dd = math.hypot(px - ax, py - ay)
            else:
                t = ((px - ax) * dx + (py - ay) * dy) / l2
                if t < 0:
                    t = 0.0
                if t > 1:
                    t = 1.0
                dd = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if dd > dmax:
                dmax = dd
                idx = k
        if dmax > eps and idx > 0:
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
    return [points[i] for i in range(len(points)) if keep[i]]


def build_features(chains, xmin, ymax, cw, ch):
    feats = []
    for p in chains:
        pts = []
        for (r, c) in p:
            xy = (xmin + (c + 0.5) * cw, ymax - (r + 0.5) * ch)
            if not pts or (abs(xy[0] - pts[-1][0]) + abs(xy[1] - pts[-1][1]) > 1e-9):
                pts.append(xy)
        pts = rdp(pts, RDP_EPS)
        if len(pts) >= 2:
            feats.append(pts)
    return feats


def write_line_feature_class(fc, feats, sr):
    arcpy.DeleteFeatures_management(fc)
    cur = arcpy.da.InsertCursor(fc, ["SHAPE@"])
    count = 0
    total = 0.0
    for xy in feats:
        parr = arcpy.Array()
        for (x, y) in xy:
            parr.add(arcpy.Point(x, y))
        g = arcpy.Polyline(parr, sr)
        if g.length <= 0 or g.partCount == 0:
            continue
        cur.insertRow([g])
        count += 1
        total += g.length
    del cur
    return count, total


def export_arcmap(raster_extent):
    xmin = raster_extent.XMin
    xmax = raster_extent.XMax
    ymin = raster_extent.YMin
    ymax = raster_extent.YMax
    if os.path.exists(C.MXD_ARCSAN_OUT):
        os.remove(C.MXD_ARCSAN_OUT)
    shutil.copyfile(C.MXD_ARCSAN, C.MXD_ARCSAN_OUT)
    mxd = arcpy.mapping.MapDocument(C.MXD_ARCSAN_OUT)
    for lyr in arcpy.mapping.ListLayers(mxd):
        if lyr.name == "ParcelLines":
            lyr.replaceDataSource(C.GDB_ARCSAN, "FILEGDB_WORKSPACE", "ParcelLines")
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    for lyr in arcpy.mapping.ListLayers(mxd):
        if lyr.name.lower().startswith("parcelscan"):
            arcpy.mapping.RemoveLayer(df, lyr)
    arcpy.mapping.AddLayer(df, arcpy.mapping.Layer(C.IMG_CLEAN), "BOTTOM")
    mxd.relativePaths = False
    mxd.save()
    arcpy.mapping.ExportToPNG(mxd, C.FIG1_3, resolution=150)
    cw = raster_extent.width / 1800.0
    ch = raster_extent.height / 2300.0
    r0, r1, c0, c1 = 1250, 1550, 650, 950
    df.extent = arcpy.Extent(xmin + c0 * cw, ymax - r1 * ch, xmin + c1 * cw, ymax - r0 * ch)
    arcpy.mapping.ExportToPNG(mxd, C.FIG1_4, resolution=150)
    del mxd


def export_matplotlib(cleaned, removed, fg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h, w = cleaned.shape
    crop = None
    for (r0, r1, c0, c1, area) in removed:
        if 120 <= area <= 400 and (r1 - r0) >= 12 and (c1 - c0) >= 10:
            crop = (r0, r1, c0, c1)
            break
    if crop is not None:
        r0, r1, c0, c1 = crop
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        rad = 130
        ra = max(0, cr - rad)
        rb = min(h, cr + rad)
        ca = max(0, cc - rad)
        cb = min(w, cc + rad)
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(fg[ra:rb, ca:cb], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax1.set_title("Before cleaning")
        ax1.axis("off")
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow((~cleaned[ra:rb, ca:cb]).astype(np.uint8), cmap="gray", interpolation="nearest")
        ax2.set_title("After cleaning")
        ax2.axis("off")
        fig.savefig(C.FIG1_1, dpi=150)
        plt.close(fig)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(fg[::4, ::4], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title("Original raster")
    ax1.axis("off")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow((~cleaned[::4, ::4]).astype(np.uint8), cmap="gray", interpolation="nearest")
    ax2.set_title("Cleaned raster")
    ax2.axis("off")
    fig.savefig(C.FIG1_2, dpi=150)
    plt.close(fig)


def export_preview(chains, cleaned):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    h, w = cleaned.shape
    fig = plt.figure(figsize=(9, 11.5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(1 - cleaned.astype(np.uint8), cmap="gray", vmin=0, vmax=1, interpolation="nearest", extent=[0, w, h, 0])
    segs = [[(c, r) for (r, c) in p] for p in chains]
    ax.add_collection(LineCollection(segs, colors=[(0.85, 0.1, 0.1)], linewidths=1.6))
    ax.axis("off")
    fig.savefig(C.FIG1_5, dpi=100)
    plt.close(fig)


def main():
    arcpy.env.overwriteOutput = True
    sys.stderr.write("exp1 read raster\n")
    d = arcpy.Describe(C.IMG_SCAN)
    xmin = d.extent.XMin
    ymax = d.extent.YMax
    cw = d.meanCellWidth
    ch = d.meanCellHeight
    arr = arcpy.RasterToNumPyArray(arcpy.Raster(C.IMG_SCAN))
    fg = arr == 0
    sys.stderr.write("exp1 components\n")
    labels, ncomp = label_components(fg)
    cleaned, removed = clean_components(fg, labels, ncomp)
    sys.stderr.write("exp1 thinning\n")
    skel = thin(cleaned)
    skel = reduce_simple_points(skel)
    sys.stderr.write("exp1 tracing\n")
    paths = trace_paths(skel)
    chains = chain_paths(paths)
    feats = build_features(chains, xmin, ymax, cw, ch)
    sr = arcpy.Describe(C.GDB_ARCSAN + "/ParcelLines").spatialReference
    count, total = write_line_feature_class(C.GDB_ARCSAN + "/ParcelLines", feats, sr)
    sys.stderr.write("exp1 features=%d length=%.1f\n" % (count, total))
    if os.path.exists(C.IMG_CLEAN):
        os.remove(C.IMG_CLEAN)
    raster = arcpy.NumPyArrayToRaster(1 - cleaned.astype(np.uint8), arcpy.Point(d.extent.XMin, d.extent.YMin), cw, ch)
    raster.save(C.IMG_CLEAN)
    export_matplotlib(cleaned, removed, fg)
    export_preview(chains, cleaned)
    export_arcmap(d.extent)
    sys.stderr.write("exp1 done\n")


if __name__ == "__main__":
    main()
