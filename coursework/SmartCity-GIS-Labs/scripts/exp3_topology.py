import os
import sys
import math
import arcpy
import arcpy.mapping
import common as C

FIX_DISTANCE = 3.0
MATCH_TOL = 0.5
DUP_MEDIAN = 1.0
MAX_ROUNDS = 5
TMP_GDB = os.path.join(C.REPO, ".cache", "topo_check.gdb")


def dataset_paths():
    gdb = C.GDB_TOPO
    lot = gdb + "/StudyArea/LotLines"
    top = gdb + "/StudyArea/StudyArea_Topology"
    return gdb, lot, top


def restore_lotlines(lot):
    arcpy.CopyFeatures_management(C.LOT_RAW, lot)
    cur = arcpy.da.UpdateCursor(lot, ["SHAPE@"])
    for row in cur:
        if row[0] is None:
            cur.deleteRow()
    del cur


def create_topology(gdb, lot, top):
    arcpy.CreateTopology_management(gdb + "/StudyArea", "StudyArea_Topology")
    arcpy.AddFeatureClassToTopology_management(top, lot, 1, 1)
    arcpy.AddRuleToTopology_management(top, "Must Not Have Dangles (Line)", lot)


def export_errors(top, target_dir, tag, copy):
    arcpy.ValidateTopology_management(top)
    if arcpy.Exists(TMP_GDB):
        arcpy.Delete_management(TMP_GDB)
    arcpy.CreateFileGDB_management(os.path.join(C.REPO, ".cache"), "topo_check.gdb")
    arcpy.ExportTopologyErrors_management(top, TMP_GDB, "err")
    counts = {}
    for suffix in ["_point", "_line", "_poly"]:
        src = TMP_GDB + "/err" + suffix
        if copy:
            dst = target_dir + "/dangles_" + tag + suffix + ".shp"
            arcpy.CopyFeatures_management(src, dst)
            counts[suffix] = int(arcpy.GetCount_management(dst).getOutput(0))
        else:
            counts[suffix] = int(arcpy.GetCount_management(src).getOutput(0))
    return counts


def read_dangle_points():
    pts = []
    with arcpy.da.SearchCursor(TMP_GDB + "/err_point", ["SHAPE@"]) as cur:
        for row in cur:
            g = row[0]
            pts.append((g.centroid.X, g.centroid.Y))
    return pts


def read_lines(lot):
    feats = []
    with arcpy.da.SearchCursor(lot, ["OID@", "SHAPE@"]) as cur:
        for oid, g in cur:
            parts = []
            for i in range(g.partCount):
                arr = g.getPart(i)
                parts.append([(p.X, p.Y) for p in arr if p])
            feats.append([oid, parts])
    return feats


def build_segs(feats):
    segs = []
    for oid, parts in feats:
        for pts in parts:
            for i in range(1, len(pts)):
                segs.append((oid, pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1]))
    return segs


def nearest_on_others(segs, px, py, exclude_oid):
    best = None
    for (oid, x0, y0, x1, y1) in segs:
        if oid == exclude_oid:
            continue
        dx = x1 - x0
        dy = y1 - y0
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / l2))
        qx = x0 + t * dx
        qy = y0 + t * dy
        d = math.hypot(qx - px, qy - py)
        if best is None or d < best[0]:
            best = (d, qx, qy, oid)
    return best


def seg_intersect(p, q, r, s):
    px, py = p
    qx, qy = q
    rx, ry = r
    sx, sy = s
    d1x = qx - px
    d1y = qy - py
    d2x = sx - rx
    d2y = sy - ry
    den = d1x * d2y - d1y * d2x
    if abs(den) < 1e-12:
        return None
    t = ((rx - px) * d2y - (ry - py) * d2x) / den
    u = ((rx - px) * d1y - (ry - py) * d1x) / den
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return (px + t * d1x, py + t * d1y)
    return None


def line_crosses(pts, segs, moid):
    crosses = []
    for i in range(1, len(pts)):
        a = pts[i - 1]
        b = pts[i]
        for (oid, x0, y0, x1, y1) in segs:
            if oid != moid:
                continue
            ip = seg_intersect(a, b, (x0, y0), (x1, y1))
            if ip is not None:
                crosses.append((ip, i))
    return crosses


def apply_updates(lot, feats, update, deleted):
    sr = arcpy.Describe(lot).spatialReference
    if deleted:
        arcpy.DeleteFeatures_management(lot)
        cur = arcpy.da.InsertCursor(lot, ["SHAPE@"])
        for oid, parts in feats:
            if oid in deleted:
                continue
            if oid in update:
                parts = update[oid]
            arr = arcpy.Array()
            for pts in parts:
                inner = arcpy.Array()
                for (x, y) in pts:
                    inner.add(arcpy.Point(x, y))
                arr.add(inner)
            cur.insertRow([arcpy.Polyline(arr, sr)])
        del cur
    elif update:
        cur = arcpy.da.UpdateCursor(lot, ["OID@", "SHAPE@"])
        for oid, g in cur:
            if oid in update:
                arr = arcpy.Array()
                for pts in update[oid]:
                    inner = arcpy.Array()
                    for (x, y) in pts:
                        inner.add(arcpy.Point(x, y))
                    arr.add(inner)
                cur.updateRow([oid, arcpy.Polyline(arr, sr)])
        del cur


def fix_round(lot, top, points, round_index):
    feats = read_lines(lot)
    segs = build_segs(feats)
    by_oid = {}
    for (px, py) in points:
        best = 1e18
        found = None
        for oid, parts in feats:
            for pts in parts:
                if not pts:
                    continue
                for endidx, endpt in ((0, pts[0]), (-1, pts[-1])):
                    d = math.hypot(endpt[0] - px, endpt[1] - py)
                    if d < best:
                        best = d
                        found = (oid, endidx)
        if found is not None and best < MATCH_TOL:
            by_oid.setdefault(found[0], set()).add(found[1])
    update = {}
    deleted = set()
    n_trim = 0
    n_extend = 0
    n_dup = 0
    n_unfixed = len(points) - sum(len(v) for v in by_oid.values())
    for oid, ends in by_oid.items():
        parts = None
        for f_oid, f_parts in feats:
            if f_oid == oid:
                parts = f_parts
                break
        pts = list(parts[0])
        if round_index == 1:
            inner = pts[1:-1]
            ds = []
            if len(inner) >= 2:
                step = max(1, len(inner) // 10)
                for p in inner[::step]:
                    nb = nearest_on_others(segs, p[0], p[1], oid)
                    if nb is not None:
                        ds.append(nb[0])
            if len(ds) >= 2:
                ds.sort()
                if ds[len(ds) // 2] < DUP_MEDIAN:
                    deleted.add(oid)
                    n_dup += 1
                    continue
        changed = False
        for endidx in (0, -1):
            if endidx not in ends:
                continue
            p = pts[0] if endidx == 0 else pts[-1]
            px, py = p
            near = nearest_on_others(segs, px, py, oid)
            fixed = False
            if near is not None:
                d, qx, qy, moid = near
                crosses = line_crosses(pts, segs, moid)
                if crosses:
                    if endidx == -1:
                        order = sorted(crosses, key=lambda c: -c[1])
                    else:
                        order = sorted(crosses, key=lambda c: c[1])
                    for ip, segi in order:
                        if math.hypot(ip[0] - px, ip[1] - py) <= FIX_DISTANCE:
                            if endidx == 0:
                                pts = [ip] + pts[segi:]
                            else:
                                pts = pts[:segi] + [ip]
                            n_trim += 1
                            fixed = True
                            changed = True
                            break
                if not fixed and d <= FIX_DISTANCE:
                    if endidx == 0:
                        pts = [(qx, qy)] + pts
                    else:
                        pts = pts + [(qx, qy)]
                    n_extend += 1
                    fixed = True
                    changed = True
            if not fixed:
                n_unfixed += 1
        if changed:
            update[oid] = [pts]
    if update or deleted:
        arcpy.Delete_management(top)
        apply_updates(lot, feats, update, deleted)
        create_topology(C.GDB_TOPO, lot, top)
    return n_trim, n_extend, n_dup, n_unfixed, bool(update or deleted)


def export_maps():
    mxd = arcpy.mapping.MapDocument(C.MXD_TOPO)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    arcpy.mapping.AddLayer(df, arcpy.mapping.Layer(C.R_TOPO + "/dangles_before_point.shp"), "TOP")
    df.extent = arcpy.Describe(C.R_TOPO + "/dangles_before_point.shp").extent
    arcpy.mapping.ExportToPNG(mxd, C.FIG3_1, resolution=150)
    arcpy.mapping.ExportToPNG(mxd, C.FIG3_2, resolution=150)
    del mxd
    mxd = arcpy.mapping.MapDocument(C.MXD_TOPO)
    arcpy.mapping.ExportToPNG(mxd, C.FIG3_3, resolution=150)
    del mxd


def main():
    arcpy.env.overwriteOutput = True
    cache = os.path.join(C.REPO, ".cache")
    if not os.path.isdir(cache):
        os.makedirs(cache)
    gdb, lot, top = dataset_paths()
    sys.stderr.write("exp3 restore source\n")
    if arcpy.Exists(top):
        arcpy.Delete_management(top)
    restore_lotlines(lot)
    create_topology(gdb, lot, top)
    counts = export_errors(top, C.R_TOPO, "before", True)
    sys.stderr.write("exp3 before errors point=%d\n" % counts["_point"])
    points = read_dangle_points()
    for round_index in range(1, MAX_ROUNDS + 1):
        if not points:
            break
        n_trim, n_extend, n_dup, n_unfixed, changed = fix_round(lot, top, points, round_index)
        sys.stderr.write("exp3 round %d dangles=%d trim=%d extend=%d dup=%d unfixed=%d\n" % (
            round_index, len(points), n_trim, n_extend, n_dup, n_unfixed))
        if not changed:
            break
        export_errors(top, C.R_TOPO, "round%d" % round_index, False)
        points = read_dangle_points()
    counts = export_errors(top, C.R_TOPO, "final", True)
    sys.stderr.write("exp3 final errors point=%d\n" % counts["_point"])
    export_maps()
    sys.stderr.write("exp3 done\n")


if __name__ == "__main__":
    main()
