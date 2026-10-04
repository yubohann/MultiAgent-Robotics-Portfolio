import os
import sys
import math
import shutil
import arcpy
import arcpy.mapping
import numpy as np
import common as C

SAMPLE_STEP = 4.0
MATCH_CAP = 40.0
NEIGHBOR_RADIUS = 60.0
MEDIAN_GATE = 12.0
TPS_LAMBDA = 1.0
PAD = 80.0
BLUE = (0.271, 0.459, 0.706)
RED = (0.843, 0.188, 0.153)
GREEN = (0.2, 0.627, 0.173)


def read_parts(fc):
    feats = []
    with arcpy.da.SearchCursor(fc, ["OID@", "SHAPE@"]) as cur:
        for oid, g in cur:
            pts = []
            if g is not None:
                for i in range(g.partCount):
                    arr = g.getPart(i)
                    pts += [(p.X, p.Y) for p in arr if p]
            feats.append((oid, pts))
    return feats


def densify(pts, step):
    dense = []
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        length = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(length / step))
        for k in range(n):
            t = float(k) / n
            dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    dense.append(pts[-1])
    return dense


def build_pairs(samples, seg_a, seg_b):
    src = []
    dst = []
    for (px, py) in samples:
        ax = seg_a[:, 0]
        ay = seg_a[:, 1]
        bx = seg_b[:, 0]
        by = seg_b[:, 1]
        dx = bx - ax
        dy = by - ay
        l2 = dx * dx + dy * dy
        l2[l2 == 0] = 1e-9
        t = ((px - ax) * dx + (py - ay) * dy) / l2
        t = np.clip(t, 0, 1)
        qx = ax + t * dx
        qy = ay + t * dy
        dd = (qx - px) ** 2 + (qy - py) ** 2
        i = int(np.argmin(dd))
        if math.sqrt(float(dd[i])) <= MATCH_CAP:
            src.append((px, py))
            dst.append((float(qx[i]), float(qy[i])))
    return np.array(src), np.array(dst)


def filter_pairs(src, dst):
    dxy = dst - src
    keep = np.ones(len(src), dtype=bool)
    for _ in range(2):
        for i in range(len(src)):
            if not keep[i]:
                continue
            dx0, dy0 = src[i]
            dist2 = (src[:, 0] - dx0) ** 2 + (src[:, 1] - dy0) ** 2
            nb = keep & (dist2 <= NEIGHBOR_RADIUS ** 2)
            if nb.sum() >= 3:
                mx = np.median(dxy[nb, 0])
                my = np.median(dxy[nb, 1])
                if math.hypot(dxy[i, 0] - mx, dxy[i, 1] - my) > MEDIAN_GATE:
                    keep[i] = False
    return src[keep], dst[keep]


def fit_tps(ctrl, targ):
    n = len(ctrl)
    k = np.zeros((n, n))
    for i in range(n):
        d2 = ((ctrl - ctrl[i]) ** 2).sum(axis=1)
        k[i] = np.where(d2 > 0, d2 * np.log(d2 + 1e-12) * 0.5, 0.0)
    p = np.hstack([np.ones((n, 1)), ctrl])
    m = np.zeros((n + 3, n + 3))
    m[:n, :n] = k + TPS_LAMBDA * np.eye(n)
    m[:n, n:] = p
    m[n:, :n] = p.T
    rhs = np.zeros((n + 3, 2))
    rhs[:n, :] = targ
    sol = np.linalg.solve(m, rhs)
    return sol[:n, :], sol[n:, :]


def make_warp(ctrl, wt, affine):
    def warp(pts):
        d2 = ((pts[:, None, :] - ctrl[None, :, :]) ** 2).sum(axis=2)
        kp = np.where(d2 > 0, d2 * np.log(d2 + 1e-12) * 0.5, 0.0)
        return np.dot(np.hstack([np.ones((len(pts), 1)), pts]), affine) + np.dot(kp, wt)
    return warp


def write_import_streets(warp, feats, sr):
    fc = C.GDB_RUBBER + "/Rubbersheet/ImportStreets"
    arcpy.DeleteFeatures_management(fc)
    cur = arcpy.da.InsertCursor(fc, ["SHAPE@"])
    for oid, pts in feats:
        dense = np.array(densify(pts, SAMPLE_STEP))
        warped = warp(dense)
        arr = arcpy.Array()
        for (x, y) in warped.tolist():
            arr.add(arcpy.Point(x, y))
        g = arcpy.Polyline(arr, sr)
        g = g.generalize(0.05)
        cur.insertRow([g])
    del cur


def write_links(src, dst, sr):
    arcpy.CreateFeatureclass_management(C.R_EXP2, os.path.basename(C.LINKS), "POLYLINE", spatial_reference=sr)
    cur = arcpy.da.InsertCursor(C.LINKS, ["SHAPE@"])
    count = 0
    for i in range(len(src)):
        if math.hypot(dst[i, 0] - src[i, 0], dst[i, 1] - src[i, 1]) <= 0:
            continue
        arr = arcpy.Array([arcpy.Point(src[i, 0], src[i, 1]), arcpy.Point(dst[i, 0], dst[i, 1])])
        cur.insertRow([arcpy.Polyline(arr, sr)])
        count += 1
    del cur
    return count


def write_showcase_links(src, dst, sr):
    rows = []
    for i in range(len(src)):
        length = math.hypot(dst[i, 0] - src[i, 0], dst[i, 1] - src[i, 1])
        if length > 0:
            rows.append((length, src[i, 0], src[i, 1], dst[i, 0], dst[i, 1]))
    rows.sort(reverse=True)
    picked = []
    for (length, x0, y0, x1, y1) in rows:
        if all((x0 - a0) ** 2 + (y0 - b0) ** 2 >= 60.0 ** 2 for (_, a0, b0, _, _) in picked):
            picked.append((length, x0, y0, x1, y1))
        if len(picked) >= 15:
            break
    arcpy.CreateFeatureclass_management(C.R_EXP2, os.path.basename(C.LINKS_SHOW), "POLYLINE", spatial_reference=sr)
    cur = arcpy.da.InsertCursor(C.LINKS_SHOW, ["SHAPE@"])
    for (length, x0, y0, x1, y1) in picked:
        arr = arcpy.Array([arcpy.Point(x0, y0), arcpy.Point(x1, y1)])
        cur.insertRow([arcpy.Polyline(arr, sr)])
    del cur


def import_box():
    d = arcpy.Describe(C.IMPORT_RAW)
    return d.extent.XMin - PAD, d.extent.YMin - PAD, d.extent.XMax + PAD, d.extent.YMax + PAD


def export_arcmap(before):
    x0, y0, x1, y1 = import_box()
    mxd = arcpy.mapping.MapDocument(C.MXD_RUBBER)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    if before:
        arcpy.mapping.ExportToPNG(mxd, C.FIG2_2, resolution=150)
        df.extent = arcpy.Extent(x0, y0, x1, y1)
        arcpy.mapping.ExportToPNG(mxd, C.FIG2_1, resolution=150)
    else:
        arcpy.mapping.ExportToPNG(mxd, C.FIG2_4, resolution=150)
        df.extent = arcpy.Extent(x0, y0, x1, y1)
        arcpy.mapping.ExportToPNG(mxd, C.FIG2_3, resolution=150)
    del mxd


def save_links_mxd():
    if os.path.exists(C.MXD_RUBBER_OUT):
        os.remove(C.MXD_RUBBER_OUT)
    shutil.copyfile(C.MXD_RUBBER, C.MXD_RUBBER_OUT)
    mxd = arcpy.mapping.MapDocument(C.MXD_RUBBER_OUT)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    x0, y0, x1, y1 = import_box()
    arcpy.mapping.AddLayer(df, arcpy.mapping.Layer(C.LINKS_SHOW), "TOP")
    df.extent = arcpy.Extent(x0, y0, x1, y1)
    arcpy.mapping.ExportToPNG(mxd, C.FIG2_5, resolution=150)
    full = arcpy.Describe(C.GDB_RUBBER + "/Rubbersheet/ExistingStreets").extent
    df.extent = arcpy.Extent(full.XMin, full.YMin, full.XMax, full.YMax)
    arcpy.mapping.ExportToPNG(mxd, C.FIG2_4, resolution=150)
    mxd.save()
    del mxd


def export_matplotlib(existing, raw, fixed, links, box):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=10)
    x0, y0, x1, y1 = box

    def draw(ax, show_links):
        for pts in existing:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=0.8, zorder=1)
        for pts in raw:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=RED, lw=1.6, zorder=2)
        if show_links:
            for pts in links:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GREEN, lw=1.2, zorder=3)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.axis("off")

    fig = plt.figure(figsize=(8, 8.6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    draw(ax, True)
    handles = [plt.Line2D([0], [0], color=BLUE, lw=1.2),
               plt.Line2D([0], [0], color=RED, lw=1.8),
               plt.Line2D([0], [0], color=GREEN, lw=1.4)]
    ax.legend(handles, ["ExistingStreets", "ImportStreets" + C.LBL_BEFORE, "links"], loc="lower right", prop=font)
    fig.savefig(C.FIG2_5, dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8.6), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    draw(ax, False)
    for pts in fixed:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GREEN, lw=1.0, zorder=3)
    handles = [plt.Line2D([0], [0], color=BLUE, lw=1.2),
               plt.Line2D([0], [0], color=RED, lw=1.8),
               plt.Line2D([0], [0], color=GREEN, lw=1.2)]
    ax.legend(handles, ["ExistingStreets", "ImportStreets" + C.LBL_BEFORE, "ImportStreets" + C.LBL_AFTER],
              loc="lower right", prop=font)
    fig.savefig(C.FIG2_6, dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 9), dpi=130)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    for pts in existing:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=BLUE, lw=0.7, zorder=1)
    for pts in raw:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=RED, lw=1.6, zorder=2)
    for pts in fixed:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GREEN, lw=1.0, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [plt.Line2D([0], [0], color=BLUE, lw=1.2),
               plt.Line2D([0], [0], color=RED, lw=1.8),
               plt.Line2D([0], [0], color=GREEN, lw=1.2)]
    ax.legend(handles, ["ExistingStreets", "ImportStreets" + C.LBL_BEFORE, "ImportStreets" + C.LBL_AFTER],
              loc="lower right", prop=font)
    fig.savefig(C.FIG2_7, dpi=130)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 9.5), dpi=140)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    draw(ax, False)
    for pts in fixed:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GREEN, lw=0.9, zorder=3)
    handles = [plt.Line2D([0], [0], color=BLUE, lw=1.2),
               plt.Line2D([0], [0], color=RED, lw=1.8),
               plt.Line2D([0], [0], color=GREEN, lw=1.2)]
    ax.legend(handles, ["ExistingStreets", "ImportStreets" + C.LBL_BEFORE, "ImportStreets" + C.LBL_AFTER],
              loc="lower right", prop=font)
    fig.savefig(C.FIG2_8, dpi=140)
    plt.close(fig)


def main():
    arcpy.env.overwriteOutput = True
    sys.stderr.write("exp2 read data\n")
    raw = read_parts(C.IMPORT_RAW)
    existing = read_parts(C.GDB_RUBBER + "/Rubbersheet/ExistingStreets")
    samples = []
    for oid, pts in raw:
        samples += densify(pts, SAMPLE_STEP)
    seg_a = []
    seg_b = []
    for oid, pts in existing:
        for i in range(1, len(pts)):
            seg_a.append(pts[i - 1])
            seg_b.append(pts[i])
    seg_a = np.array(seg_a)
    seg_b = np.array(seg_b)
    src, dst = build_pairs(samples, seg_a, seg_b)
    src, dst = filter_pairs(src, dst)
    before_mean = float(np.mean(np.hypot(*(dst - src).T)))
    sys.stderr.write("exp2 links=%d mean_offset=%.2f\n" % (len(src), before_mean))
    wt, affine = fit_tps(src, dst)
    warp = make_warp(src, wt, affine)
    resid = np.hypot(*(warp(src) - dst).T)
    sys.stderr.write("exp2 tps residual mean=%.3f\n" % float(resid.mean()))
    sr = arcpy.Describe(C.GDB_RUBBER + "/Rubbersheet/ImportStreets").spatialReference
    export_arcmap(True)
    write_import_streets(warp, raw, sr)
    nlinks = write_links(src, dst, sr)
    write_showcase_links(src, dst, sr)
    sys.stderr.write("exp2 links_written=%d\n" % nlinks)
    export_arcmap(False)
    save_links_mxd()
    fixed = read_parts(C.GDB_RUBBER + "/Rubbersheet/ImportStreets")
    links = read_parts(C.LINKS_SHOW)
    export_matplotlib([p for _, p in existing], [p for _, p in raw], [p for _, p in fixed],
                      [p for _, p in links], import_box())
    sys.stderr.write("exp2 done\n")


if __name__ == "__main__":
    main()
