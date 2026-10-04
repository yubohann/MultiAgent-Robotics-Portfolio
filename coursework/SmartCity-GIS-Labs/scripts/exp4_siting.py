import os
import sys
import math
import shutil
import arcpy
import arcpy.mapping
import numpy as np
import common as C

GRADE_FIELD = u"\u7b49\u7ea7"
GREY_BG = (0.72, 0.72, 0.72)
GREY_MID = (0.55, 0.55, 0.55)
BLUE = (0.271, 0.459, 0.706)
RED = (0.843, 0.188, 0.153)
GREEN = (0.2, 0.627, 0.173)
YELLOW = (1.0, 0.878, 0.44)


def add_value_field(fc, name, value):
    arcpy.AddField_management(fc, name, "SHORT")
    arcpy.CalculateField_management(fc, name, str(value), "PYTHON_9.3")


def build_buffers():
    jobs = [("mainstreet", 50, C.BUF_MAIN), ("residential", 100, C.BUF_RESI),
            ("stops", 100, C.BUF_STOPS), ("othermarkets", 500, C.BUF_MARKET)]
    for name, dist, out in jobs:
        arcpy.Buffer_analysis(C.GDB_CITY + "/" + name, out, "%d Meters" % dist, "FULL", "ROUND", "ALL")
        sys.stderr.write("exp4 buffer %s %dm\n" % (name, dist))


def build_selection():
    arcpy.Intersect_analysis([C.BUF_STOPS, C.BUF_MAIN, C.BUF_RESI], C.THREE, "ALL", "", "INPUT")
    arcpy.Erase_analysis(C.THREE, C.BUF_MARKET, C.PERFECT)
    add_value_field(C.BUF_STOPS, "stops", 1)
    add_value_field(C.BUF_MAIN, "mainstreet", 1)
    add_value_field(C.BUF_RESI, "residentia", 1)
    add_value_field(C.BUF_MARKET, "markets", -1)
    arcpy.Union_analysis([C.BUF_STOPS, C.BUF_MAIN, C.BUF_RESI, C.BUF_MARKET], C.UNION, "ALL", "", "NO_GAPS")
    code = "def calc(m, r, s, t):\n    return (m or 0) + (r or 0) + (s or 0) + (t or 0)"
    arcpy.CalculateField_management(C.UNION, "class", "calc(!markets!, !residentia!, !mainstreet!, !stops!)", "PYTHON_9.3", code)
    arcpy.CalculateField_management(C.UNION, GRADE_FIELD, "4 - !class!", "PYTHON_9.3")


def build_task():
    arcpy.MakeFeatureLayer_management(C.D_TASK + "/data/famous place.shp", "parks_lyr", "NAME LIKE '%GONGYUAN%'")
    arcpy.CopyFeatures_management("parks_lyr", C.PARK)
    arcpy.Delete_management("parks_lyr")
    arcpy.Buffer_analysis(C.PARK, C.BUF_PARK, "200 Meters", "FULL", "ROUND", "ALL")
    arcpy.Buffer_analysis(C.D_TASK + "/data/school.shp", C.BUF_SCHOOL, "500 Meters", "FULL", "ROUND", "ALL")
    arcpy.Buffer_analysis(C.D_TASK + "/data/network.shp", C.BUF_ROAD, "100 Meters", "FULL", "ROUND", "ALL")
    arcpy.Intersect_analysis([C.BUF_PARK, C.BUF_SCHOOL], C.CAND, "ALL")
    arcpy.Erase_analysis(C.CAND, C.BUF_ROAD, C.TASK_RESULT)


def read_parts(fc, shape_type):
    out = []
    with arcpy.da.SearchCursor(fc, ["SHAPE@"]) as cur:
        for row in cur:
            g = row[0]
            if shape_type == "Point":
                out.append((g.centroid.X, g.centroid.Y))
                continue
            for i in range(g.partCount):
                arr = g.getPart(i)
                pts = [(p.X, p.Y) for p in arr if p]
                if len(pts) >= 2:
                    out.append(pts)
    return out


def read_graded(fc):
    polys = {}
    with arcpy.da.SearchCursor(fc, ["SHAPE@", GRADE_FIELD]) as cur:
        for row in cur:
            g = row[0]
            grade = row[1]
            for i in range(g.partCount):
                arr = g.getPart(i)
                pts = [(p.X, p.Y) for p in arr if p]
                if len(pts) >= 3:
                    polys.setdefault(grade, []).append(pts)
    return polys


def export_arcmap():
    mxd = arcpy.mapping.MapDocument(C.MXD_CITY)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    for shp in [C.BUF_MAIN, C.BUF_RESI, C.BUF_STOPS, C.BUF_MARKET]:
        arcpy.mapping.AddLayer(df, arcpy.mapping.Layer(shp), "TOP")
    arcpy.mapping.ExportToPNG(mxd, C.FIG4_1, resolution=150)
    del mxd
    mxd = arcpy.mapping.MapDocument(C.MXD_CITY)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    for shp in [C.BUF_MARKET, C.THREE, C.PERFECT]:
        arcpy.mapping.AddLayer(df, arcpy.mapping.Layer(shp), "TOP")
    arcpy.mapping.ExportToPNG(mxd, C.FIG4_2, resolution=150)
    del mxd
    if os.path.exists(C.MXD_SITING_OUT):
        os.remove(C.MXD_SITING_OUT)
    shutil.copyfile(C.MXD_CITY, C.MXD_SITING_OUT)
    mxd = arcpy.mapping.MapDocument(C.MXD_SITING_OUT)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    for shp in [C.BUF_MAIN, C.BUF_RESI, C.BUF_STOPS, C.BUF_MARKET, C.THREE, C.PERFECT]:
        arcpy.mapping.AddLayer(df, arcpy.mapping.Layer(shp), "TOP")
    mxd.save()
    del mxd
    mxd = arcpy.mapping.MapDocument(C.MXD_TASK)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    if os.path.exists(C.MXD_TASK_OUT):
        os.remove(C.MXD_TASK_OUT)
    mxd.saveACopy(C.MXD_TASK_OUT)
    del mxd
    mxd = arcpy.mapping.MapDocument(C.MXD_TASK_OUT)
    df = arcpy.mapping.ListDataFrames(mxd)[0]
    for shp in [C.BUF_PARK, C.BUF_SCHOOL, C.BUF_ROAD, C.TASK_RESULT]:
        arcpy.mapping.AddLayer(df, arcpy.mapping.Layer(shp), "TOP")
    mxd.save()
    del mxd


def export_graded(graded, streets, perfect):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    font = FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=10)
    shades = {1: (0.07, 0.07, 0.07), 2: (0.24, 0.24, 0.24), 3: (0.43, 0.43, 0.43),
              4: (0.65, 0.65, 0.65), 5: (0.85, 0.85, 0.85)}
    fig = plt.figure(figsize=(9, 8), dpi=150)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    for pts in streets:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=GREY_BG, lw=0.4, zorder=1)
    for grade in [5, 4, 3, 2, 1]:
        patches = [MplPoly(np.array(pts), closed=True) for pts in graded.get(grade, [])]
        if patches:
            coll = PatchCollection(patches, facecolor=shades[grade], edgecolor=GREY_MID, linewidths=0.15, zorder=2)
            ax.add_collection(coll)
    for pts in perfect:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=RED, lw=1.4, zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=shades[g]) for g in [1, 2, 3, 4, 5]]
    handles.append(plt.Line2D([0], [0], color=RED, lw=1.4))
    labels = [C.LBL_G1, C.LBL_G2, C.LBL_G3, C.LBL_G4, C.LBL_G5, C.LBL_PERFECT]
    ax.legend(handles, labels, loc="lower right", prop=font)
    fig.savefig(C.FIG4_3, dpi=150)
    plt.close(fig)


def export_task(network, schools, parks, b_park, b_school, b_road, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection
    font = FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=10)
    fig = plt.figure(figsize=(8.5, 8.5), dpi=150)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    patches = [MplPoly(np.array(pts), closed=True) for pts in b_road]
    ax.add_collection(PatchCollection(patches, facecolor=(0.87, 0.87, 0.87), edgecolor="none", zorder=1))
    for pts in network:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=(0.67, 0.67, 0.67), lw=0.3, zorder=2)
    patches = [MplPoly(np.array(pts), closed=True) for pts in b_school]
    ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=BLUE, linewidths=0.7, zorder=3))
    patches = [MplPoly(np.array(pts), closed=True) for pts in b_park]
    ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=GREEN, linewidths=0.9, zorder=4))
    patches = [MplPoly(np.array(pts), closed=True) for pts in result]
    ax.add_collection(PatchCollection(patches, facecolor=YELLOW, edgecolor=RED, linewidths=0.8, zorder=5))
    ax.plot([p[0] for p in schools], [p[1] for p in schools], "o", color=BLUE, ms=3, zorder=6)
    ax.plot([p[0] for p in parks], [p[1] for p in parks], "s", color=GREEN, ms=4, zorder=6)
    ax.set_aspect("equal")
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=(0.87, 0.87, 0.87)),
               plt.Rectangle((0, 0), 1, 1, fc="none", ec=BLUE),
               plt.Rectangle((0, 0), 1, 1, fc="none", ec=GREEN),
               plt.Rectangle((0, 0), 1, 1, fc=YELLOW, ec=RED),
               plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, ms=5),
               plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=GREEN, ms=5)]
    labels = [C.LBL_ROAD_BUF, C.LBL_SCHOOL_BUF, C.LBL_PARK_BUF, C.LBL_TASK_AREA, C.LBL_SCHOOL, C.LBL_PARK]
    ax.legend(handles, labels, loc="lower right", prop=font)
    fig.savefig(C.FIG5_1, dpi=150)
    plt.close(fig)


def main():
    arcpy.env.overwriteOutput = True
    sys.stderr.write("exp4 buffers\n")
    build_buffers()
    sys.stderr.write("exp4 selection\n")
    build_selection()
    sys.stderr.write("exp4 task\n")
    build_task()
    graded = read_graded(C.UNION)
    streets = read_parts(C.GDB_CITY + "/mainstreet", "Polyline")
    perfect = read_parts(C.PERFECT, "Polygon")
    export_graded(graded, streets, perfect)
    network = read_parts(C.D_TASK + "/data/network.shp", "Polyline")
    schools = read_parts(C.D_TASK + "/data/school.shp", "Point")
    parks = read_parts(C.PARK, "Point")
    b_park = read_parts(C.BUF_PARK, "Polygon")
    b_school = read_parts(C.BUF_SCHOOL, "Polygon")
    b_road = read_parts(C.BUF_ROAD, "Polygon")
    task_result = read_parts(C.TASK_RESULT, "Polygon")
    export_task(network, schools, parks, b_park, b_school, b_road, task_result)
    export_arcmap()
    sys.stderr.write("exp4 done\n")


if __name__ == "__main__":
    main()
