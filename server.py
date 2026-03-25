"""
F1 Web Tracker — Flask Backend
Requires: pip install fastf1 flask flask-cors numpy

Run: python server.py
Then open index.html in your browser (or serve it with: python -m http.server 3000)
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import fastf1
import os
import traceback

app = Flask(__name__)
CORS(app)

# ── Cache setup ──────────────────────────────────────────────────────────────
CACHE_DIR = "f1_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)


# ── Helpers ──────────────────────────────────────────────────────────────────
def rotate(xy, angle):
    rot = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot)


def hex_to_rgb(hex_str):
    if not hex_str or hex_str == "":
        return "#C8C8C8"
    h = hex_str.lstrip("#")
    if len(h) == 6:
        return f"#{h.upper()}"
    return "#C8C8C8"


def scale_track(raw, canvas_w=900, canvas_h=600, padding=60):
    min_x, max_x = raw[:, 0].min(), raw[:, 0].max()
    min_y, max_y = raw[:, 1].min(), raw[:, 1].max()
    span_x = max_x - min_x or 1
    span_y = max_y - min_y or 1
    scale = min((canvas_w - 2 * padding) / span_x, (canvas_h - 2 * padding) / span_y)
    cx = canvas_w / 2 - ((min_x + max_x) / 2) * scale
    cy = canvas_h / 2 - ((min_y + max_y) / 2) * scale
    return scale, cx, cy


def to_canvas(points, scale, cx, cy, canvas_h):
    """World → canvas coords (Y flipped)."""
    xs = (points[:, 0] * scale + cx).tolist()
    ys = (canvas_h - (points[:, 1] * scale + cy)).tolist()
    return list(zip(xs, ys))


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/api/session", methods=["GET"])
def get_session():
    """
    Query params:
        year   - e.g. 2023
        event  - e.g. Silverstone
        mode   - gp | q  (Grand Prix or Qualifying)
        count  - number of drivers (default 20)
    """
    try:
        year  = int(request.args.get("year",  2023) or 2023)
        event = (request.args.get("event", "Silverstone") or "Silverstone").strip()
        mode  = (request.args.get("mode",  "q") or "q").lower()
        count = int(request.args.get("count", 5) or 5)

        session_type = "R" if mode == "gp" else "Q"
        session = fastf1.get_session(year, event, session_type)

        try:
            session.load(telemetry=True, laps=True)
            has_telemetry = True
        except Exception:
            session.load(telemetry=False, laps=True)
            has_telemetry = False

        # ── Track ────────────────────────────────────────────────────────────
        try:
            ref_lap  = session.laps.pick_fastest()
            pos_data = ref_lap.get_pos_data().fill_missing()
            angle    = session.get_circuit_info().rotation / 180 * np.pi
            track_raw = rotate(pos_data.loc[:, ("X", "Y")].to_numpy(), angle)
        except Exception:
            angle = 0
            track_raw = np.array([
                [np.cos(a) * 2000, np.sin(a) * 1000]
                for a in np.linspace(0, 2 * np.pi, 100)
            ])

        CANVAS_W, CANVAS_H = 900, 600
        scale, cx, cy = scale_track(track_raw, CANVAS_W, CANVAS_H)
        track_pts = to_canvas(track_raw, scale, cx, cy, CANVAS_H)

        # ── Corner labels ────────────────────────────────────────────────────
        corners = []
        try:
            circuit_info = session.get_circuit_info()
            for _, corner in circuit_info.corners.iterrows():
                off_angle = corner["Angle"] / 180 * np.pi
                off = rotate(np.array([[500, 0]]), off_angle)[0]
                tx, ty = rotate(
                    np.array([[corner["X"] + off[0], corner["Y"] + off[1]]]), angle
                )[0]
                wx, wy = rotate(
                    np.array([[corner["X"], corner["Y"]]]), angle
                )[0]
                def w2c(wx, wy):
                    return (wx * scale + cx, CANVAS_H - (wy * scale + cy))
                lx, ly = w2c(tx, ty)
                px, py = w2c(wx, wy)
                corners.append({
                    "label": f"{int(corner['Number'])}{corner['Letter']}",
                    "lx": round(lx, 1), "ly": round(ly, 1),
                    "px": round(px, 1), "py": round(py, 1),
                })
        except Exception:
            pass

        # ── Drivers ──────────────────────────────────────────────────────────
        results = session.results
        abbrevs = results["Abbreviation"].iloc[:count].tolist()

        drivers_out = []
        global_min = float("inf")

        for drv in abbrevs:
            try:
                d_laps = session.laps.pick_drivers(drv)
                if d_laps.empty:
                    continue

                color = hex_to_rgb(session.get_driver(drv)["TeamColor"])

                times_list, positions_list, dist_list = None, None, None

                if has_telemetry:
                    tele = (
                        d_laps.get_pos_data().fill_missing()
                        if mode == "gp"
                        else d_laps.pick_fastest().get_pos_data().fill_missing()
                    )
                    times = tele["Time"].dt.total_seconds().to_numpy()
                    if mode == "gp":
                        global_min = min(global_min, times[0])
                    else:
                        times = times - times[0]

                    raw_pos = rotate(tele.loc[:, ("X", "Y")].to_numpy(), angle)
                    canvas_pos = to_canvas(raw_pos, scale, cx, cy, CANVAS_H)
                    diffs = np.diff(raw_pos, axis=0)
                    seg_dists = np.sqrt((diffs ** 2).sum(axis=1))
                    dists = np.concatenate([[0.0], np.cumsum(seg_dists)])

                    # Downsample to ≤ 2000 points so the JSON stays manageable
                    step = max(1, len(times) // 2000)
                    times_list = times[::step].tolist()
                    positions_list = canvas_pos[::step]
                    dist_list = dists[::step].tolist()

                drivers_out.append({
                    "name":      drv,
                    "color":     color,
                    "times":     times_list,
                    "positions": [list(p) for p in positions_list] if positions_list else None,
                    "distances": dist_list,
                })
            except Exception as e:
                print(f"Skipping {drv}: {e}")

        # Normalise GP times
        if has_telemetry and mode == "gp" and global_min != float("inf"):
            for d in drivers_out:
                if d["times"]:
                    d["times"] = [t - global_min for t in d["times"]]

        max_time = 0
        for d in drivers_out:
            if d["times"]:
                max_time = max(max_time, d["times"][-1])
        if max_time == 0:
            max_time = 5000

        return jsonify({
            "ok":           True,
            "event":        session.event["Location"],
            "year":         year,
            "mode":         mode,
            "hasTelemetry": has_telemetry,
            "maxTime":      round(max_time, 2),
            "canvasW":      CANVAS_W,
            "canvasH":      CANVAS_H,
            "track":        [list(p) for p in track_pts],
            "corners":      corners,
            "drivers":      drivers_out,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/events", methods=["GET"])
def list_events():
    """Return a list of well-known events for the autocomplete."""
    events = [
        "Australia", "Bahrain", "Saudi Arabia", "Japan", "China",
        "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
        "Austria", "Great Britain", "Hungary", "Belgium",
        "Netherlands", "Italy", "Azerbaijan", "Singapore",
        "United States", "Mexico", "Brazil", "Las Vegas", "Qatar",
        "Abu Dhabi", "Silverstone",
    ]
    return jsonify(events)


if __name__ == "__main__":
    print("F1 Tracker API  →  http://localhost:5000")
    print("Frontend        →  open index.html (or serve on port 3000)")
    app.run(debug=True, port=5000)


# ── Focus / zoom endpoint ─────────────────────────────────────────────────────
# Stores the last loaded session in memory so /api/focus can reference it
# without re-loading from disk.
_session_cache = {}

@app.route("/api/focus", methods=["GET"])
def get_focus():
    """
    Returns a zoomed view centred on one driver at a given time.

    Query params:
        year    - e.g. 2023
        event   - e.g. Silverstone
        mode    - gp | q
        driver  - driver abbreviation e.g. VER
        time    - elapsed seconds (float)
        zoom    - canvas zoom factor (default 5.5)

    Returns:
        track       - track points clipped to the zoomed viewport
        drivers     - all driver positions at this moment (canvas coords)
        focused     - the focused driver's canvas position + heading
        viewport    - {cx, cy, w, h} of the zoomed region in base canvas coords
    """
    try:
        year   = int(request.args.get("year",  2023) or 2023)
        event  = (request.args.get("event", "Silverstone") or "Silverstone").strip()
        mode   = (request.args.get("mode",  "q") or "q").lower()
        driver = (request.args.get("driver", "") or "").strip().upper()
        t      = float(request.args.get("time", 0) or 0)
        zoom   = float(request.args.get("zoom", 5.5) or 5.5)

        cache_key = f"{year}_{event}_{mode}"

        # ── Load or retrieve cached session ──────────────────────────────────
        if cache_key not in _session_cache:
            session_type = "R" if mode == "gp" else "Q"
            session = fastf1.get_session(year, event, session_type)
            try:
                session.load(telemetry=True, laps=True)
            except Exception:
                session.load(telemetry=False, laps=True)

            # Build track
            try:
                ref_lap   = session.laps.pick_fastest()
                pos_data  = ref_lap.get_pos_data().fill_missing()
                angle     = session.get_circuit_info().rotation / 180 * np.pi
                track_raw = rotate(pos_data.loc[:, ("X", "Y")].to_numpy(), angle)
            except Exception:
                angle = 0
                track_raw = np.array([
                    [np.cos(a) * 2000, np.sin(a) * 1000]
                    for a in np.linspace(0, 2 * np.pi, 100)
                ])

            CANVAS_W, CANVAS_H = 900, 600
            scale, cx_off, cy_off = scale_track(track_raw, CANVAS_W, CANVAS_H)
            track_pts = to_canvas(track_raw, scale, cx_off, cy_off, CANVAS_H)

            # Build driver telemetry
            drivers_data = {}
            global_min = float("inf")
            results = session.results
            for drv_abbr in results["Abbreviation"].tolist():
                try:
                    d_laps = session.laps.pick_drivers(drv_abbr)
                    if d_laps.empty:
                        continue
                    color = hex_to_rgb(session.get_driver(drv_abbr)["TeamColor"])
                    tele = (
                        d_laps.get_pos_data().fill_missing() if mode == "gp"
                        else d_laps.pick_fastest().get_pos_data().fill_missing()
                    )
                    times = tele["Time"].dt.total_seconds().to_numpy()
                    if mode == "gp":
                        global_min = min(global_min, times[0])
                    else:
                        times = times - times[0]

                    raw_pos    = rotate(tele.loc[:, ("X", "Y")].to_numpy(), angle)
                    canvas_pos = to_canvas(raw_pos, scale, cx_off, cy_off, CANVAS_H)
                    step = max(1, len(times) // 2000)
                    drivers_data[drv_abbr] = {
                        "color":     color,
                        "times":     times[::step].tolist(),
                        "positions": [list(p) for p in canvas_pos[::step]],
                    }
                except Exception:
                    pass

            # Normalise GP times
            if mode == "gp" and global_min != float("inf"):
                for d in drivers_data.values():
                    d["times"] = [t2 - global_min for t2 in d["times"]]

            _session_cache[cache_key] = {
                "track":      track_pts,
                "drivers":    drivers_data,
                "canvasW":    900,
                "canvasH":    600,
            }

        cached = _session_cache[cache_key]
        CANVAS_W = cached["canvasW"]
        CANVAS_H = cached["canvasH"]

        # ── Interpolate helper ────────────────────────────────────────────────
        def interp_pos(drv_data, t_query):
            times = drv_data["times"]
            pos   = drv_data["positions"]
            if not times:
                return None
            if t_query <= times[0]:
                return pos[0]
            if t_query >= times[-1]:
                return pos[-1]
            import bisect
            idx = bisect.bisect_left(times, t_query) - 1
            idx = max(0, min(idx, len(times) - 2))
            frac = (t_query - times[idx]) / (times[idx+1] - times[idx])
            return [
                pos[idx][0] + (pos[idx+1][0] - pos[idx][0]) * frac,
                pos[idx][1] + (pos[idx+1][1] - pos[idx][1]) * frac,
            ]

        def interp_heading(drv_data, t_query):
            times = drv_data["times"]
            pos   = drv_data["positions"]
            if not times or len(times) < 2:
                return 0.0
            import bisect
            import math
            idx = bisect.bisect_left(times, t_query) - 1
            idx = max(0, min(idx, len(times) - 2))
            dx  = pos[idx+1][0] - pos[idx][0]
            dy  = pos[idx+1][1] - pos[idx][1]
            return math.atan2(dy, dx)

        # ── Focused driver position ───────────────────────────────────────────
        focused_pos     = None
        focused_heading = 0.0
        if driver and driver in cached["drivers"]:
            focused_pos     = interp_pos(cached["drivers"][driver], t)
            focused_heading = interp_heading(cached["drivers"][driver], t)

        if focused_pos is None:
            return jsonify({"ok": False, "error": f"Driver {driver} not found or no telemetry"}), 404

        fcx, fcy = focused_pos  # focused driver canvas centre

        # ── Compute zoomed viewport in base canvas space ──────────────────────
        # The viewport is the region of the 900×600 canvas that maps to
        # the full screen when zoomed in by `zoom`.
        vp_w = CANVAS_W / zoom
        vp_h = CANVAS_H / zoom
        vp_x1 = fcx - vp_w / 2
        vp_y1 = fcy - vp_h / 2
        vp_x2 = fcx + vp_w / 2
        vp_y2 = fcy + vp_h / 2

        # ── Clip track to viewport (keep points within + 20% margin) ─────────
        margin = max(vp_w, vp_h) * 0.2
        clipped_track = [
            p for p in cached["track"]
            if vp_x1 - margin <= p[0] <= vp_x2 + margin
            and vp_y1 - margin <= p[1] <= vp_y2 + margin
        ]

        # ── All driver positions at time t ────────────────────────────────────
        driver_positions = {}
        for abbr, drv_data in cached["drivers"].items():
            p = interp_pos(drv_data, t)
            if p:
                driver_positions[abbr] = {
                    "x":       round(p[0], 2),
                    "y":       round(p[1], 2),
                    "color":   drv_data["color"],
                    "heading": round(interp_heading(drv_data, t), 4),
                    "in_view": (vp_x1 <= p[0] <= vp_x2 and vp_y1 <= p[1] <= vp_y2),
                }

        return jsonify({
            "ok":      True,
            "focused": {
                "driver":  driver,
                "x":       round(fcx, 2),
                "y":       round(fcy, 2),
                "heading": round(focused_heading, 4),
                "color":   cached["drivers"][driver]["color"],
            },
            "viewport": {
                "cx":   round(fcx, 2),
                "cy":   round(fcy, 2),
                "w":    round(vp_w, 2),
                "h":    round(vp_h, 2),
                "x1":   round(vp_x1, 2),
                "y1":   round(vp_y1, 2),
                "x2":   round(vp_x2, 2),
                "y2":   round(vp_y2, 2),
                "zoom": zoom,
            },
            "track":   [[round(p[0],1), round(p[1],1)] for p in clipped_track],
            "drivers": driver_positions,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e)}), 500