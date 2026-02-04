"""
draw_track_example.py

Example script that pre-draws an F1 circuit from FastF1 telemetry and optionally animates a car
running around the track. Run:
    python draw_track_example.py

Dependencies: fastf1, matplotlib, numpy
"""

import os
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import animation
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import itertools
import random

try:
    import fastf1 as ff1
except Exception:
    ff1 = None


def smooth(xs, k=11):
    """Simple moving-average smoothing (keeps dependency-free).
    k should be odd for nicer symmetry but it's not required."""
    if k <= 1:
        return np.asarray(xs)
    kernel = np.ones(k) / k
    return np.convolve(np.asarray(xs), kernel, mode='same')


def resample_along_length(x, y, s_vals):
    """Interpolate points (x,y) at cumulative-arc positions s_vals."""
    x = np.asarray(x)
    y = np.asarray(y)
    # cumulative arc length
    ds = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate(([0.0], np.cumsum(ds)))
    # ensure monotonic s
    if s_vals[-1] > s[-1]:
        s_vals = np.clip(s_vals, 0, s[-1])
    xi = np.interp(s_vals, s, x)
    yi = np.interp(s_vals, s, y)
    return xi, yi


def draw_dashed_centerline(ax, x, y, dash_len=6.0, gap=6.0, color='yellow', linewidth=1.0, zorder=3):
    """Draw evenly spaced short dashes along the centreline.

    This avoids visual gaps caused by NaNs or very short segments in raw telemetry.
    dash_len and gap are in the same units as x/y (meters).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return
    # remove NaN runs by masking and drawing per continuous segment
    isnan = np.isnan(x) | np.isnan(y)
    segments = []
    i = 0
    while i < len(x):
        # skip nans
        while i < len(x) and isnan[i]:
            i += 1
        if i >= len(x):
            break
        j = i
        while j < len(x) and not isnan[j]:
            j += 1
        xs = x[i:j]
        ys = y[i:j]
        if len(xs) < 2:
            i = j
            continue
        # compute arc-length and locations for dashes
        ds = np.hypot(np.diff(xs), np.diff(ys))
        total = ds.sum()
        if total <= 0:
            i = j
            continue
        step = dash_len + gap
        positions = np.arange(0, total, step)
        for p in positions:
            start = p
            end = min(p + dash_len, total)
            if end <= start:
                continue
            s_vals = np.array([start, end])
            xi, yi = resample_along_length(xs, ys, s_vals)
            segments.append(np.column_stack([xi, yi]))
        i = j
    if len(segments) == 0:
        return
    lc = LineCollection(segments, colors=color, linewidths=linewidth, zorder=zorder)
    ax.add_collection(lc)



def draw_track(ax, x, y, width=12, road_color='#2f2f2f', grass_color='#2b7a1f', kerb_color='#ffffff'):
    """Draws a filled road polygon, grass ring, centerline and simple kerbs.

    x, y: 1D arrays of centre-line coordinates (meters)
    width: track width in the same units as x,y
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size < 3:
        raise ValueError('Need at least 3 points for a track')

    # smooth centreline for nicer normals
    x_s = smooth(x, k=max(3, len(x)//100 if len(x) > 200 else 11))
    y_s = smooth(y, k=max(3, len(y)//100 if len(y) > 200 else 11))

    # tangents and normals
    dx = np.gradient(x_s)
    dy = np.gradient(y_s)
    lengths = np.hypot(dx, dy)
    nx = -dy / (lengths + 1e-8)
    ny = dx / (lengths + 1e-8)

    half = width / 2.0
    left_x = x_s + nx * half
    left_y = y_s + ny * half
    right_x = x_s - nx * half
    right_y = y_s - ny * half

    # road polygon
    poly_coords = np.vstack([np.column_stack([left_x, left_y]),
                             np.column_stack([right_x, right_y])[::-1]])
    road = Polygon(poly_coords, closed=True, facecolor=road_color, edgecolor='k', zorder=1)
    ax.add_patch(road)

    # grass polygon (a bit larger)
    buff = width * 2.5
    left_x2 = x_s + nx * (half + buff)
    left_y2 = y_s + ny * (half + buff)
    right_x2 = x_s - nx * (half + buff)
    right_y2 = y_s - ny * (half + buff)
    grass_coords = np.vstack([np.column_stack([left_x2, left_y2]),
                              np.column_stack([right_x2, right_y2])[::-1]])
    grass = Polygon(grass_coords, closed=True, facecolor=grass_color, edgecolor=None, zorder=0)
    ax.add_patch(grass)

    # centreline — draw robust dashed centerline (avoid gaps and jumps)
    try:
        draw_dashed_centerline(ax, x_s, y_s, dash_len=max(2.0, len(x_s)/200.0), gap=max(2.0, len(x_s)/200.0), color='yellow', linewidth=1.0, zorder=3)
    except Exception:
        # fallback to simple plot
        ax.plot(x_s, y_s, color='yellow', linewidth=0.8, linestyle='--', zorder=3)

    step = max(1, len(x_s) // 120)
    for i in range(0, len(x_s), step):
        ax.plot([left_x[i], left_x[i] - nx[i] * 1.5],
                [left_y[i], left_y[i] - ny[i] * 1.5],
                color=kerb_color, linewidth=1.2, zorder=2)
        ax.plot([right_x[i], right_x[i] + nx[i] * 1.5],
                [right_y[i], right_y[i] + ny[i] * 1.5],
                color=kerb_color, linewidth=1.2, zorder=2)

    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')


def simple_animation(ax, x, y, color='red', ms=80, interval=20):
    """Compatibility simple animation: single moving marker."""
    x = np.asarray(x)
    y = np.asarray(y)
    scat = ax.scatter([x[0]], [y[0]], s=ms, c=color, zorder=4)

    def update(i):
        scat.set_offsets([x[i % len(x)], y[i % len(y)]])
        return (scat,)

    ani = animation.FuncAnimation(ax.figure, update, frames=len(x), interval=interval, blit=True)
    return ani


def draw_speed_colored(ax, x, y, speed, cmap='viridis', linewidth=3, zorder=2, alpha=0.9):
    """Plot the centreline colored by speed using a LineCollection."""
    x = np.asarray(x)
    y = np.asarray(y)
    speed = np.asarray(speed)
    if len(x) < 2:
        return
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    # Normalize speed to colormap
    smin = float(np.nanmin(speed))
    smax = float(np.nanmax(speed))
    norm = mcolors.Normalize(vmin=smin, vmax=smax)
    lc = LineCollection(segs, cmap=cm.get_cmap(cmap), norm=norm, linewidths=linewidth, zorder=zorder, alpha=alpha)
    lc.set_array(speed[:-1])
    ax.add_collection(lc)
    # colorbar (small inset)
    try:
        cax = ax.figure.add_axes([0.92, 0.15, 0.02, 0.3])
        cb = ax.figure.colorbar(lc, cax=cax)
        cb.set_label('Speed (kph)')
    except Exception:
        pass


def animate_with_overlay(ax, tel, ms=80, interval=20, trail=60, cmap='viridis'):
    """Animate car with colored speed line, trail, speed text and throttle/brake inset (if available)."""
    x = tel['X'].to_numpy()
    y = tel['Y'].to_numpy()

    if 'Speed' in tel.columns:
        speed = tel['Speed'].to_numpy()
    else:
        # approximate speed from displacement (m) per sample -> kph (assumes uniform sampling, approximate)
        dx = np.gradient(x)
        dy = np.gradient(y)
        dist = np.hypot(dx, dy)
        # scale — not exact, but provides visual variation
        speed = dist / np.nanmax(dist + 1e-8) * 300

    draw_speed_colored(ax, x, y, speed, cmap=cmap, linewidth=3, zorder=2, alpha=0.95)

    # marker and trail
    marker = ax.scatter([x[0]], [y[0]], s=ms, c='red', zorder=5)
    trail_line, = ax.plot([], [], lw=2, color='red', alpha=0.6, zorder=4)
    speed_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.6), zorder=6)

    # throttle/brake inset
    has_throttle = 'Throttle' in tel.columns
    has_brake = 'Brake' in tel.columns
    if has_throttle or has_brake:
        ax_ins = ax.inset_axes([0.65, 0.02, 0.3, 0.12])
        ax_ins.set_xlim(0, 1)
        ax_ins.set_ylim(0, 1)
        ax_ins.axis('off')
        bar_th = ax_ins.barh(0.7, 0, height=0.25, color='tab:green', alpha=0.9)
        bar_br = ax_ins.barh(0.2, 0, height=0.25, color='tab:red', alpha=0.9)
        ax_ins.text(0, 0.95, 'Throttle', fontsize=9, color='white')
        ax_ins.text(0, 0.45, 'Brake', fontsize=9, color='white')
    else:
        ax_ins = None
        bar_th = None
        bar_br = None

    def update(i):
        idx = i % len(x)
        marker.set_offsets([x[idx], y[idx]])
        start = max(0, idx - trail)
        trail_line.set_data(x[start:idx+1], y[start:idx+1])
        sp = speed[idx]
        speed_text.set_text(f'{sp:.0f} kph')
        if ax_ins is not None:
            if has_throttle:
                try:
                    val = float(tel['Throttle'].iat[idx])
                except Exception:
                    val = 0.0
                bar_th[0].set_width(val)
            if has_brake:
                try:
                    val = float(tel['Brake'].iat[idx])
                except Exception:
                    val = 0.0
                bar_br[0].set_width(val)
        return marker, trail_line, speed_text

    ani = animation.FuncAnimation(ax.figure, update, frames=len(x), interval=interval, blit=True)
    return ani


# --- Race / Multi-driver helpers ---

def build_driver_race_tel(session, driver):
    """Concatenate telemetry for all laps for `driver` in session order.

    Returns a dict with arrays: X, Y, Speed, Lap, Time (cumulative seconds).
    The Time array is made continuous across laps (if per-lap telemetry uses relative times).
    """
    laps = session.laps[session.laps['Driver'] == driver].sort_values('LapNumber')
    xs = []
    ys = []
    spd = []
    lapnums = []
    times = []
    last_time = 0.0
    for _, lap in laps.iterrows():
        try:
            t = lap.get_telemetry()
            if t is None or 'X' not in t.columns or 'Y' not in t.columns:
                continue
            x = t['X'].to_numpy()
            y = t['Y'].to_numpy()
            if 'Time' in t.columns:
                tt = t['Time'].to_numpy()
                # make relative to start of this concatenation
                tt = (tt - tt[0]) + last_time
            else:
                # assume nominal 20Hz if Time missing
                tt = last_time + np.arange(len(x)) * 0.05
            last_time = float(tt[-1]) + 1e-6
            xs.append(x)
            ys.append(y)
            times.append(tt)
            if 'Speed' in t.columns:
                spd.append(t['Speed'].to_numpy())
            else:
                spd.append(np.full_like(x, np.nan, dtype=float))
            lapnums.append(np.full_like(x, lap['LapNumber'], dtype=int))
        except Exception:
            continue
    if len(xs) == 0:
        return None
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    Time = np.concatenate(times)
    Speed = np.concatenate(spd)
    LapNum = np.concatenate(lapnums)
    return {'X': X, 'Y': Y, 'Speed': Speed, 'Lap': LapNum, 'Time': Time} 


def resample_tel_to_time(tel, time_base, smoothing=3):
    """Resample a driver's concatenated telemetry to the shared time_base.
    Returns arrays X, Y, Speed, Lap aligned to time_base.
    """
    time = tel['Time']
    def interp_field(field):
        arr = tel[field]
        mask = ~np.isnan(arr)
        if mask.sum() < 2:
            return np.full_like(time_base, np.nan, dtype=float)
        return np.interp(time_base, time[mask], arr[mask], left=np.nan, right=np.nan)

    X = interp_field('X')
    Y = interp_field('Y')
    Speed = interp_field('Speed')
    # Lap: nearest integer via interpolation
    Lap = np.round(np.interp(time_base, time, tel['Lap'], left=np.nan, right=np.nan)).astype(np.int32)

    # apply light smoothing to reduce jitter
    if smoothing and smoothing > 1:
        k = smoothing if smoothing % 2 == 1 else smoothing + 1
        X = smooth(X, k=k)
        Y = smooth(Y, k=k)
    return {'X': X, 'Y': Y, 'Speed': Speed, 'Lap': Lap}


def animate_race(ax, drivers_tel, drivers_list=None, ms=60, interval=40, trail=120, max_drivers=99, cmap='plasma', save=None, fps=20, max_frames=2000, smoothing=3):
    """Animate multiple drivers over a common resampled time base to avoid glitches.
    drivers_tel: dict driver_code -> telemetry dict from build_driver_race_tel
    drivers_list: list to determine order and colours (defaults to keys)
    fps: frames per second for resampling
    max_frames: cap the number of frames for performance
    smoothing: smoothing window for X/Y after resampling
    """
    if drivers_list is None:
        drivers_list = list(drivers_tel.keys())

    # limit number of drivers to avoid clutter
    drivers_list = drivers_list[:max_drivers]

    # determine global time range
    times = [drivers_tel[d]['Time'] for d in drivers_list if d in drivers_tel]
    if len(times) == 0:
        raise ValueError('No telemetry available for requested drivers')
    t0 = min(t[0] for t in times)
    t1 = max(t[-1] for t in times)
    total = max(1e-3, t1 - t0)
    frames = int(min(max_frames, max(2, total * fps)))
    time_base = np.linspace(t0, t1, frames)

    # resample all drivers onto the same time base
    drivers_res = {}
    for d in drivers_list:
        tel = drivers_tel.get(d)
        if tel is None:
            continue
        try:
            drivers_res[d] = resample_tel_to_time(tel, time_base, smoothing=smoothing)
        except Exception:
            drivers_res[d] = None

    # assign colors
    cmap_obj = plt.get_cmap('tab10')
    colors = [cmap_obj(i % 10) for i in range(len(drivers_list))]

    markers = {}
    trails = {}
    texts = {}

    # pre-create artists
    for i, d in enumerate(drivers_list):
        res = drivers_res.get(d)
        if res is None:
            markers[d] = None
            trails[d] = None
            texts[d] = None
            continue
        # find first valid index
        valid = ~np.isnan(res['X']) & ~np.isnan(res['Y'])
        if not np.any(valid):
            markers[d] = None
            trails[d] = None
            texts[d] = None
            continue
        idx0 = np.argmax(valid)
        markers[d] = ax.scatter([res['X'][idx0]], [res['Y'][idx0]], s=ms, c=[colors[i]], zorder=6)
        trails[d], = ax.plot([], [], lw=2, color=colors[i], alpha=0.8, zorder=5)
        texts[d] = ax.text(res['X'][idx0], res['Y'][idx0], f"{d} L{res['Lap'][idx0]}", fontsize=9, color='white', zorder=7)

    lap_display = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5), zorder=8)

    def update(frame):
        t = time_base[frame]
        for d in drivers_list:
            res = drivers_res.get(d)
            if res is None:
                continue
            x = res['X']
            y = res['Y']
            lap = res['Lap']
            if np.isnan(x[frame]) or np.isnan(y[frame]):
                if markers[d] is not None:
                    markers[d].set_offsets([np.nan, np.nan])
                    trails[d].set_data([], [])
                    texts[d].set_text(f"{d} (off)")
                continue
            if markers[d] is None:
                continue
            markers[d].set_offsets([x[frame], y[frame]])
            start = max(0, frame - trail)
            trails[d].set_data(x[start:frame+1], y[start:frame+1])
            texts[d].set_position((x[frame] + 2.0, y[frame] + 2.0))
            texts[d].set_text(f"{d} L{lap[frame]}")
        # display current time nicely
        mins = int(t // 60)
        secs = int(t % 60)
        lap_display.set_text(f'Time: {mins:02d}:{secs:02d}  Frame: {frame}/{frames}')
        return list(markers.values()) + list(trails.values()) + list(texts.values()) + [lap_display]

    ani = animation.FuncAnimation(ax.figure, update, frames=frames, interval=1000.0 / fps, blit=False)

    if save:
        try:
            ani.save(save, dpi=200)
        except Exception as e:
            warnings.warn(f'Failed to save animation: {e}')
    return ani


def main():
    parser = argparse.ArgumentParser(description='Draw and optionally animate a track from FastF1 telemetry')
    parser.add_argument('--year', type=int, default=2021)
    parser.add_argument('--gp', type=str, default='British Grand Prix')
    parser.add_argument('--session', type=str, default='Q', help='Session type: Q, R, S (Practice as 1/2/3 or P1/P2/P3)')
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--race', action='store_true', help='Animate the entire race with multiple drivers')
    parser.add_argument('--drivers', type=str, default='all', help='Comma-separated driver codes to include or "all"')
    parser.add_argument('--limit', type=int, default=8, help='Max drivers to show (for readability)')
    parser.add_argument('--save', type=str, default=None, help='Optional filename to save animation (mp4/gif)')
    parser.add_argument('--fps', type=int, default=20, help='Frames per second for race resampling')
    parser.add_argument('--max-frames', type=int, default=2000, help='Cap on number of frames to avoid overload')
    parser.add_argument('--smooth', type=int, default=3, help='Smoothing window (samples) to reduce jitter')
    parser.add_argument('--cache', type=str, default='./ff1_cache')
    args = parser.parse_args()

    if ff1 is None:
        warnings.warn('fastf1 not available: running demo with synthetic track')
        # make a synthetic circuit (simple oval)
        theta = np.linspace(0, 2 * np.pi, 800)
        r = 200 + 30 * np.sin(3 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_track(ax, x, y, width=12)
        if args.animate:
            ani = simple_animation(ax, x, y)
        plt.show()
        return

    ff1.Cache.enable_cache(args.cache)
    # tolerate user-friendly session strings
    sess_str = args.session
    try:
        session = ff1.get_session(args.year, args.gp, sess_str)
        session.load()  # loads laps, weather, telemetry when needed
    except Exception as e:
        warnings.warn(f'Could not load session from FastF1 ({e}), running synthetic demo instead')
        theta = np.linspace(0, 2 * np.pi, 800)
        r = 200 + 30 * np.sin(3 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_track(ax, x, y, width=12)
        if args.animate:
            ani = simple_animation(ax, x, y)
        plt.show()
        return

    # pick a lap with coordinates (fastest by default)
    try:
        lap = session.laps.pick_fastest()
    except Exception:
        lap = session.laps.iloc[0]

    try:
        tel = lap.get_telemetry().reset_index()
    except Exception as e:
        warnings.warn(f'No telemetry available on the chosen lap ({e}), using lap position if present')
        # if there is no telemetry, try to use lap['X/Y'] columns (not common)
        tel = None

    if tel is None or 'X' not in tel.columns or 'Y' not in tel.columns:
        warnings.warn('Telemetry lacks X/Y. Running synthetic demo instead')
        theta = np.linspace(0, 2 * np.pi, 800)
        r = 200 + 30 * np.sin(3 * theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_track(ax, x, y, width=12)
        if args.animate:
            ani = simple_animation(ax, x, y)
        plt.show()
        return

    x = tel['X'].to_numpy()
    y = tel['Y'].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 7))
    # set black background for dramatic race view
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    draw_track(ax, x, y, width=12, road_color='#111111', grass_color='#082008')

    if args.race:
        # build telemetry for requested drivers
        if args.drivers.lower() == 'all':
            drivers = list(session.laps['Driver'].unique())
        else:
            drivers = [d.strip() for d in args.drivers.split(',') if d.strip()]
        # limit for readability (set --limit 0 to show all)
        if args.limit and args.limit > 0:
            drivers = drivers[:args.limit]

        drivers_tel = {}
        warnings.warn(f'Attempting to build telemetry for drivers: {drivers}')
        for d in drivers:
            laps_count = int((session.laps[session.laps['Driver'] == d]).shape[0])
            warnings.warn(f'Driver {d}: laps found = {laps_count}')
            dt = build_driver_race_tel(session, d)
            if dt is not None:
                warnings.warn(f'  -> built telemetry for {d}, samples={len(dt["X"])}')
                drivers_tel[d] = dt
            else:
                warnings.warn(f'  -> no usable telemetry for {d}')
        drivers_with_tel = list(drivers_tel.keys())
        if len(drivers_with_tel) == 0:
            warnings.warn('No drivers had usable telemetry; aborting race animation')
        else:
            ani = animate_race(ax, drivers_tel, drivers_list=drivers_with_tel, ms=60, interval=25, trail=200, max_drivers=args.limit if args.limit and args.limit>0 else 999, save=args.save, fps=args.fps, max_frames=args.max_frames, smoothing=args.smooth)
    else:
        if args.animate:
            ani = animate_with_overlay(ax, tel)

    plt.show()


if __name__ == '__main__':
    main()
