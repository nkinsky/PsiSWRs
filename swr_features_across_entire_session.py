# -*- coding: utf-8 -*-
"""
ALL-ripples SWR/ripple analysis.

This code includes ALL detected ripples, regardless of sleep state.

Default post-injection analysis window:
- All animals and all sessions (Saline 1, Psilocybin, Saline 2):
      first 1 hour after injection.
- The previous animal-specific rule can be enabled in USER SETTINGS.
- Events are selected by peak time relative to injection.
- Recordings shorter than 1 hour contribute only their available events.

No WAKE-only filtering is applied.
No SleepState file is required for this ALL_RIPPLES analysis.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

from neuropy.core import Epoch, ProcessData, Shank, Probe, ProbeGroup
from neuropy.analyses import oscillations
from neuropy.analyses.oscillations import Ripple

# =============================================================================
# --- FIGURE SETTINGS ---
# =============================================================================
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["text.usetex"] = False

sns.set_style("white")

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.labelsize"] = 22
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["legend.fontsize"] = 26
mpl.rcParams["axes.linewidth"] = 1.6

SAVE_TRANSPARENT = False
FIG_BG = "white" if not SAVE_TRANSPARENT else "none"
AX_BG = "white" if not SAVE_TRANSPARENT else "none"

SPINE_COLOR = "#5a5a5a"
BRACKET_COLOR = "#4a4a4a"
TEXT_COLOR = "#2b2b2b"

# =============================================================================
# --- USER SETTINGS ---
# =============================================================================
BASE_DIR_ROOT = r"/Users/Yoga/Desktop/data_psilocybin"

# Choose ONE analysis mode by leaving only one assignment uncommented.
# Option 1 (default): first hour after injection for all animals and sessions.
WINDOW_MODE = "first_hour"

# Option 2: previous animal-specific analysis windows.
# Finn2 Saline 1/2: first hour; all other animal/session pairs: first 5 hours.
# To enable this option, comment out Option 1 and uncomment the next line.
# WINDOW_MODE = "original_windows"

if WINDOW_MODE not in ("first_hour", "original_windows"):
    raise ValueError(f"Unknown WINDOW_MODE: {WINDOW_MODE}")

POST_WINDOW_SEC = 60 * 60
FULL_POST_WINDOW_SEC = 5 * 60 * 60
POST_WINDOW_TAG = (
    "ALL_RIPPLES_ALL_ANIMALS_FIRST1H"
    if WINDOW_MODE == "first_hour"
    else "ALL_RIPPLES_FINN2_SALINE1H_OTHERS_FULL5H"
)

EXPORT_DIR = os.path.join(BASE_DIR_ROOT, f"ripple_exports_{POST_WINDOW_TAG}")
os.makedirs(EXPORT_DIR, exist_ok=True)

FIXED_THRESHOLD = 4.0

animals = ["finn", "rey", "rose", "finn2"]
order_sessions = ["Saline 1", "Psilocybin", "Saline 2"]

nchan_buf = 4

chan_dict = {
    "rey":   {"Saline 1": 21, "Psilocybin": 21, "Saline 2": 22},
    "finn":  {"Saline 1": 27, "Psilocybin": 27, "Saline 2": 27},
    "rose":  {"Saline 1": 27, "Psilocybin": 26, "Saline 2": 25},
    "finn2": {"Saline 1": 4,  "Psilocybin": 4,  "Saline 2": 4}
}

palette = {
    "Saline 1": "#4A86D9",
    "Psilocybin": "#F28A3C",
    "Saline 2": "#59B85D"
}

pairs_sessions = [
    ("Saline 1", "Psilocybin"),
    ("Psilocybin", "Saline 2"),
    ("Saline 1", "Saline 2")
]

features = {
    "duration": "Duration (s)",
    "peak_power": "Peak Power",
    "peak_frequency_bp": "Peak Frequency (Hz)",
    "sharp_wave_amplitude": "SW Amplitude (mV)"
}

zoom_specs = {
    "peak_power": (0, 125000),
    "sharp_wave_amplitude": (0, 20)
}

BRACKET_SETTINGS = {
    "duration": {
        "top_margin_frac": 0.10,
        "step_frac": 0.085,
        "h_frac": 0.015,
        "text_frac": 0.008,
        "fontsize": 15
    },
    "peak_power": {
        "top_margin_frac": 0.11,
        "step_frac": 0.085,
        "h_frac": 0.015,
        "text_frac": 0.008,
        "fontsize": 14
    },
    "peak_frequency_bp": {
        "top_margin_frac": 0.10,
        "step_frac": 0.085,
        "h_frac": 0.015,
        "text_frac": 0.008,
        "fontsize": 15
    },
    "sharp_wave_amplitude": {
        "top_margin_frac": 0.10,
        "step_frac": 0.090,
        "h_frac": 0.015,
        "text_frac": 0.008,
        "fontsize": 15
    },
}


# =============================================================================
# --- WINDOW HELPERS ---
# =============================================================================
def get_post_window_sec(animal_key, session_label):
    """Return the post-injection duration selected by WINDOW_MODE."""
    if WINDOW_MODE == "first_hour":
        return POST_WINDOW_SEC
    animal_key = str(animal_key).strip().lower()
    if animal_key == "finn2" and session_label in ("Saline 1", "Saline 2"):
        return POST_WINDOW_SEC
    return FULL_POST_WINDOW_SEC


def get_window_rule(animal_key, session_label):
    """Return the active window rule for exported metadata."""
    if WINDOW_MODE == "first_hour":
        return "all_animals_all_sessions_first1h"
    animal_key = str(animal_key).strip().lower()
    if animal_key == "finn2" and session_label in ("Saline 1", "Saline 2"):
        return "finn2_saline_1h"
    if animal_key == "finn2" and session_label == "Psilocybin":
        return "finn2_psilocybin_full5h"
    return "other_animals_full5h"


# =============================================================================
# --- BASIC HELPERS ---
# =============================================================================
def normalize_animal_name(name):
    return name.strip().lower()


def session_from_folder(folder_name):
    f = folder_name.lower().replace("-", "").replace("_", "")

    if "saline1" in f:
        return "Saline 1"

    if "psilocybin" in f:
        return "Psilocybin"

    if "saline2" in f:
        return "Saline 2"

    return None


def recording_token_from_session(session_label):
    if session_label == "Saline 1":
        return "saline1"

    if session_label == "Psilocybin":
        return "psilocybin"

    if session_label == "Saline 2":
        return "saline2"

    return None


def find_animal_dirs(base_dir):
    """
    Recursively finds animal folders.

    Supported examples:
        data_psilocybin / Rose_sleep / Rose / 2022_08_09_saline1
        data_psilocybin / Rey_sleep  / Rey  / 2022_06_03_saline2
        data_psilocybin / Finn2 / session_folder
    """

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"BASE_DIR_ROOT not found: {base_dir}")

    animal_dirs = []

    for dirpath, dirnames, filenames in os.walk(base_dir):
        folder_name = os.path.basename(dirpath)
        animal_key = normalize_animal_name(folder_name)

        if animal_key not in animals:
            continue

        has_session_child = False

        try:
            for child in os.listdir(dirpath):
                child_path = os.path.join(dirpath, child)

                if os.path.isdir(child_path) and session_from_folder(child) is not None:
                    has_session_child = True
                    break

        except Exception:
            has_session_child = False

        if has_session_child:
            animal_dirs.append({
                "animal_key": animal_key,
                "animal_name": folder_name,
                "path": dirpath
            })

    unique_items = {}

    for item in animal_dirs:
        key = (item["animal_key"], os.path.abspath(item["path"]))
        unique_items[key] = item

    return list(unique_items.values())


def find_injection_file(session_dir, animal_name, rec_token):
    patterns = [
        os.path.join(session_dir, f"{animal_name}_{rec_token}_denoised.injection.npy"),
        os.path.join(session_dir, f"*{rec_token}*injection*.npy"),
        os.path.join(session_dir, "*injection*.npy"),
    ]

    for pat in patterns:
        files = glob.glob(pat)

        if files:
            return files[0]

    return None


def find_artifact_file(session_dir, animal_name, rec_token):
    patterns = [
        os.path.join(session_dir, f"{animal_name}_{rec_token}_denoised.artifact.csv"),
        os.path.join(session_dir, f"{animal_name}_{rec_token}_denoised.artifact"),
        os.path.join(session_dir, f"*{rec_token}*artifact*.csv"),
        os.path.join(session_dir, f"*{rec_token}*artifact*"),
        os.path.join(session_dir, "*artifact*.csv"),
        os.path.join(session_dir, "*artifact*"),
    ]

    for pat in patterns:
        files = glob.glob(pat)

        if files:
            return files[0]

    return None


def load_injection_time_from_file(injection_file):
    try:
        obj = np.load(injection_file, allow_pickle=True).item()

        idx = next(
            (k for k, v in obj["epochs"]["label"].items() if v == "POST"),
            None
        )

        if idx is None:
            return None

        return float(obj["epochs"]["start"][idx])

    except Exception as e:
        print(f"[Injection load error] {injection_file}: {e}")
        return None


def load_artifact_epochs_from_file(artifact_file):
    if artifact_file is None or not os.path.exists(artifact_file):
        return None

    try:
        df = pd.read_csv(artifact_file)

        start_col = next((c for c in df.columns if "start" in c.lower()), None)

        stop_col = next(
            (c for c in df.columns if "stop" in c.lower() or "end" in c.lower()),
            None
        )

        if start_col is None or stop_col is None:
            return None

        out = pd.DataFrame({
            "start": pd.to_numeric(df[start_col], errors="coerce"),
            "stop": pd.to_numeric(df[stop_col], errors="coerce"),
            "label": "artifact"
        }).dropna()

        if out.empty:
            return None

        return Epoch(out.reset_index(drop=True))

    except Exception as e:
        print(f"[Artifact load error] {artifact_file}: {e}")
        return None


# =============================================================================
# --- CHANNEL SELECTION ---
# =============================================================================
def get_good_local_channels(sess, ripple_channel_id, nbuf=4):
    try:
        good = sess.prbgrp.get_connected_channels()
        good = np.asarray(good).astype(int).flatten()

    except Exception:
        good = np.arange(sess.eegfile.n_channels).astype(int)

    lo = int(ripple_channel_id) - int(nbuf)
    hi = int(ripple_channel_id) + int(nbuf)

    local = good[(good >= lo) & (good <= hi)]
    local = np.unique(local)

    if local.size == 0:
        if int(ripple_channel_id) in good:
            local = np.array([int(ripple_channel_id)], dtype=int)
        else:
            local = np.array(
                [int(good[np.argmin(np.abs(good - int(ripple_channel_id)))])],
                dtype=int
            )

    return np.sort(local)


# =============================================================================
# --- SHARP-WAVE AMPLITUDE ---
# =============================================================================
def compute_sharp_wave_amplitude(eegfile, valid_epochs, local_chans):
    if not hasattr(Ripple, "get_sharp_wave_amplitude"):
        raise RuntimeError(
            "Ripple.get_sharp_wave_amplitude was not found. "
            "Please update NeuroPy from GitHub."
        )

    try:
        sw = Ripple.get_sharp_wave_amplitude(
            eegfile,
            valid_epochs,
            local_chans
        )

    except TypeError:
        sw = Ripple.get_sharp_wave_amplitude(
            eegfile,
            valid_epochs,
            channels=local_chans
        )

    return np.asarray(sw).astype(float).flatten()


# =============================================================================
# --- PLOT HELPERS ---
# =============================================================================
def p_to_star(p):
    if p <= 1e-4:
        return "****"
    elif p <= 1e-3:
        return "***"
    elif p <= 1e-2:
        return "**"
    elif p <= 5e-2:
        return "*"
    else:
        return "ns"


def add_manual_brackets(
    ax,
    feat,
    animal_order,
    animal_map,
    pval_lookup,
    ylim=None,
    hue_order=order_sessions,
    pairs=pairs_sessions,
    width=0.78
):
    cfg = BRACKET_SETTINGS.get(feat, {})

    top_margin_frac = cfg.get("top_margin_frac", 0.10)
    step_frac = cfg.get("step_frac", 0.085)
    h_frac = cfg.get("h_frac", 0.015)
    text_frac = cfg.get("text_frac", 0.008)
    fontsize = cfg.get("fontsize", 14)

    if ylim is None:
        y0, y1 = ax.get_ylim()
    else:
        y0, y1 = ylim

    yr = (y1 - y0) if y1 > y0 else 1.0
    h = h_frac * yr
    text_pad = text_frac * yr
    base = y1 - top_margin_frac * yr
    step = step_frac * yr

    n_hue = len(hue_order)
    hue_w = width / n_hue
    sess_to_idx = {s: i for i, s in enumerate(hue_order)}

    for ai, a in enumerate(animal_order):
        a_lbl = animal_map[a]
        x_center = ai

        for k, (s1, s2) in enumerate(pairs):
            key = (feat, a_lbl, s1, s2)

            if key not in pval_lookup:
                continue

            p = float(pval_lookup[key])
            star = p_to_star(p)

            i1 = sess_to_idx[s1]
            i2 = sess_to_idx[s2]

            off1 = (-width / 2) + (hue_w / 2) + i1 * hue_w
            off2 = (-width / 2) + (hue_w / 2) + i2 * hue_w

            x1 = x_center + off1
            x2 = x_center + off2
            y = base - k * step

            ax.plot(
                [x1, x1, x2, x2],
                [y, y + h, y + h, y],
                lw=1.7,
                c=BRACKET_COLOR,
                clip_on=True,
                solid_capstyle="butt"
            )

            ax.text(
                (x1 + x2) / 2,
                y + h + text_pad,
                star,
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=BRACKET_COLOR,
                clip_on=True
            )


def recolor_boxplot_outline(ax, hue_order, palette):
    if len(ax.patches) == 0:
        return

    box_centers = []

    for p in ax.patches:
        bb_disp = p.get_extents()
        bb_data = bb_disp.transformed(ax.transData.inverted())
        cx = 0.5 * (bb_data.x0 + bb_data.x1)
        box_centers.append(cx)

    box_centers = np.asarray(box_centers)
    cat_ids = np.rint(box_centers).astype(int)

    patch_colors = [None] * len(ax.patches)

    for cat in np.unique(cat_ids):
        idx = np.where(cat_ids == cat)[0]
        idx_sorted = idx[np.argsort(box_centers[idx])]

        for j, pi in enumerate(idx_sorted):
            hue = hue_order[j % len(hue_order)]
            patch_colors[pi] = palette[hue]

    for i, p in enumerate(ax.patches):
        c = patch_colors[i] if patch_colors[i] is not None else "#000000"
        p.set_facecolor((0, 0, 0, 0))
        p.set_edgecolor(c)
        p.set_linewidth(1.5)

    centers = box_centers.copy()

    colors = np.array(
        [
            patch_colors[i] if patch_colors[i] is not None else "#000000"
            for i in range(len(ax.patches))
        ],
        dtype=object
    )

    for ln in ax.lines:
        xd = ln.get_xdata()

        if xd is None or len(xd) == 0:
            continue

        xmean = float(np.nanmean(xd))
        j = int(np.argmin(np.abs(centers - xmean)))

        ln.set_color(colors[j])
        ln.set_linewidth(1.3)


def style_axis(ax):
    ax.set_facecolor(AX_BG)
    ax.grid(False)

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    for side in ["left", "bottom"]:
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(1.6)

    ax.tick_params(axis="both", colors=TEXT_COLOR, width=1.2, length=4)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.set_rasterization_zorder(0)


def draw_box_strip_on_ax(
    ax,
    df_plot,
    feat,
    ylab,
    animal_order,
    animal_map,
    ylim=None,
    pval_lookup=None
):
    style_axis(ax)

    if df_plot.empty:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=18
        )
        ax.set_ylabel(ylab)
        return

    sns.boxplot(
        data=df_plot,
        x="animal_label",
        y=feat,
        hue="session",
        order=[animal_map[a] for a in animal_order],
        hue_order=order_sessions,
        palette=palette,
        width=0.72,
        dodge=True,
        showfliers=False,
        linewidth=1.4,
        saturation=1,
        ax=ax
    )

    recolor_boxplot_outline(ax, order_sessions, palette)

    sns.stripplot(
        data=df_plot,
        x="animal_label",
        y=feat,
        hue="session",
        order=[animal_map[a] for a in animal_order],
        hue_order=order_sessions,
        palette=palette,
        dodge=True,
        jitter=0.23,
        size=1.55,
        alpha=0.38,
        linewidth=0,
        zorder=-10,
        ax=ax
    )

    if ax.legend_ is not None:
        ax.legend_.remove()

    ax.set_xlabel("")
    ax.set_ylabel(ylab, fontweight="normal")
    ax.set_xticklabels([animal_map[a] for a in animal_order], rotation=0)

    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if feat == "peak_power":
        ax.ticklabel_format(axis="y", style="sci", scilimits=(5, 5))
        ax.yaxis.get_offset_text().set_size(17)
        ax.yaxis.get_offset_text().set_color(TEXT_COLOR)

    if pval_lookup is not None:
        add_manual_brackets(
            ax=ax,
            feat=feat,
            animal_order=animal_order,
            animal_map=animal_map,
            pval_lookup=pval_lookup,
            ylim=ylim
        )


# =============================================================================
# --- CORE ALL-RIPPLES ANALYSIS ---
# =============================================================================
def run_all_ripples_analysis():
    tag = "ALL_RIPPLES"
    state_filter_description = "all_states_no_filter"

    merged_rows = []
    summary_rows = []
    debug_rows = []

    animal_dirs = find_animal_dirs(BASE_DIR_ROOT)

    print("\nDetected animal directories:")

    for x in animal_dirs:
        print(f"  {x['animal_name']} -> {x['path']}")

    for animal_info in animal_dirs:
        animal_key = animal_info["animal_key"]
        animal_name = animal_info["animal_name"]
        animal_path = animal_info["path"]

        if animal_key not in chan_dict:
            continue

        for sess_folder in os.listdir(animal_path):
            session_dir = os.path.join(animal_path, sess_folder)

            if not os.path.isdir(session_dir):
                continue

            session_label = session_from_folder(sess_folder)

            if session_label is None:
                continue

            rec_token = recording_token_from_session(session_label)
            session_post_window_sec = get_post_window_sec(animal_key, session_label)
            window_rule = get_window_rule(animal_key, session_label)

            print("\n" + "=" * 90)
            print(
                f"ANALYZING ALL RIPPLES: "
                f"animal={animal_name} | folder={sess_folder} | session={session_label}"
            )
            print(
                f"Window rule: {window_rule} | "
                f"0-{session_post_window_sec} s "
                f"({session_post_window_sec / 3600:.1f} h)"
            )
            print("=" * 90)

            dbg = {
                "analysis_mode": tag,
                "state_filter": state_filter_description,
                "animal": animal_key,
                "animal_name": animal_name,
                "folder": sess_folder,
                "session": session_label,
                "post_window_sec": session_post_window_sec,
                "post_window_hr": session_post_window_sec / 3600.0,
                "window_rule": window_rule,
                "status": "fail",
                "reason": ""
            }

            injection_file = find_injection_file(
                session_dir,
                animal_name,
                rec_token
            )

            if injection_file is None:
                print("[SKIP] Injection file not found.")
                dbg["reason"] = "no_injection"
                debug_rows.append(dbg)
                continue

            inj_time = load_injection_time_from_file(injection_file)

            if inj_time is None:
                print("[SKIP] Injection time could not be read.")
                dbg["reason"] = "bad_injection"
                debug_rows.append(dbg)
                continue

            print(f"[Injection file] {injection_file}")
            print(f"[Injection time] {inj_time:.3f} s")

            try:
                sess = ProcessData(session_dir)
                full_signal = sess.eegfile.get_signal()

            except Exception as e:
                print(f"[SKIP] ProcessData / EEG error: {e}")
                dbg["reason"] = "eeg_error"
                debug_rows.append(dbg)
                continue

            ripple_channel_id = chan_dict[animal_key][session_label]

            print(f"[Ripple channel] {ripple_channel_id}")

            chan_ids = np.arange(0, 32, 1)

            shank = Shank().auto_generate(
                columns=1,
                contacts_per_column=32,
                xpitch=0,
                ypitch=-50,
                channel_id=chan_ids
            )

            if hasattr(sess, "recinfo") and hasattr(sess.recinfo, "skipped_channels"):
                shank.set_disconnected_channels(sess.recinfo.skipped_channels)
            else:
                print(f"[WARNING] {sess.basepath}: missing skipped_channels (.xml?)")

            sess.prbgrp = ProbeGroup()
            sess.prbgrp.add_probe(Probe(shank))

            local_chans = get_good_local_channels(
                sess,
                ripple_channel_id,
                nbuf=nchan_buf
            )

            print(f"[Local chans] {local_chans.tolist()}")

            artifact_file = find_artifact_file(
                session_dir,
                animal_name,
                rec_token
            )

            art_epochs = load_artifact_epochs_from_file(artifact_file)

            print(f"[Artifact file] {artifact_file}")

            try:
                ripple_epochs, _ = oscillations.detect_ripple_epochs(
                    full_signal,
                    sess.prbgrp,
                    thresh=(FIXED_THRESHOLD, None),
                    ripple_channel=ripple_channel_id,
                    ignore_epochs=art_epochs,
                    mindur=0.05,
                    return_power=True
                )

            except Exception as e:
                print(f"[SKIP] Ripple detection error: {e}")
                dbg["reason"] = "ripple_detect_error"
                debug_rows.append(dbg)
                continue

            if ripple_epochs is None or len(ripple_epochs) == 0:
                print("[INFO] No ripples detected.")

                summary_rows.append({
                    "analysis_mode": tag,
                    "state_filter": state_filter_description,
                    "animal": animal_key,
                    "session": session_label,
                    "folder": sess_folder,
                    "n_total_detected": 0,
                    "n_post_window": 0,
                    "n_used": 0,
                    "post_window_sec": session_post_window_sec,
                    "post_window_hr": session_post_window_sec / 3600.0,
                    "window_rule": window_rule
                })

                dbg["reason"] = "no_ripple"
                debug_rows.append(dbg)
                continue

            try:
                ripple_res = Ripple.get_peak_ripple_freq(
                    sess.eegfile,
                    rpl_channel=ripple_channel_id,
                    rpl_epochs=ripple_epochs
                )

                rpl_df = ripple_res.to_dataframe()

            except Exception as e:
                print(f"[SKIP] Peak frequency computation error: {e}")
                dbg["reason"] = "peak_freq_error"
                debug_rows.append(dbg)
                continue

            if "peak_time" not in rpl_df.columns:
                rpl_df["peak_time"] = (
                    rpl_df["start"] + rpl_df["stop"]
                ) / 2

            rpl_df["duration"] = rpl_df["stop"] - rpl_df["start"]

            n_total = len(rpl_df)

            print(f"[Detected total ripples] {n_total}")

            # Align ripple times to injection
            rpl_df["start"] = rpl_df["start"] - inj_time
            rpl_df["stop"] = rpl_df["stop"] - inj_time
            rpl_df["peak_time"] = rpl_df["peak_time"] - inj_time

            # Retain peaks within the configured animal/session window.
            rpl_df = rpl_df[
                (rpl_df["peak_time"] >= 0) &
                (rpl_df["peak_time"] <= session_post_window_sec)
            ].copy()

            n_post_window = len(rpl_df)
            n_used = len(rpl_df)

            print(
                f"[POST ripples within applied window] {n_post_window} "
                f"(0-{session_post_window_sec} s; "
                f"{session_post_window_sec / 3600:.1f} h)"
            )

            if rpl_df.empty:
                summary_rows.append({
                    "analysis_mode": tag,
                    "state_filter": state_filter_description,
                    "animal": animal_key,
                    "session": session_label,
                    "folder": sess_folder,
                    "n_total_detected": n_total,
                    "n_post_window": 0,
                    "n_used": 0,
                    "post_window_sec": session_post_window_sec,
                    "post_window_hr": session_post_window_sec / 3600.0,
                    "window_rule": window_rule
                })

                dbg["reason"] = "no_post_window"
                debug_rows.append(dbg)
                continue

            print(f"[All ripples used] {n_used}")

            summary_rows.append({
                "analysis_mode": tag,
                "state_filter": state_filter_description,
                "animal": animal_key,
                "session": session_label,
                "folder": sess_folder,
                "n_total_detected": n_total,
                "n_post_window": n_post_window,
                "n_used": n_used,
                "post_window_sec": session_post_window_sec,
                "post_window_hr": session_post_window_sec / 3600.0,
                "window_rule": window_rule
            })

            try:
                valid_epochs = Epoch(pd.DataFrame({
                    "start": rpl_df["start"] + inj_time,
                    "stop": rpl_df["stop"] + inj_time,
                    "label": "ripple"
                }))

                sw_amp = compute_sharp_wave_amplitude(
                    sess.eegfile,
                    valid_epochs,
                    local_chans
                )

                if sw_amp.shape[0] != len(rpl_df):
                    m = min(sw_amp.shape[0], len(rpl_df))
                    rpl_df = rpl_df.iloc[:m].copy()
                    sw_amp = sw_amp[:m]

                rpl_df["sharp_wave_amplitude"] = sw_amp

            except Exception as e:
                print(f"[WARNING] SW amplitude computation error: {e}")
                rpl_df["sharp_wave_amplitude"] = np.nan

            rpl_df["session"] = session_label
            rpl_df["animal"] = animal_key
            rpl_df["folder"] = sess_folder
            rpl_df["analysis_mode"] = tag
            rpl_df["state_filter"] = state_filter_description
            rpl_df["post_window_sec"] = session_post_window_sec
            rpl_df["post_window_hr"] = session_post_window_sec / 3600.0
            rpl_df["window_rule"] = window_rule
            rpl_df["ripple_channel"] = int(ripple_channel_id)
            rpl_df["local_chans"] = ",".join(
                [str(int(x)) for x in local_chans]
            )

            dbg["status"] = "success"
            debug_rows.append(dbg)

            desired_cols = [
                "start",
                "stop",
                "peak_time",
                "duration",
                "peak_power",
                "peak_frequency_bp",
                "sharp_wave_amplitude",
                "animal",
                "session",
                "folder",
                "analysis_mode",
                "state_filter",
                "post_window_sec",
                "post_window_hr",
                "window_rule",
                "ripple_channel",
                "local_chans"
            ]

            existing_cols = [c for c in desired_cols if c in rpl_df.columns]
            export_df = rpl_df[existing_cols].copy()

            safe_animal = animal_key
            safe_session = session_label.replace(" ", "_")
            safe_folder = re.sub(r"[^a-zA-Z0-9_]", "_", sess_folder)

            fname_base = (
                f"{safe_animal}_{safe_session}_{safe_folder}_"
                f"ALL_RIPPLES_{POST_WINDOW_TAG}"
            )

            csv_path = os.path.join(EXPORT_DIR, fname_base + ".csv")
            npy_path = os.path.join(EXPORT_DIR, fname_base + ".npy")

            export_df.to_csv(csv_path, index=False)
            np.save(npy_path, export_df.to_dict(orient="list"))

            print(f"[EXPORT] CSV: {csv_path}")
            print(f"[EXPORT] NPY: {npy_path}")

            merged_rows.append(export_df)

    summary_df = pd.DataFrame(summary_rows)
    debug_df = pd.DataFrame(debug_rows)

    summary_csv = os.path.join(
        BASE_DIR_ROOT,
        f"ALL_RIPPLES_RippleCount_Summary_{POST_WINDOW_TAG}.csv"
    )

    debug_csv = os.path.join(
        BASE_DIR_ROOT,
        f"ALL_RIPPLES_Debug_{POST_WINDOW_TAG}.csv"
    )

    summary_df.to_csv(summary_csv, index=False)
    debug_df.to_csv(debug_csv, index=False)

    print(f"\n[SAVED] Summary CSV: {summary_csv}")
    print(f"[SAVED] Debug CSV: {debug_csv}")

    if not merged_rows:
        print("[INFO] No ALL_RIPPLES data were available.")
        return None

    full_df = pd.concat(merged_rows, ignore_index=True)

    merged_csv = os.path.join(
        BASE_DIR_ROOT,
        f"ALL_RIPPLES_MERGED_{POST_WINDOW_TAG}.csv"
    )

    merged_npy = os.path.join(
        BASE_DIR_ROOT,
        f"ALL_RIPPLES_MERGED_{POST_WINDOW_TAG}.npy"
    )

    full_df.to_csv(merged_csv, index=False)
    np.save(merged_npy, full_df.to_dict(orient="list"))

    print(f"[EXPORT] Merged CSV: {merged_csv}")
    print(f"[EXPORT] Merged NPY: {merged_npy}")

    return full_df


# =============================================================================
# --- STATS + 2x2 FIGURE ---
# =============================================================================
def stats_and_plot(full_df, tag):
    if full_df is None or full_df.empty:
        print(f"[INFO] Empty DataFrame for {tag}. Statistics and plots were not generated.")
        return

    full_df = full_df.copy()

    for feat in features.keys():
        if feat not in full_df.columns:
            full_df[feat] = np.nan

    animal_order = animals[:]
    animal_map = {a: f"Animal {i + 1}" for i, a in enumerate(animal_order)}

    full_df["animal_label"] = full_df["animal"].map(animal_map)

    out_pdf = os.path.join(
        BASE_DIR_ROOT,
        f"Final_2x2_Boxplots_{tag}_{POST_WINDOW_TAG}.pdf"
    )

    out_png = os.path.join(
        BASE_DIR_ROOT,
        f"Final_2x2_Boxplots_{tag}_{POST_WINDOW_TAG}.png"
    )

    out_svg = os.path.join(
        BASE_DIR_ROOT,
        f"Final_2x2_Boxplots_{tag}_{POST_WINDOW_TAG}.svg"
    )

    stats_csv = os.path.join(
        BASE_DIR_ROOT,
        f"MannWhitneyU_{tag}_{POST_WINDOW_TAG}.csv"
    )

    stats_rows = []
    pval_lookup = {}

    for feat in features.keys():
        df_plot = full_df.dropna(
            subset=[feat, "animal_label", "session"]
        ).copy()

        for a_lbl in [animal_map[a] for a in animal_order]:
            for s1, s2 in pairs_sessions:
                v1 = df_plot[
                    (df_plot["animal_label"] == a_lbl) &
                    (df_plot["session"] == s1)
                ][feat].dropna().values

                v2 = df_plot[
                    (df_plot["animal_label"] == a_lbl) &
                    (df_plot["session"] == s2)
                ][feat].dropna().values

                if len(v1) == 0 or len(v2) == 0:
                    continue

                try:
                    U_stat, p_val = stats.mannwhitneyu(
                        v1,
                        v2,
                        alternative="two-sided",
                        method="auto"
                    )

                except TypeError:
                    U_stat, p_val = stats.mannwhitneyu(
                        v1,
                        v2,
                        alternative="two-sided"
                    )

                stats_rows.append({
                    "comparison": f"{a_lbl} | {feat} | {s1} vs {s2}",
                    "alternative": "two-sided",
                    "test": "Mann-Whitney U",
                    "test_statistic": float(U_stat),
                    "p_value": float(p_val),
                    "feature": feat,
                    "animal_label": a_lbl,
                    "session_1": s1,
                    "session_2": s2,
                    "n_1": int(len(v1)),
                    "n_2": int(len(v2)),
                })

                pval_lookup[(feat, a_lbl, s1, s2)] = float(p_val)

    pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)

    print(f"[SAVED] Statistics CSV: {stats_csv}")

    panel_specs = [
        ("duration", "Duration (s)", None, "A"),
        ("peak_power", "Peak Power", zoom_specs["peak_power"], "B"),
        ("peak_frequency_bp", "Peak Frequency (Hz)", None, "C"),
        (
            "sharp_wave_amplitude",
            "SW Amplitude (mV)",
            zoom_specs["sharp_wave_amplitude"],
            "D"
        ),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(15.6, 10.6))
    fig.patch.set_facecolor(FIG_BG)
    axs = axs.flatten()

    for ax, (feat, ylab, ylim, panel_letter) in zip(axs, panel_specs):
        df_plot = full_df.dropna(
            subset=[feat, "animal_label", "session"]
        ).copy()

        draw_box_strip_on_ax(
            ax=ax,
            df_plot=df_plot,
            feat=feat,
            ylab=ylab,
            animal_order=animal_order,
            animal_map=animal_map,
            ylim=ylim,
            pval_lookup=pval_lookup
        )

        ax.text(
            -0.24,
            1.04,
            panel_letter,
            transform=ax.transAxes,
            fontsize=30,
            fontweight="normal",
            ha="left",
            va="top",
            color=TEXT_COLOR
        )

    legend_handles = [
        Patch(
            facecolor="none",
            edgecolor=palette["Saline 1"],
            linewidth=3.2,
            label="Saline 1"
        ),
        Patch(
            facecolor="none",
            edgecolor=palette["Psilocybin"],
            linewidth=3.2,
            label="Psilocybin"
        ),
        Patch(
            facecolor="none",
            edgecolor=palette["Saline 2"],
            linewidth=3.2,
            label="Saline 2"
        ),
    ]

    leg = fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.98),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        borderpad=0.4,
        handlelength=1.8,
        fontsize=28
    )

    leg.get_frame().set_facecolor("white" if not SAVE_TRANSPARENT else "none")
    leg.get_frame().set_edgecolor("#c8c8c8")
    leg.get_frame().set_linewidth(1.8)

    plt.subplots_adjust(
        left=0.08,
        right=0.80,
        top=0.95,
        bottom=0.08,
        wspace=0.16,
        hspace=0.14
    )

    facecol = "white" if not SAVE_TRANSPARENT else "none"

    fig.savefig(
        out_pdf,
        dpi=600,
        bbox_inches="tight",
        facecolor=facecol,
        transparent=SAVE_TRANSPARENT
    )

    fig.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
        facecolor=facecol,
        transparent=SAVE_TRANSPARENT
    )

    fig.savefig(
        out_svg,
        dpi=1200,
        bbox_inches="tight",
        facecolor=facecol,
        transparent=SAVE_TRANSPARENT
    )

    plt.close(fig)

    print(f"[SAVED] Plot PDF: {out_pdf}")
    print(f"[SAVED] Plot PNG: {out_png}")
    print(f"[SAVED] Plot SVG: {out_svg}")


# =============================================================================
# --- RUN ---
# =============================================================================
if __name__ == "__main__":

    print("\n" + "=" * 90)
    print("ALL-ripples SWR analysis")
    print("No sleep-state filtering is applied.")
    print(f"Window mode: {WINDOW_MODE}")
    print(f"BASE_DIR_ROOT: {BASE_DIR_ROOT}")
    print(f"Output tag: {POST_WINDOW_TAG}")
    print("=" * 90)

    df_all = run_all_ripples_analysis()

    stats_and_plot(
        df_all,
        tag="ALL_RIPPLES_NRKlayers"
    )

    print("\n[DONE] ALL-ripples analysis completed using the selected window mode.")
