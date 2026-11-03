from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
from neuropy.io.sleepscoremasterio import SleepScoreIO
from neuropy.plotting.epochs import plot_hypnogram
from neuropy.core.epoch import Epoch
from neuropy.utils.plot_util import match_axis_lims
import seaborn as sns
from Psilocybin.subjects import get_animal_num

consolidate_wake = True  # if True, combine "quiet" and "active" into "wake"
unit = "h"  # "s" = seconds or "h" = hours
rec_dir = Path("/Users/nkinsky/Documents/UM/Working/Psilocybin/Recording_Rats")
plot_dir = Path("/Users/nkinsky/Library/CloudStorage/Dropbox-UniversityofMichigan/Nathaniel Kinsky/Manuscripts/Psilocybin/plots")

sessions = ["saline1", "psilocybin", "saline2"]
animal_name = "Rose"
animal_dir = rec_dir / animal_name
fig, ax = plt.subplots(1, 3, figsize=(8.5, 0.9))
fig.suptitle(f"Animal {get_animal_num(animal_name)}")
ax[0].set_title("Saline1")
ax[1].set_title("Psilocybin")
ax[2].set_title("Saline2")

assert unit in ["s", "h"]
unit_norm = 1 if unit == "s" else 3600

for ids, session_name in enumerate(sessions):
    base_dir = sorted(animal_dir.glob(f"*_{session_name}"))[0]

    sleep = SleepScoreIO(base_dir)
    brainstates = sleep.read_states(plot_states=False)

    # Consolidate wake states
    if consolidate_wake:
        labels = brainstates.labels
        labels[np.isin(brainstates.labels, ["active", "quiet"])] = "wake"
        brainstates = brainstates.set_labels(labels)
        state_labels = ["NREM", "REM", "WAKE"]
    else:
        state_labels = ["NREM", "REM", "QUIET", "ACTIVE"]

    # Make states all upper-case
    brainstates = brainstates.set_labels([state.upper() for state in brainstates.labels])
    # if unit == "h":
    #     brainstates = brainstates.scale(1 / unit_norm)

    try:
        inj_epochs = Epoch(epochs=None, file=sorted(base_dir.glob("*.injection.npy"))[0])
        inj_time = inj_epochs["POST"].starts[0]
    except IndexError:
        print(f"injection.npy not found in {base_dir}")
        inj_time = None

    plot_hypnogram(brainstates, ax=ax[ids], annotate=True, labels=["NREM", "REM", "WAKE"], unit=unit)
    if inj_time is not None:
        ax[ids].axvline(inj_time / unit_norm, color='red', linestyle='--', linewidth=1)

    if animal_name == "Finn2":
        ax[ids].set_xticks(np.array([0, 3600, 7200, 10800, 14400]) / unit_norm)
    else:
        ax[ids].set_xticks(np.array([0, 3600, 7200]) / unit_norm)

    ax[ids].set_xlabel(f"Time ({unit})")
    ax[ids].axis("on")
    sns.despine(ax=ax[ids], left=True)
    ax[ids].set_yticks([])

match_axis_lims(ax, "x")
plt.show()
fig.savefig(plot_dir / f"{animal_name}_hypnogramsnew.pdf")
