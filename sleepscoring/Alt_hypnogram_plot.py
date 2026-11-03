from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
from neuropy.io.sleepscoremasterio import SleepScoreIO
from neuropy.plotting.epochs import plot_hypnogram
from neuropy.utils.plot_util import match_axis_lims
import seaborn as sns
from Psilocybin.subjects import get_animal_num
alt_dir = Path("/Users/nkinsky/Documents/UM/Working/Alternation/Recording_Rats")
plot_dir = Path("/Users/nkinsky/Library/CloudStorage/Dropbox-UniversityofMichigan/Nathaniel Kinsky/Manuscripts/Psilocybin/plots")

unit = "h"  # "s" = seconds or "h" = hours
consolidate_wake = True

assert unit in ["s", "h"]
unit_norm = 1 if unit == "s" else 3600

# session_name = "alternation"
animal_names = ["Finn", "Rey", "Rose"]
session_names = ["alternation3", "alternation4", "alternation3"]
fig, ax = plt.subplots(1, 3, figsize=(11.3, 1.2))
fig.suptitle("Alternation")
ax[0].set_title(f"Animal {get_animal_num('Finn')}")
ax[1].set_title(f"Animal {get_animal_num('Rey')}")
ax[2].set_title(f"Animal {get_animal_num('Rose')}")
for ids, (animal_name, session_name) in enumerate(zip(animal_names, session_names)):
    base_dir = sorted((alt_dir / animal_name).glob(f"*_{session_name}*"))[0]
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

    plot_hypnogram(brainstates, ax=ax[ids], annotate=True, labels=["NREM", "REM", "WAKE"], unit=unit)
    ax[ids].set_xticks(np.array([0, 3600, 7200, 10800, 14400]) / unit_norm)
    ax[ids].set_xlabel("Time (s)")
    ax[ids].axis("on")
    sns.despine(ax=ax[ids], left=True)
match_axis_lims(ax, "x")
plt.show()
fig.savefig(plot_dir / "alternation_hypnograms.pdf")
