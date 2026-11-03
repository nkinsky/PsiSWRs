import numpy as np
from copy import deepcopy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from neuropy.io.sleepscoremasterio import SleepScoreIO
from neuropy.core import Epoch
from Psilocybin.subjects import get_animal_num

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plot_dir = Path("/Users/nkinsky/Library/CloudStorage/Dropbox-UniversityofMichigan/Nathaniel Kinsky/Manuscripts/Psilocybin/plots")

# Channel dictionary from HVS_Detection.py
chan_dict = {
    "Finn":  {"Saline1": 27, "Psilocybin": 27, "Saline2": 27},
    "Rey":   {"Saline1": 21, "Psilocybin": 21, "Saline2": 21},
    "Rose":  {"Saline1": 26, "Psilocybin": 26, "Saline2": 26},
    "Finn2": {"Saline1": 4,  "Psilocybin": 4,  "Saline2": 4}
}

hvfile_dict = {"Rey": "hv3", "Finn": "hv3", "Rose": "hv3", "Finn2": "hv3"}

# Date dictionary - UPDATE THESE FOR EACH ANIMAL AS NEEDED
date_dict = {
    "Rey": {
        "Saline1": "2022_06_01",
        "Psilocybin": "2022_06_02",
        "Saline2": "2022_06_03"
    },
    "Finn": {
        "Saline1": "2022_02_15",
        "Psilocybin": "2022_02_17",
        "Saline2": "2022_02_18"
    },
    "Rose": {
        "Saline1": "2022_08_09",
        "Psilocybin": "2022_08_10",
        "Saline2": "2022_08_11"
    },
    "Finn2": {
        "Saline1": "2023_05_24",
        "Psilocybin": "2023_05_25",
        "Saline2": "2023_05_26"
    }
}

# List of animals and sessions
animals = ["Finn", "Rey", "Rose", "Finn2"]
sessions = ["Saline1", "Psilocybin", "Saline2"]

# Collect data
data = []
for animal in animals:
    data_dict = {}
    for session in sessions:
        date = date_dict[animal][session]
        session_lower = session.lower()
        base_path = f"/Users/nkinsky/Documents/UM/Working/Psilocybin/Recording_Rats/{animal}/{date}_{session_lower}"

        print(f"Processing {animal} {session}: {base_path}")

        # Find HVS epochs file using glob
        base_dir = Path(base_path)
        hvs_files = list(base_dir.glob("*.hvs_epochs.npy"))
        if not hvs_files:
            print(f"No .hvs_epochs.npy file found in {base_dir}")
            continue
        hvs_file = sorted(hvs_files)[0]

        try:
            # Load HVS epochs
            hvs_epochs = Epoch(epochs=None, file=hvs_file)
            print(f"Loaded HVS epochs: {len(hvs_epochs)} epochs")

            # Load injection time
            inj_epochs = Epoch(epochs=None, file=sorted(Path(base_path).glob("*.injection.npy"))[0])
            inj_time = inj_epochs["POST"].starts[0]
            print(f"Injection time: {inj_time} seconds")

            # Zero HVS epochs to 0 = inj time
            hvs_epochs = hvs_epochs.shift(-inj_time)

            # Filter to first hour (0-3600 seconds)
            post_length_use = np.min((3600, hvs_epochs.stops.max()))
            hvs_epochs = hvs_epochs.time_slice(0, post_length_use)

            # Sum durations for entire session
            total_hvs_time = hvs_epochs.durations.sum()
            print(f"Total HVS time: {total_hvs_time}")

            # Calculate proportion as percentage
            proportion = (total_hvs_time / post_length_use) * 100
            print(f"Proportion: {proportion}%")

            # Append to data
            data.append({
                "animal": animal,
                "animal_num": get_animal_num(animal),
                "session": session,
                "total_time": total_hvs_time,
                "tag": "HVS"
            })
        except Exception as e:
            print(f"Error processing {hvs_file}: {e}")
            continue

        try:
            # Load sleep states
            sleep = SleepScoreIO(base_path)
            brainstates = sleep.read_states(plot_states=False)
            print(f"Loaded brainstates: {len(brainstates)} epochs")

            # Load injection time
            inj_epochs = Epoch(epochs=None, file=sorted(Path(base_path).glob("*.injection.npy"))[0])
            inj_time = inj_epochs["POST"].starts[0]
            print(f"Injection time: {inj_time} seconds")

            # Make time 0 = injection time
            brainstates = brainstates.shift(-inj_time)

            # Get NREM epochs
            if 'nrem' in brainstates.labels:
                nrem_epochs = brainstates['nrem']
                print(f"NREM epochs: {len(nrem_epochs)}")

                # Filter to first hour (0-3600 seconds)
                nrem_first_hour = nrem_epochs.time_slice(0, np.min((3600, brainstates.stops.max())))
                print(f"First hour NREM epochs: {len(nrem_first_hour)} epochs")

                # Sum durations
                total_nrem_time = nrem_first_hour.durations.sum()
                print(f"Total NREM time: {total_nrem_time} seconds")
            else:
                total_nrem_time = 0
                print("No NREM epochs found")

            # Append to data
            data.append({
                "animal": animal,
                "animal_num": get_animal_num(animal),
                "session": session,
                "total_time": total_nrem_time,
                "tag": "NREM"
            })

        except Exception as e:
            print(f"Error processing {base_path}: {e}")
            continue

# Create DataFrame
df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(df.head())

# Plot using seaborn stripplot
if df.empty:
    print("No data collected to plot")
else:
    # fig1 = plt.figure(figsize=(3, 2.5), layout='tight')
    # sns.stripplot(data=df, x="session", y="total_time", hue="tag", dodge=True)
    # sns.despine(ax=plt.gca())
    # # fig1.savefig(plot_dir / "HVSTotalPlot_percent.pdf")
    # plt.show()
    # Create broken axis plot with adjusted sizing (subplots to skip middle values)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3, 2.5), height_ratios=(1.7, 5), layout='tight')
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes

    # Plot on both axes
    sns.stripplot(data=df, x="session", y="total_time", hue="tag", dodge=True, legend=False, ax=ax1)
    sns.stripplot(data=df, x="session", y="total_time", hue="tag", dodge=True, linewidth=0.5, edgecolor="w",
                  legend=False, ax=ax2)
    ax2.set_xlabel("")
    ax2.set_ylabel("Time (s)")
    ax1.set_ylabel("")
    fig.suptitle("NREM vs HVS", y=0.95)

    # Top subplot: 1790-1910 to make axis line shorter and bring 1800/1900 closer
    ax1.set_ylim(1680, 1920)
    ax1.set_yticks([1700, 1900])

    # Bottom subplot: -10-500 to show dots at 0
    ax2.set_ylim(-50, 620)

    # Hide spines to connect
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    # ax1.tick_params(axis='x', bottom=False)x
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()

    # Despine
    sns.despine(ax=ax1, bottom=True)
    sns.despine(ax=ax2)

    fig.savefig(plot_dir / "NREM_v_HVS.pdf")

    plt.show()

# Now run stats
from scipy.stats import mannwhitneyu
print("Running Stats")
for session in sessions:
    df_sesh = df[df.session == session]
    hvs_vals = df_sesh[df_sesh.tag == "HVS"]["total_time"].values
    nrem_vals = df_sesh[df_sesh.tag == "NREM"]["total_time"].values
    print(f"{session} NREM vs HVS 2-sided MWU pval={mannwhitneyu(nrem_vals, hvs_vals)}")


