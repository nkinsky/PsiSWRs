import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from neuropy.core.epoch import Epoch
from neuropy.io.sleepscoremasterio import SleepScoreIO
from Psilocybin.subjects import get_animal_num
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

plot_dir = Path("/Users/nkinsky/Library/CloudStorage/Dropbox-UniversityofMichigan/Nathaniel Kinsky/Manuscripts/Psilocybin/plots")

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
    for session in sessions:
        date = date_dict[animal][session]
        session_lower = session.lower()
        base_path = f"/Users/nkinsky/Documents/UM/Working/Psilocybin/Recording_Rats/{animal}/{date}_{session_lower}"

        print(f"Processing {animal} {session}: {base_path}")

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
                "session": session_lower,
                "total_nrem_time": total_nrem_time
            })
        except Exception as e:
            print(f"Error processing {base_path}: {e}")
            continue

print(f"Collected data for {len(data)} entries")

# Create DataFrame
df = pd.DataFrame(data)
df["session"] = pd.Categorical([_.capitalize() for _ in df.session.values], sessions)
print(f"DataFrame shape: {df.shape}")
print(df.head())

# Create broken axis plot with adjusted sizing (subplots to skip middle values)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(3, 2.5), height_ratios=(1.7, 5), layout='tight')
fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# Define custom palette for more distinct colors
custom_palette = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}

# Plot on both axes
sns.stripplot(data=df, x="session", y="total_nrem_time", hue="animal_num", dodge=True, palette=custom_palette, ax=ax1,
              legend=False)
sns.stripplot(data=df, x="session", y="total_nrem_time", hue="animal_num", dodge=True, palette=custom_palette, ax=ax2,
              legend=False)

# Top subplot: 1790-1910 to make axis line shorter and bring 1800/1900 closer
ax1.set_ylim(1680, 1920)
ax1.set_yticks([1700, 1900])

# Bottom subplot: -10-500 to show dots at 0
ax2.set_ylim(-50, 620)

# Hide spines to connect
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

# Set title and labels
fig.suptitle("Total NREM 1st Hr Post-Injection", y=0.95)
ax2.set_xlabel("Session")
ax1.set_ylabel("")
ax2.set_ylabel("Time (sec)")

# Legend on ax1 (top)
handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles, labels, title="Animal Number", loc='upper right')

# Despine
sns.despine(ax=ax1, bottom=True)
sns.despine(ax=ax2)

plt.savefig(plot_dir / "NREMtime_Plot_broken.pdf")
plt.show()

from scipy.stats import mannwhitneyu
sal1 = df[df.session == "Saline1"].total_nrem_time.values
sal2 = df[df.session == "Saline2"].total_nrem_time.values
psi = df[df.session == "Psilocybin"].total_nrem_time.values
print("Running Stats")
print(mannwhitneyu(sal1, psi))
print(mannwhitneyu(sal2, psi))
