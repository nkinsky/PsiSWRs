import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from neuropy.io.sleepscoremasterio import SleepScoreIO
from neuropy.core import Epoch
from Psilocybin.subjects import get_animal_num
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Define directories
primary_dir = Path("/Users/nkinsky/Documents/UM/Working/Psilocybin/Recording_Rats") # Path("D:\data\Nat\Psilocybin\Recording_Rats")
secondary_dir = Path("/Users/nkinsky/Documents/UM/Working/Alternation/Recording_Rats") # Path("D:\data\Nat\Alternation\Recording_Rats")
plot_dir = Path("/Users/nkinsky/Library/CloudStorage/Dropbox-UniversityofMichigan/Nathaniel Kinsky/Manuscripts/Psilocybin/plots")

# List of animals
animals = ["Finn", "Rey", "Rose", "Finn2"]

epoch_min_time = 30  # seconds

# Collect data
data = []
for animal in animals:
    # Define sessions and base_dirs based on animal
    sessions = ["saline1", "psilocybin", "saline2"]
    session_labels = ["Saline1", "Psilocybin", "Saline2"]
    base_dirs = [primary_dir, primary_dir, primary_dir]
    # if animal == "Finn2":
    #     sessions = sessions[1:]
    #     session_labels = session_labels[1:]
    #     base_dirs = base_dirs[1:]

    for idx, session in enumerate(sessions):
        base_dir = sorted((base_dirs[idx] / animal).glob(f"*_{session}"))[0]

        print(f"Processing {animal} {session_labels[idx]}: {base_dir}")

        try:
            # Load sleep states
            sleep = SleepScoreIO(base_dir)
            brainstates = sleep.read_states(plot_states=False)
            print(f"Loaded brainstates: {len(brainstates)} epochs")

            # Load injection time
            try:
                inj_epochs = Epoch(epochs=None, file=sorted(base_dir.glob("*.injection.npy"))[0])
                inj_time = inj_epochs["POST"].starts[0]
                print(f"Injection time: {inj_time} seconds")
            except IndexError:
                print(f"injection.npy not found in {base_dir}")
                if session_labels[idx] == "Alternation":
                    inj_time = 0  # For alternation, use session start as reference
                    print("Using session start (0) as reference for Alternation")
                else:
                    inj_time = None

            # Find latency to first NREM after reference time (only NREM >=30 seconds)
            if 'nrem' in brainstates.labels and inj_time is not None:
                nrem_epochs = brainstates['nrem']
                post_ref_mask = nrem_epochs.starts >= inj_time
                long_mask = nrem_epochs.durations >= epoch_min_time
                valid_nrem_starts = nrem_epochs.starts[post_ref_mask & long_mask]
                if len(valid_nrem_starts) > 0:
                    first_onset = valid_nrem_starts.min() - inj_time  # Latency in seconds
                    print(f"Time to first NREM (>={epoch_min_time}s): {first_onset} seconds")
                else:
                    first_onset = np.nan
                    print(f"No NREM (>={epoch_min_time}s) found")
            else:
                first_onset = np.nan
                if 'nrem' not in brainstates.labels:
                    print("No NREM epochs found")
                if inj_time is None:
                    print("No reference time found")

            # Append to data
            data.append({
                "animal": animal,
                "animal_num": get_animal_num(animal),
                "session": session_labels[idx],
                "onset_time": first_onset
            })
        except Exception as e:
            print(f"Error processing {base_dir}: {e}")
            continue

print(f"Collected data for {len(data)} entries")

# Create DataFrame
df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(df.head())

# Convert onset_time to minutes
df['onset_time'] = df['onset_time'] / 60

# Plot using seaborn stripplot
if df.empty:
    print("No data collected to plot")
else:
    plt.figure(figsize=(3, 2.5), layout='tight')
    # Define custom palette for more distinct colors
    custom_palette = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}
    sns.stripplot(data=df, x="session", y="onset_time", hue="animal_num", dodge=True, palette=custom_palette)
    plt.title(f"Latency to 1st NREM >={epoch_min_time}s")
    plt.xlabel("Session")
    plt.ylabel("Time (min)")
    plt.legend(title="Animal Number")
    sns.despine(ax=plt.gca())
    plt.savefig(plot_dir / "SleepOnset_time.pdf")
    # plt.savefig(Path(r"D:\data\Nat\Psilocybin\Recording_Rats") / "SleepOnset_time.pdf")
    plt.show()
