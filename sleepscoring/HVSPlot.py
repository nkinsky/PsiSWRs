import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from neuropy.core.epoch import Epoch
from neuropy.io.neuroscopeio import NeuroscopeIO
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

hvfile_dict = {"Rey": "hv4", "Finn": "hv4", "Rose": "hv4", "Finn2": "hv4"}

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
animals = ["Rey", "Finn", "Rose", "Finn2"]
sessions = ["Saline1", "Psilocybin", "Saline2"]

# Collect data
data = []
for animal in animals:
    for session in sessions:
        date = date_dict[animal][session]
        session_lower = session.lower()
        base_path = rf"D:\data\Nat\Psilocybin\Recording_Rats\{animal}\{date}_{session_lower}"

        print(f"Processing {animal} {session}: {base_path}")

        # Find HVS epochs file using glob
        base_dir = Path(base_path)
        # all_evt_files = sorted(base_dir.glob("*.evt*"))
        # evt_files = [f for f in all_evt_files if "hv" in f.name]
        # print(f"Found evt files: {all_evt_files}")
        # print(f"Filtered HVS evt files: {evt_files}")
        # if not evt_files:
        #     print(f"Warning: No HVS evt files found in {base_path}")
        #     continue
        # evt_file = evt_files[0]  # Use the first HVS evt file found
        evt_file = sorted(base_dir.glob(f"*.evt.{hvfile_dict[animal]}"))[0]

        try:
            # Load HVS epochs
            hvs_epochs = NeuroscopeIO.event_to_epochs(evt_file)
            print(f"Loaded HVS epochs: {len(hvs_epochs)} epochs")

            # Load injection time
            inj_epochs = Epoch(epochs=None, file=sorted(Path(base_path).glob("*.injection.npy"))[0])
            inj_time = inj_epochs["POST"].starts[0]
            print(f"Injection time: {inj_time} seconds")

            # Make time 0 = inj time
            hvs_epochs = hvs_epochs.shift(-inj_time)

            # Filter to first hour (0-3600 seconds)
            hvs_first_hour = hvs_epochs.time_slice(0, 3600)
            print(f"First hour epochs: {len(hvs_first_hour)} epochs")

            # Sum durations
            total_hvs_time = hvs_first_hour.durations.sum()
            print(f"Total HVS time: {total_hvs_time}")

            # Append to data
            data.append({
                "animal": animal,
                "animal_num": get_animal_num(animal),
                "session": session_lower,
                "total_hvs_time": total_hvs_time
            })
        except Exception as e:
            print(f"Error processing {evt_file}: {e}")
            continue

print(f"Collected data for {len(data)} entries")

# Create DataFrame
df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(df.head())

# Plot using seaborn stripplot
plt.figure(figsize=(3, 2.5), layout='tight')
# Define custom palette for more distinct colors
custom_palette = {1: 'blue', 2: 'red', 3: 'green', 4: 'orange'}
sns.stripplot(data=df, x="session", y="total_hvs_time", hue="animal_num", dodge=True, palette=custom_palette)
plt.title("HVS 1st Hour Post-Injection")
plt.xlabel("Session")
plt.ylabel("Time (seconds)")
sns.despine(ax=plt.gca())
# plt.legend(title="Animal Number")
plt.savefig(plot_dir / "HVSPlot.pdf")
plt.show()
