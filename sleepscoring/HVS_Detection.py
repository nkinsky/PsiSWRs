import numpy as np
from pathlib import Path
import pathlib
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
import scipy.io as sio
from neuropy.analyses.oscillations import _detect_freq_band_epochs
from neuropy.core import Signal, Epoch, ProbeGroup
from neuropy.utils import signal_process
from neuropy.io.neuroscopeio import NeuroscopeIO
from neuropy.io.binarysignalio import BinarysignalIO

from Psilocybin import subjects

print(f"HVS_Detection.py Path module={Path.__module__} and pathlib.__file__ = {pathlib.__file__}")

chan_dict = {
    "Rey":   {"Saline1": 21, "Psilocybin": 21, "Saline2": 21},
    "Finn":  {"Saline1": 27, "Psilocybin": 27, "Saline2": 27},
    "Rose":  {"Saline1": 26, "Psilocybin": 26, "Saline2": 26},
    "Finn2": {"Saline1": 4,  "Psilocybin": 4,  "Saline2": 4}
}

# animal_dir = Path(r"D:\data\Nat\Psilocybin\Recording_Rats\Rose\2022_08_10_psilocybin")
# sessions = ["saline1", "psilocybin", "saline2"]

def get_good_times(basedir):
    basedir = Path(basedir)
    files = sorted(basedir.glob("*.SleepState.states.mat"))

    assert len(files) == 1, f"1 state file expected, f{len(files)} files found"
    file = files[0]

    states_from_mat = sio.loadmat(file, simplify_cells=True)
    good_times = states_from_mat['SleepState']['detectorinfo']['detectionparms']['SleepScoreMetrics']['t_clus']

    return good_times

def detect_hvs_epochs(
    signal: Signal,
    probegroup: ProbeGroup = None,
    freq_band=(10, 20),  # Spindle frequency band
    freq_band2=(4, 9),  # Low frequency band for HVS intersection
    thresh=(2, None),  # Threshold for spindle detection
    thresh2=(3, None),  # Threshold for low band HVS detection
    edge_cutoff=1.0,  # Edge cutoff for z-scored power
    mindur=0.20,
    maxdur=20,
    mergedist=0.10,
    sigma=0.125,
    ignore_epochs: Epoch = None,
    return_power = False,
    basedir = None,
):
    """
    Detect high voltage spindle (HVS) epochs.

    HVS are defined as spindles that overlap between two frequency bands:
    - High frequency band (default 10-20 Hz) with threshold 2 SD
    - Low frequency band (default 4-9 Hz) with threshold _ SD

    Based on Buzsaki et al., 1988 and paper protocols.

    Parameters
    ----------
    signal : Signal
        LFP signal object
    probegroup : ProbeGroup, optional
        Probe group for channel selection
    freq_band : tuple
        High frequency band for spindle detection (default (10, 20))
    freq_band2 : tuple
        Low frequency band for HVS intersection (default (6, 9))
    thresh : tuple
        Threshold for high frequency band detection (default (2, None))
    thresh2 : tuple
        Threshold for low frequency band detection (default (7, None))
    edge_cutoff : float, optional
        Edge cutoff for z-scored power (default 1.0)
    mindur : float
        Minimum duration in seconds (default 0.20)
    maxdur : float
        Maximum duration in seconds (default 8)
    mergedist : float
        Merge distance for overlapping epochs (default 0.10)
    sigma : float, optional
        Gaussian smoothing sigma (default 0.125)
    ignore_epochs : Epoch, optional
        Epochs to ignore during detection

    Returns
    -------
    Epoch
        Detected HVS epochs (intersection of high and low band detections)
    """
    print("Starting HVS detection")
    print(f"Signal sampling rate: {signal.sampling_rate}, channels: {signal.channel_id}")

    if probegroup is None:
        selected_chans = signal.channel_id
        traces = signal.traces
    else:
        # Select best channel for HVS detection (similar to spindle detection)
        channel_ids = np.concatenate(probegroup.get_connected_channels(groupby="shank")).astype("int")
        signal_slice = signal.time_slice(channel_id=channel_ids, t_start=signal.t_start,
                                       t_stop=min(signal.t_start + 3600, signal.t_stop))
        hil_stat = signal_process.hilbert_amplitude_stat(
            signal_slice.traces, freq_band=freq_band, fs=signal.sampling_rate, statistic="mean"
        )
        selected_chans = channel_ids[np.argmax(hil_stat)].reshape(-1)
        traces = signal.time_slice(channel_id=selected_chans).traces.reshape(1, -1)

    print(f"Selected channel for HVS: {selected_chans}")

    # Handle ignore epochs and bad times from SleepScoreMaster
    # if basedir is not None:
    #     good_times = get_good_times(basedir)
    #     bad_times_bool = np.ones(len(signal.time), dtype=bool)
    #     good_indices = np.searchsorted(signal.time, good_times)
    #     good_indices = good_indices[good_indices < len(signal.time)]
    #     bad_times_bool[good_indices] = False
    #     bad_epochs = Epoch.from_boolean_array(bad_times_bool, signal.time)
    #     if ignore_epochs is not None:
    #         ignore_epochs = ignore_epochs + bad_epochs
    #     else:
    #         ignore_epochs = bad_epochs

    if ignore_epochs is not None and ignore_epochs.n_epochs > 0:
        ignore_times = ignore_epochs.shift(-signal.t_start).as_array()
    else:
        ignore_times = None

    # Detect spindles in high frequency band (10-20 Hz)
    epochs_high, power_high = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band,
        thresh=thresh,
        edge_cutoff=edge_cutoff,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        sigma=sigma,
        ignore_times=ignore_times,
        return_power=return_power,
    )
    epochs_high = epochs_high.shift(dt=signal.t_start)

    # Detect events in low frequency band (4-9 Hz) with higher threshold
    epochs_low, power_low = _detect_freq_band_epochs(
        signals=traces,
        freq_band=freq_band2,
        thresh=thresh2,
        edge_cutoff=edge_cutoff,
        mindur=mindur,
        maxdur=maxdur,
        mergedist=mergedist,
        fs=signal.sampling_rate,
        sigma=sigma,
        ignore_times=ignore_times,
        return_power=return_power,
    )
    epochs_low = epochs_low.shift(dt=signal.t_start)

    # Find intersection of the two detections
    if len(epochs_high) == 0 or len(epochs_low) == 0:
        epochs = Epoch.from_array([], [], [])  # Empty epoch if either detection has no events
    else:
        res = 1 / signal.sampling_rate  # time resolution
        t_start = np.min((epochs_high.starts.min(), epochs_low.starts.min()))
        t_stop = np.max((epochs_high.stops.max(), epochs_low.stops.max()))
        times, bool1 = epochs_high.to_point_process(t_start, t_stop, bin_size=res)
        _, bool2 = epochs_low.to_point_process(t_start, t_stop, bin_size=res)
        epochs = Epoch.from_boolean_array(np.bitwise_and(bool1, bool2), times)
    epochs.metadata = dict(channels=selected_chans, freq_band=freq_band, freq_band2=freq_band2)
    print(f"Detected {len(epochs_high)} high-band spindles, {len(epochs_low)} low-band events, {len(epochs)} HVS epochs after intersection.")

    if not return_power:
        return epochs
    else:
        return epochs, power_high, power_low

def plot_hvs_epochs(epochs, ax=None, color='red', alpha=0.5, label='HVS'):
    """
    Plot HVS epochs over time.

    Parameters
    ----------
    epochs : Epoch
        HVS epochs to plot
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates new figure.
    color : str, optional
        Color for the epochs
    alpha : float, optional
        Transparency
    label : str, optional
        Label for the plot

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))

    for start, stop in epochs.as_array():
        ax.axvspan(start, stop, alpha=alpha, color=color, label=label if start == epochs.as_array()[0][0] else "")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('High Voltage Spindles (mV)')
    ax.set_title('High Voltage Spindles over Time')
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    if label:
        ax.legend()

    return ax

if __name__ == "__main__":
    # signal = Signal(traces=np.random.randn(1, 100000), sampling_rate=1250, channel_id=[0])
    # hvs_epochs = detect_hvs_epochs(signal)
    # plot_hvs_epochs(hvs_epochs)
    # plt.show()
    # plot_hvs_epochs(hvs_epochs)

    animal_name = "Rey"
    session_name = "Saline1"
    animal_dir = subjects.get_psi_dir(animal_name, session_name)
    # fig, ax = plt.subplots(1, 3, figsize=(11.3, 1.2))
    # fig.suptitle(animal_name)
    # ax[0].set_title("Saline1")
    # ax[1].set_title("Psilocybin")
    # ax[2].set_title("Saline2")

    xml_file = sorted(animal_dir.glob("*.xml"))[0]

    thresh2 = (4, None)
    recinfo = NeuroscopeIO(xml_file)
    eeg_file = recinfo.eeg_filename

    # Load signal from binary file
    pyr_channel = chan_dict[animal_name]["Psilocybin"]
    # loader = BinarysignalIO(r'D:\data\Nat\Psilocybin\Recording_Rats\Finn\2022_02_17_psilocybin\2022_02_17_psilocybin.eeg', dtype="int16", n_channels=35, sampling_rate=1250)
    # signal = loader.get_signal(channel_indx=pyr_channel)
    eegfile = BinarysignalIO(recinfo.eeg_filename, dtype="int16", n_channels=recinfo.n_channels, sampling_rate=recinfo.eeg_sampling_rate)
    signal = eegfile.get_signal(channel_indx=pyr_channel)

    # Load in artifact.npy file for each session as "art_epochs"
    art_epochs = Epoch(epochs=None, file=sorted(animal_dir.glob("*.artifact.npy"))[0])
    hvs_epochs = detect_hvs_epochs(signal, ignore_epochs=art_epochs, thresh2=thresh2)
    hvs_epochs.save(recinfo.eeg_filename.with_suffix(".hvs_epochs.npy"))  # save to .npy file for NeuroPy
    recinfo.write_epochs(hvs_epochs, f'hv{thresh2[0]}')

    plot_hvs_epochs(hvs_epochs)
    plt.show()

    # # Plot using seaborn stripplot
    # plt.figure(figsize=(10, 6))
    # sns.stripplot(data=df, x="session", y="total_hvs_time", hue="animal", dodge=True)
    # plt.title("Total HVS Time in First Hour Post-Injection")
    # plt.xlabel("Session")
    # plt.ylabel("Total HVS Time (seconds)")
    # plt.legend(title="Animal")