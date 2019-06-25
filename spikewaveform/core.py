import numpy as np
import scipy
from scipy.cluster.vq import kmeans, vq


def calculate_waveform_features(sptrs):
    """Calculates waveform features for spiketrains; full-width half-maximum
    (half width) and minimum-to-maximum peak width (peak-to-peak width) for
    mean spike, and average firing rate.

    Parameters
    ----------
    sptrs : list
        a list of neo spiketrains

    Returns
    ----------
    half_width_mean : array of floats
        full-width half-maximum (in ms) for mean spike of each spiketrain
    peak_to_peak_mean : array of floats
        minimum-to-maximum peak width (in ms) for mean spike of each spiketrain
    average_firing_rate : list of floats
        average firing rate (in Hz) for each spiketrain
    """
    for sptr in sptrs:
        if not hasattr(sptr.waveforms, 'shape'):
            raise AttributeError('Argument provided (sptr) has no attribute\
                                  waveforms.shape')

    average_firing_rate = calculate_average_firing_rate(sptrs)

    times = np.arange(sptrs[0].waveforms.shape[2], dtype=np.float32) /\
        sptrs[0].sampling_rate

    half_width_list = []
    peak_to_peak_list = []
    for i in range(len(sptrs)):
        mean_wf = np.mean(sptrs[i].waveforms, axis=0).magnitude
        max_amplitude_channel = np.argmin(mean_wf.min(axis=1))
        wf = mean_wf[max_amplitude_channel, :]

        half_width_list.append(np.array(half_width(wf, times)))
        peak_to_peak_list.append(np.array(peak_to_trough(wf, times)))

    return half_width_list, peak_to_peak_list, average_firing_rate


def interpolate(x, x0, x1, y0, y1):
    p = sorted([x0, x1])
    assert p[0] < x and x < p[1], 'x not between x0 and x1'
    return y0 + (x - x0) * ((y1 - y0) / (x1 - x0))


def half_width(wf, times):
    """Calculates full-width half-maximum (half width) for spikes.

    Parameters
    ----------
    wf : array
        Spike waveform
    times : array
        array of times for when the spike waveform is measured

    Returns
    ----------
    half_width : float
        full-width half-maximum
    """
    throughs,_ = scipy.signal.find_peaks(-wf)
    index_min = throughs[np.argmin(wf[throughs])]
    half_amplitude = wf[index_min] * 0.5
    half_wf = wf - half_amplitude
    # there might be multiple intersections, we take the closest to the peak
    shifts_1 = np.diff(half_wf[:index_min] > 0)
    shifts_1_idxs, = np.where(shifts_1 == 1)
    p1 = shifts_1_idxs.max()

    t1 = interpolate(0, half_wf[p1], half_wf[p1 + 1], times[p1], times[p1 + 1])

    shifts_2 = np.diff(half_wf[index_min:] > 0)
    shifts_2_idxs, = np.where(shifts_2 == 1)
    p2 = shifts_2_idxs.min() + index_min

    t2 = interpolate(0, half_wf[p2], half_wf[p2 + 1], times[p2], times[p2 + 1])
    return t2 - t1


def peak_to_trough(wf, times):
    """Calculates minimum-to-maximum
    peak width (peak-to-peak width) for spikes.

    Parameters
    ----------
    wf : array
        Spike waveform
    times : array
        array of times for when the spike waveform is measured

    Returns
    ----------
    peak_to_trough : float
        minimum-to-maximum peak width
    """
    throughs,_ = scipy.signal.find_peaks(-wf)
    index_min = throughs[np.argmin(wf[throughs])]
    peaks,_ = scipy.signal.find_peaks(wf[index_min:])
    index_max = np.min(peaks) + index_min # first peak after through
    return times[index_max] - times[index_min]


def calculate_average_firing_rate(sptrs):
    """Calculates average firing rate for spiketrains.

    Parameters
    ----------
    sptrs : list
        a list of neo spiketrains

    Returns
    ----------
    average_firing_rate : list of floats
        average firing rate (in Hz)
    """
    average_firing_rate = []
    for sptr in sptrs:
        nr_spikes = sptr.waveforms.shape[0]
        dt = sptr.t_stop - sptr.t_start
        rate = (nr_spikes/dt).rescale('Hz').magnitude
        average_firing_rate.append(rate)
    return average_firing_rate


def cluster_waveform_features(feature1, feature2, n_clusters=2):
    """Divides the spiketrains into groups using the k-means algorithm on the
    average waveform of each spiketrain. The average waveform is calculated
    based on feature1 and feature2 of mean spike to the spiketrain.

    Parameters
    ----------
    feature1 : array of floats
        array of floats describing a feature of the spiketrains mean spike
    feature2 : array of floats
        array of floats describing a feature of the spiketrains mean spike
    n_clusters : int
        number of clusters you want, minimum 2

    Returns
    -------
    idx : list of integers
        for example when n_clusters is 2 containts 0s and 1s

    """
    if n_clusters < 2:
        raise ValueError('Number of clusters must be minimum 2')

    features = np.stack(np.array([feature1, feature2]), axis=-1)
    centroids, _ = kmeans(features, n_clusters)
    idx, _ = vq(features, centroids)

    return idx
