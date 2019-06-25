import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq


def calculate_waveform_features(sptrs, calc_all_spikes=False):
    """Calculates waveform features for spiketrains; full-width half-maximum
    (half width) and minimum-to-maximum peak width (peak-to-peak width) for
    mean spike, and average firing rate. If calc_all_spikes is True it also
    calculates half width and peak-to-peak width for all the single spikes in
    each of the spiketrains.

    Parameters
    ----------
    sptrs : list
        a list of neo spiketrains
    calc_all_spikes : bool
        returns half_width_all_spikes and peak_to_peak_all_spikes if True

    Returns
    ----------
    half_width_mean : array of floats
        full-width half-maximum (in ms) for mean spike of each spiketrain
    peak_to_peak_mean : array of floats
        minimum-to-maximum peak width (in ms) for mean spike of each spiketrain
    average_firing_rate : list of floats
        average firing rate (in Hz) for each spiketrain
    half_width_all_spikes : list of arrays with floats
        if calc_all_spikes is True, full-width half-maximum (in ms) for every
        single spike in each spiketrain
    peak_to_peak_all_spikes : list of arrays with floats
        if calc_all_spikes is True, minimum-to-maximum peak width (in ms) for
        every single spike in each spiketrain
    """
    for sptr in sptrs:
        if not hasattr(sptr.waveforms, 'shape'):
            raise AttributeError('Argument provided (sptr) has no attribute\
                                  waveforms.shape')

    average_firing_rate = calculate_average_firing_rate(sptrs)

    stime = np.arange(sptrs[0].waveforms.shape[2], dtype=np.float32) /\
        sptrs[0].sampling_rate
    stime = stime.rescale('ms')

    half_width_list = []
    peak_to_peak_list = []
    for i in range(len(sptrs)):
        mean_wf = np.mean(sptrs[i].waveforms, axis=0)
        max_amplitude_channel = np.argmin(mean_wf.min(axis=1))
        wf = mean_wf[max_amplitude_channel, :]

        half_width_list.append(np.array(half_width(wf, stime)))
        peak_to_peak_list.append(np.array(peak_to_trough(wf)))

    return half_width_mean, peak_to_peak_mean, average_firing_rate


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
    index_min = np.argmin(wf)
    half_amplitude = wf[index_min] * 0.5
    half_wf = np.abs(wf - half_amplitude)
    # there might be multiple intersections, we take the closest to the peak
    p1_idxs, = np.where(half_wf[:index_min] < 1e-5)
    p1 = np.max(p1_idxs)

    p2_idxs, = np.where(half_wf[index_min:] < 1e-5)
    p2 = index_min + np.min(p2_idxs)
    return times[p2] - times[p1]


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
    index_min = np.argmin(wf)
    index_max = index_min + np.argmax(wf[index_min:])
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
        average_firing_rate.append(nr_spikes/dt)
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
