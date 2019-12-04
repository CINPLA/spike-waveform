import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.cluster.vq import kmeans, vq
import quantities as pq


def calculate_waveform_features(sptrs, use_max_channel=True):
    """Calculates waveform features for spiketrains; full-width half-maximum
    (half width) and minimum-to-maximum peak width (peak-to-peak width) for
    mean spike, and average firing rate.

    Parameters
    ----------
    sptrs : list
        a list of neo spiketrains
    use_max_channel : bool
        if True the features are computed on the channel with the largest peak

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

    times = np.arange(
        sptrs[0].waveforms.shape[2], dtype=np.float32) / sptrs[0].sampling_rate

    half_widths = []
    peak_to_throughs = []
    for sptr in sptrs:
        if isinstance(sptr.waveforms, pq.Quantity):
            mean_wf = np.mean(sptr.waveforms, axis=0).magnitude
        else:
            mean_wf = np.mean(sptr.waveforms, axis=0)

        if not use_max_channel:
            half_width_ch = []
            peak_to_through_ch = []
            for ch in range(mean_wf.shape[0]):
                wf = mean_wf[ch, :]
                half_width_ch.append(np.array(half_width(wf, times)))
                peak_to_through_ch.append(np.array(peak_to_trough(wf, times)))
        else:
            # extract max channel
            max_ch = np.unravel_index(np.argmax(np.abs(mean_wf)), mean_wf.shape)[0]
            wf = mean_wf[max_ch, :]
            half_width_ch = np.array(half_width(wf, times))
            peak_to_through_ch = np.array(peak_to_trough(wf, times))
        half_widths.append(half_width_ch)
        peak_to_throughs.append(peak_to_through_ch)

    return np.array(half_widths), np.array(peak_to_throughs), np.array(average_firing_rate)


def calculate_waveform_features_from_template(template, sampling_rate):
    """Calculates waveform features for spiketrains; full-width half-maximum
    (half width) and minimum-to-maximum peak width (peak-to-peak width) for
    mean spike, and average firing rate.

    Parameters
    ----------
    template : array
        mean waveform

    Returns
    ----------
    half_width_mean : array of floats
        full-width half-maximum (in ms) for mean spike of each spiketrain
    peak_to_peak_mean : array of floats
        minimum-to-maximum peak width (in ms) for mean spike of each spiketrain
    """

    times = np.arange(template.shape[1], dtype=np.float32) / sampling_rate

    half_width_ch = []
    peak_to_through_ch = []
    for ch in range(template.shape[0]):
        wf = template[ch, :]
        half_width_ch.append(np.array(half_width(wf, times)))
        peak_to_through_ch.append(np.array(peak_to_trough(wf, times)))

    return np.array(half_width_ch), np.array(peak_to_through_ch)


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
    throughs, _ = find_peaks(-wf)
    if len(throughs) == 0:
        return np.nan
    index_min = throughs[np.argmin(wf[throughs])]
    half_amplitude = wf[index_min] * 0.5
    half_wf = wf - half_amplitude
    # there might be multiple intersections, we take the closest to the through
    shifts_1 = np.diff(half_wf[:index_min] > 0)
    shifts_1_idxs, = np.where(shifts_1 == 1)
    if len(shifts_1_idxs) == 0:
        return np.nan
    p1 = shifts_1_idxs.max()

    t1 = interp1d([half_wf[p1], half_wf[p1 + 1]], [times[p1], times[p1 + 1]])(0)

    shifts_2 = np.diff(half_wf[index_min:] > 0)
    shifts_2_idxs, = np.where(shifts_2 == 1)
    if len(shifts_2_idxs) == 0:
        return np.nan
    p2 = shifts_2_idxs.min() + index_min

    t2 = interp1d([half_wf[p2], half_wf[p2 + 1]], [times[p2], times[p2 + 1]])(0)
    return t2 - t1


def peak_to_trough(wf, times, n_interp=10000):
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
    f = interp1d(times, wf, kind='cubic')
    t = np.linspace(times.min(), times.max(), n_interp)
    wf = f(t)
    throughs, _ = find_peaks(-wf)
    if len(throughs) == 0:
        return np.nan
    index_min = throughs[np.argmin(wf[throughs])]
    peaks, _ = find_peaks(wf[index_min:])
    if len(peaks) == 0:
        return np.nan
    index_max_rel_min = peaks[np.argmax(wf[index_min:][peaks])]
    index_max = index_max_rel_min + index_min  # first peak after through

    return t[index_max] - t[index_min]


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
        rate = (nr_spikes / dt).rescale('Hz').magnitude
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
        for example when n_clusters is 2 containts 0s and 1s,
        sorted according to the sum of means of features

    """
    if n_clusters < 2:
        raise ValueError('Number of clusters must be minimum 2')

    features = np.stack(np.array([feature1, feature2]), axis=-1)
    centroids, _ = kmeans(features, n_clusters)
    idxs, _ = vq(features, centroids)
    ref_idxs = np.unique(idxs)
    vals = []
    for idx in ref_idxs:
        avg1 = np.mean(feature1[idxs == idx])
        avg2 = np.mean(feature2[idxs == idx])
        vals.append(avg1 + avg2)
    new_idxs = ref_idxs[np.argsort(vals)]
    sort_idxs = np.zeros_like(idxs)
    for new_idx, ref_idx in zip(new_idxs, ref_idxs):
        sort_idxs[idxs == ref_idx] = new_idx
    return sort_idxs
