import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
from math import log, floor
from scipy import signal
from scipy.signal import savgol_filter
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from scipy.io import savemat
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import math
import pandas as pd
from scipy.signal import resample
import seaborn as sns
from scipy.fft import fft, fftfreq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
from scipy import signal
from scipy.ndimage import shift
import matplotlib.pyplot as plt
import sys
import pywt
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from multiprocessing import Pool
import time
from multiprocessing import Pool
from tqdm.autonotebook import tqdm
import ex_features
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import moving_avg
# from tsfresh import extract_features
# from tsfresh.utilities.dataframe_functions import make_forecasting_frame

def anova_func(features):

    features_arr = ['MIN_f',
            'MAX_f','MEAN_f','VAR_f','SKEW_f','KURTOSIS_f']
    anova_results = {}

    for feature in features_arr:
        # Prepare the formula for the ANOVA model
        formula = f'{feature} ~ C(Group)'
        
        # Perform ANOVA
        model = ols(formula, data=features).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Store results in dictionary
        anova_results[feature] = anova_table
    
        # Optionally print the results
        print(f"ANOVA results for {feature}:")
        print(anova_table)
        print("\n")



def _log_n(min_n, max_n, factor):
    """
    Creates a list of integer values by successively multiplying a minimum
    value min_n by a factor > 1 until a maximum value max_n is reached.

    Used for detrended fluctuation analysis (DFA).

    Function taken from the nolds python package
    (https://github.com/CSchoel/nolds) by Christopher Scholzel.

    Parameters
    ----------
    min_n (float):
        minimum value (must be < max_n)
    max_n (float):
        maximum value (must be > min_n)
    factor (float):
       factor used to increase min_n (must be > 1)

    Returns
    -------
    list of integers:
        min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
        without duplicates
    """
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)



def pop_events_extract(onsets, ends, pop_ca_data, ca_trace):

    """
    Extract events from the ca pop signal using events' onsets and ends 

    """
    #main_ROI_trace = signal.resample(main_ROI_trace, vm_len)
    events_comb = []
    k = 0 
    len_met = 0
    
    for exp, on_co, end_co in zip(ca_trace, onsets, ends):
        fs = len(exp)/100 
        #fs = 10000
        window_size = int(10 * fs)

        start_pt = int(window_size/2)
        end_pt = int(len(exp) - (window_size/2))
        exp = exp[start_pt:end_pt]

        ca_pop = pop_ca_data[k] 
        #print("ca_pop shape:", ca_pop.shape)
        events_neu_per_net = []
        for j, neu_exp in enumerate(ca_pop): #neu_exp is each neuron in a network/recording
            
            # print("neu_exp shape:", neu_exp.shape)
            # print("exp shape:", exp.shape)
            events = []
            for i in range(len(on_co)): #iterating through ons, ends for each neuron in the network
                on = on_co[i]
                end = end_co[i]
                if math.isnan(on) or math.isnan(end):
                    continue
       
                adj = int(0.5*fs)
    
                if on > adj:
                    on_2 = on - adj
                else:
                    on_2 = on 
                    
                if len(exp) > (end + adj):
                    end_2 = end + adj
                else:
                    end_2 = end 
                    
               #main_ROI_trace = signal.resample(main_ROI_trace, vm_len)
                event = signal.resample(neu_exp, len(exp)) 
                event_2 = event[on_2:end_2]  #event i from n events for each neuron in network 
    
                if len(event) > 0:
                        events.append(event_2)
         
            _, idx = np.unique([arr.tobytes() for arr in events], return_index=True)
            events_uni = [events[i] for i in idx]
            if j == 0:
                len_met = len_met + len(events_uni)
            
            events_neu_per_net.append(events_uni) #all events for a neuron in network added, 
            #final length should be 10 as there are 10 neurons for each network   
            print("len:events_neu_per_net ", len(events_neu_per_net))
            
        print("len:events_neu_per_net ", len(events_neu_per_net))
        df = pd.DataFrame(events_neu_per_net)
        df_array = df.values
        np.savez(f'/Users/bhavikagopalani/classifier/net_events/network_events_{k:04d}.npz', df_array=df_array)
    
        #events_comb.append(events_neu_per_net)
        k = k + 1 
        print("k = ", k)
    print("len_mat = ", len_met)
   



    

def graph_features(main_dir):
    #edge density 
    #average centrality 
    #average clustering coefficient 
    #average correlated pair ratio

    
    folder_loc = f"{main_dir}/Onset-End[0.7s]_[4.5s]/ca_pop/ca_events"
    
    items = os.listdir(folder_loc)
    sorted_files = sorted(items)
    counter = 0
    features_comb = []
    
    for file in sorted_files:
        if file.endswith('.csv'):
            print(file)
            filepath = f"{folder_loc}/{file}"
            df = pd.read_csv(f'{filepath}', header=1)
            data = df.to_numpy()
            if data.shape[1]< 40000 and data.shape[1] > 1000:  
                cg = CaGraph(data= filepath, threshold = 0.2)
                cg_graph = cg.get_graph()
                density = cg.analysis.get_density()
                path = cg.analysis.get_shortest_path_length()
                avg_centrality = np.mean(cg.analysis.get_betweenness_centrality())
                avg_clustering_coeff = np.mean(cg.analysis.get_clustering_coefficient())
                avg_corr_pair_ratio = np.mean(cg.analysis.get_correlated_pair_ratio())
                #avg_eg_centr = np.mean(cg.analysis.get_eigenvector_centrality())
                features = [density, avg_centrality,avg_clustering_coeff,avg_corr_pair_ratio]
                features_comb.append(features)

    return features_comb

def dbscan_clus(features):
    
    #data = StandardScaler().fit_transform(features)
   clustering = DBSCAN(eps=0.5, min_samples=50).fit(features)
   labels = clustering.labels_

   # unique_labels = set(labels)
   # signals_dict = {label: [] for label in unique_labels}
   # print(labels)
        
   # for labels, events in signals_dict.items():
   #      print(f"Label {labels}, N= {len(events)}")
   frequency = Counter(labels)
   print(frequency)

    


def generate_ca_pop_data(main_dir):


    
    folder_loc = f"{main_dir}/Data"

    items = os.listdir(folder_loc)
    sorted_files = sorted(items)
    pop_ca_data = []

    for file, i in zip(sorted_files, range(0, len(sorted_files))):
        if file.endswith('.npz'):
            filepath = f"{folder_loc}/{file}"
            
            with np.load(filepath, allow_pickle=True) as data:  
 
                if data['vm'].shape[0]>0: 
                    nortraces = data['norTraces'] 
                    ROI_no = data['ROI_no']
                    
                    if nortraces.shape[0] > ROI_no:
                        #vm = data['vm']
                        main_ROI_trace = nortraces[ROI_no]
                        has_nan = any(math.isnan(x) for x in main_ROI_trace if isinstance(x, float))
                        if has_nan:
                            continue 
                        # main_ROI_trace = signal.resample(main_ROI_trace, 855000)
                        # ca_trace.append(main_ROI_trace)
                        # vm_full.append(vm[:855000])
                        num_ROI = nortraces.shape[0]
                        pop_traces = []
                        pop_traces.append(main_ROI_trace)
                        for i in range(0, num_ROI):
                            if i == ROI_no:
                                continue 
                            pop_traces.append(nortraces[i])
                        pop_traces = np.array(pop_traces)

                    correlations = np.array([np.corrcoef(pop_traces[0], pop_traces[i])[0, 1] for i in range(len(pop_traces))])
                    top_10_indices = np.argsort(correlations)[-11:]
                    top_10_arrays = pop_traces[top_10_indices] #top 10 most functionally correlated neurons and the main ROI neuron
                    pop_ca_data.append(top_10_arrays)
                    
                    
    return pop_ca_data

def event_data_load(ca, vm):

    ca_div = []
    vm_div = []
    for ca_ev, vm_ev in zip(ca, vm):
        num = len(ca_ev)//2000
        for i in range(0, num):
            start = i*2000
            end = (i+1)*2000
            ca_div.append(np.array(ca_ev[start:end]))
            vm_div.append(np.array(vm_ev[start:end]))
            if not len(vm_ev[start:end])==1000:
                print(len(vm_ev[start:end]))
    
    
    return vm_div, ca_div

def make_df(events):
    
    events_df = []
    event_t = []
    event_id = []
    signal_list = []
    df_stacked = pd.DataFrame(columns=['id', 'time', 'signal'])
    
    for i in range(len(events)):
        
        event_len = len(events[i])
        event_t = np.linspace(1, event_len, event_len)
        event_id = np.ones_like(event_t)*(i+1)
        df_temp = pd.DataFrame({'id': event_id, 'time': event_t, 'signal': events[i]})
        df_stacked = pd.concat([df_stacked, df_temp], axis=0, ignore_index=True)
        
    return df_stacked
    
    
def extract_features_org(events):

    """

    Extracts features from the set of signals 

    #FEATURES = ['MAX','VAR','STD','P2P','SKEW','KURTOSIS', 'MIN_f',
            'MAX_f','MEAN_f','VAR_f','SKEW_f','KURTOSIS_f', 'Higuchi Fractal Dimension', 'Detrended Fluctuation Analysis (DFA)',  
              'Mean of absolute values of second differences (normalized signal)', 'Mean (DWT coeffs)', 'Power (DWT coeffs)', 'STD (DWT coeffs)', 'Skew (DWT coeffs)'
            , 'Kurtosis (DWT coeffs)']
    """

    

    features_comb = []

    for signal in events:
        if len(signal)>0:
            ft = fft(signal) ##The resulting array ft contains 
            ##complex numbers that represent the amplitude and phase of each frequency component present in the original signal.
            
            S = np.abs(ft**2)/len(signal) ##This line calculates the power spectrum of the signal, 
            ##which is a way to represent the distribution of power into frequency components making up the signal.
            
            normalized_signal = (signal - np.mean(signal)) / np.std(signal)
            
            wavelet_name = 'db4'  # Daubechies wavelet
            wavelet = pywt.Wavelet(wavelet_name)
            coeff = pywt.wavedec(signal, wavelet, mode='symmetric')
            coeff = np.concatenate(coeff)
            
            features = [np.max(signal),
            np.var(signal), np.std(signal), np.ptp(signal), 
            stats.skew(signal), stats.kurtosis(signal),
            np.min(S), np.max(S), np.mean(S), np.var(S), stats.skew(S), stats.kurtosis(S), _higuchi_fd(signal), _dfa(signal),
            np.mean(np.abs(np.diff(normalized_signal, n=2))), 
            np.mean(coeff), np.mean(coeff**2), np.var(coeff)
            , stats.skew(coeff), stats.kurtosis(coeff)]
            
            features_comb.append(features)

    return features_comb

def extract_features_time(events):

    """

    Extracts features from the set of signals 

    #FEATURES = ['MAX','VAR','STD','P2P','SKEW','KURTOSIS','Higuchi Fractal Dimension', 'Detrended Fluctuation Analysis (DFA)',  
              'Mean of absolute values of second differences (normalized signal)']
    """

    features_comb = []

    for signal in events:
        if len(signal)>0:
            ft = fft(signal) ##The resulting array ft contains 
            ##complex numbers that represent the amplitude and phase of each frequency component present in the original signal.
            
            S = np.abs(ft**2)/len(signal) ##This line calculates the power spectrum of the signal, 
            ##which is a way to represent the distribution of power into frequency components making up the signal.
            
            normalized_signal = (signal - np.mean(signal)) / np.std(signal)
            
            wavelet_name = 'db4'  # Daubechies wavelet
            wavelet = pywt.Wavelet(wavelet_name)
            coeff = pywt.wavedec(signal, wavelet, mode='symmetric')
            coeff = np.concatenate(coeff)
            features = [np.max(signal),
                        np.var(signal), np.std(signal), np.ptp(signal), 
                        stats.skew(signal), stats.kurtosis(signal),
                        _higuchi_fd(signal), _dfa(signal),
                       np.mean(np.abs(np.diff(normalized_signal, n=2)))]
            
            features_comb.append(features)

    return features_comb

def add_gaussian_noise(signal, sigma=0.1):
    """
    Add Gaussian noise to a signal.
    
    Parameters:
    - signal: numpy array representing the signal.
    - sigma: Standard deviation of the Gaussian noise.
    
    Returns:
    - Noisy signal.
    """
    signal = np.array(signal)
    noise = np.random.normal(0, sigma, signal.shape)
    return signal + noise

def add_salt_and_pepper_noise(signal, amount=0.05, salt_vs_pepper=0.5):
    """
    Add salt-and-pepper noise to a signal.
    
    Parameters:
    - signal: numpy array representing the signal.
    - amount: Fraction of total pixels to alter.
    - salt_vs_pepper: Proportion of salt noise vs. pepper noise.
    
    Returns:
    - Noisy signal.
    """
    signal = np.array(signal)
    noisy_signal = np.copy(signal)
    num_pixels = signal.size
    
    # Salt noise (set to maximum value)
    num_salt = int(np.ceil(amount * num_pixels * salt_vs_pepper))
    salt_coords = tuple(
        np.random.randint(0, i, num_salt) for i in signal.shape
    )
    noisy_signal[salt_coords] = signal.max()
    
    # Pepper noise (set to minimum value)
    num_pepper = int(np.ceil(amount * num_pixels * (1.0 - salt_vs_pepper)))
    pepper_coords = tuple(
        np.random.randint(0, i, num_pepper) for i in signal.shape
    )
    noisy_signal[pepper_coords] = signal.min()
    
    return noisy_signal

def add_poisson_noise(signal):
    """
    Add Poisson noise to a signal.
    
    Parameters:
    - signal: numpy array representing the signal.
    
    Returns:
    - Noisy signal.
    """
    # Poisson noise requires non-negative values; ensure signal is scaled appropriately.
    # Here we scale the signal to a higher range to emulate counts, then scale back.
    signal = np.array(signal)
    vals = len(np.unique(signal))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(signal * vals) / float(vals)
    return noisy

def add_noise_to_signals(signals, noise_type='gaussian', **kwargs):
    """
    Apply specified noise to a list of signals.
    
    Parameters:
    - signals: List of numpy arrays.
    - noise_type: Type of noise ('gaussian', 'salt_and_pepper', or 'poisson').
    - kwargs: Additional parameters for noise functions (e.g., sigma, amount).
    
    Returns:
    - List of noisy signals.
    """
    
    noisy_nested_signals = []
    for ev in signals:
        noisy_signals = []
        for signal in ev:
            if noise_type == 'gaussian':
                sigma = kwargs.get('sigma', 0.1) 
                noisy = add_gaussian_noise(signal, sigma)
            elif noise_type == 'salt_and_pepper':
                amount = kwargs.get('amount', 0.05)
                salt_vs_pepper = kwargs.get('salt_vs_pepper', 0.5)
                noisy = add_salt_and_pepper_noise(signal, amount, salt_vs_pepper)
            elif noise_type == 'poisson':
                noisy = add_poisson_noise(signal)
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
            noisy_signals.append(noisy)
        noisy_nested_signals.append(noisy_signals)

    return noisy_nested_signals

def extract_features_freq(events):

    """

    Extracts features from the set of signals 

    # old FEATURES = ['MIN_f','MAX_f','MEAN_f','VAR_f','SKEW_f','KURTOSIS_f', 'min (DWT)', 'max (DWT)', 
              'Mean (DWT coeffs)', 'Power (DWT coeffs)', 'STD (DWT coeffs)', 'Skew (DWT coeffs)'
            , 'Kurtosis (DWT coeffs)']
    # new = ['MIN_f','MAX_f','MEAN_f','VAR_f','SKEW_f','KURTOSIS_f', 
              'Mean (DWT coeffs)', 'Power (DWT coeffs)', 'STD (DWT coeffs)', 'Skew (DWT coeffs)'
            , 'Kurtosis (DWT coeffs)']
    """

    

    features_comb = []
    for ev in events:
        for signal in ev:
            if len(signal)>0:
                ft = fft(signal) ##The resulting array ft contains 
                ##complex numbers that represent the amplitude and phase of each frequency component present in the original signal.
                #fs = 10000
                #f, S = welch(signal, fs, nperseg=10)
                S = np.abs(ft**2)/len(signal) ##This line calculates the power spectrum of the signal, 
                ##which is a way to represent the distribution of power into frequency components making up the signal.
                
                normalized_signal = (signal - np.mean(signal)) / np.std(signal)
                
                wavelet_name = 'db4'  # Daubechies wavelet
                wavelet = pywt.Wavelet(wavelet_name)
                coeff = pywt.wavedec(signal, wavelet, mode='symmetric')
                coeff = np.concatenate(coeff)
                features = [np.min(S), np.max(S), np.mean(S), np.var(S), stats.skew(S), stats.kurtosis(S), np.min(coeff), np.max(coeff),
                            np.mean(coeff), np.mean(coeff**2), 
                           np.std(coeff), stats.skew(coeff), stats.kurtosis(coeff)]
                # features = [np.max(S), np.mean(S), stats.skew(S), stats.kurtosis(S), 
                #     np.mean(coeff), np.mean(coeff**2), 
                #     np.std(coeff), stats.skew(coeff), stats.kurtosis(coeff)]
                
                features_comb.append(features)

    return features_comb

def ap_detect(events):

    ap = []

    for sig in events:
        for ev in sig:
            ap_list = []
    
            # zero_crossings = np.where(np.diff(np.sign(ev)))[0]
           
            # peaks, _ = find_peaks(ev)
            fs = 8550
            #fs = len(exp)/100
            
            thresh_min = -10                    # Min threshold to detect spikes
            thresh_prominence = 15              # Min spike amplitude  
            thresh_min_width = 0.5 * (fs/1000) # Min required width in ms
            thresh_max_width = 2 * (fs/1000)# Max required width in ms
            distance_min = 1 * (fs/1000)        # Min horizontal distance between peaks
            pretrigger_window = (1.5 * fs)/1000
            posttrigger_window = (2 * fs)/1000
             
            # Find peaks function
            peaks, peaks_dict = find_peaks(ev, 
                       height=thresh_min, 
                       threshold=thresh_min,  
                       distance=distance_min,  
                       prominence=thresh_prominence,  
                       width=thresh_min_width,
                       wlen=None,       # Window length to calculate prominence
                       rel_height=0.5,  # Relative height at which the peak width is measured
                       plateau_size=None)
            
            # Select peaks that are at zero crossings
            #zero_crossing_peaks = [zc for zc in zero_crossings if zc in peaks]
            ap.append(peaks)
        
    return ap 
        

    

def extract_features_par(signal):

    """

    Extracts Amplitude, Min, Mean, Variance, STD, RMS, Skew, Kurtosis from the set of signals 

    #FEATURES = ['MAX','VAR','STD','P2P','SKEW','KURTOSIS', 'MIN_f',
            'MAX_f','MEAN_f','VAR_f','SKEW_f','KURTOSIS_f', 'Higuchi Fractal Dimension', 'Detrended Fluctuation Analysis (DFA)',  
              'Mean of absolute values of second differences (normalized signal)', 'Mean (DWT coeffs)', 'Power (DWT coeffs)', 'STD (DWT coeffs)', 'Skew (DWT coeffs)'
            , 'Kurtosis (DWT coeffs)']
    """

    if len(signal)<0:
        return
    features_comb = []

   
    if len(signal)>0:
        ft = fft(signal) ##The resulting array ft contains 
        ##complex numbers that represent the amplitude and phase of each frequency component present in the original signal.
        
        S = np.abs(ft**2)/len(signal) ##This line calculates the power spectrum of the signal, 
        ##which is a way to represent the distribution of power into frequency components making up the signal.
        
        normalized_signal = (signal - np.mean(signal)) / np.std(signal)
        
        wavelet_name = 'db4'  # Daubechies wavelet
        wavelet = pywt.Wavelet(wavelet_name)
        coeff = pywt.wavedec(signal, wavelet, mode='symmetric')
        coeff = np.concatenate(coeff)
        features = [np.max(signal),
                    np.var(signal), np.std(signal), np.ptp(signal), 
                    stats.skew(signal), stats.kurtosis(signal),
                   np.min(S), np.max(S), np.mean(S), np.var(S), stats.skew(S), stats.kurtosis(S), _higuchi_fd(signal), _dfa(signal),
                   np.mean(np.abs(np.diff(normalized_signal, n=2))), 
                    np.mean(coeff), np.mean(coeff**2), 
                   np.std(coeff), stats.skew(coeff), stats.kurtosis(coeff)]
        
        features_comb.append(features)

    return features_comb

def parallel_feature_computation(events):
    
    num_processes = 4  # Adjust this to the number of cores or logical processors available
    with Pool() as pool:
        results = pool.map(extract_features_par, events)

    return results

def _higuchi_fd(x):
    """Utility function for `higuchi_fd`.
    """
    
#      -----
#     Original code from the `mne-features <https://mne.tools/mne-features/>`_
#     package by Jean-Baptiste Schiratti and Alexandre Gramfort.

#     This function uses Numba to speed up the computation.

#     References
#     ----------
#     Higuchi, Tomoyuki. "Approach to an irregular time series on the
#     basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
#     (1988): 277-283.
        
#     if not len(x)/2 ==0:
#         kmax = int(len(x)//2)
#     else:
#         kmax = int(len(x)//2) - 1
    kmax = 50
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
#             print(m, k)
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
#             print(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi

def _dfa(x):
    """
    Utility function for detrended fluctuation analysis
    """
    
#     The code is a faster (Numba) adaptation of the original code by Christopher
#     Scholzel.

#     References
#     ----------
#     * C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,
#       H. E. Stanley, and A. L. Goldberger, “Mosaic organization of
#       DNA nucleotides,” Physical Review E, vol. 49, no. 2, 1994.

#     * R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,
#       V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,
#       “Detrended fluctuation analysis: A scale-free view on neuronal
#       oscillations,” Frontiers in Physiology, vol. 30, 2012.

    N = len(x)
    nvals = _log_n(4, 0.1 * N, 1.2)
    walk = np.cumsum(x - x.mean())
    fluctuations = np.zeros(len(nvals))

    for i_n, n in enumerate(nvals):
        d = np.reshape(walk[:N - (N % n)], (N // n, n))
        ran_n = np.array([float(na) for na in range(n)])
        d_len = len(d)
        trend = np.empty((d_len, ran_n.size))
        for i in range(d_len):
            slope, intercept = _linear_regression(ran_n, d[i])
            trend[i, :] = intercept + slope * ran_n
        # Calculate root mean squares of walks in d around trend
        # Note that np.mean on specific axis is not supported by Numba
        flucs = np.sum((d - trend) ** 2, axis=1) / n
        # https://github.com/neuropsychology/NeuroKit/issues/206
        fluctuations[i_n] = np.sqrt(np.mean(flucs))

    # Filter zero
    nonzero = np.nonzero(fluctuations)[0]
    fluctuations = fluctuations[nonzero]
    nvals = nvals[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        dfa = np.nan
    else:
        dfa, _ = _linear_regression(np.log(nvals), np.log(fluctuations))
    return dfa


def _linear_regression(x, y):
    """Fast linear regression using Numba.

    Parameters
    ----------
    x, y : ndarray, shape (n_times,)
        Variables

    Returns
    -------
    slope : float
        Slope of 1D least-square regression.
    intercept : float
        Intercept
    """
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept
       

def data_load(main_dir):

    """
    Loads data from the main directory with the .npz files
    
    For a file named: "02022022_1_2_0pA6calcium traces TOTAL.npz" (for example), 02022022 is second February 2022, 1_2 is sample/session, 
    0pA is no driving current applied (spontaneous activity) 

    """

    folder_loc = main_dir
    ca_trace = []
    vm_full = []

    items = os.listdir(folder_loc)
    sorted_files = sorted(items)

    for file, i in zip(sorted_files, range(0, len(sorted_files))):
        if file.endswith('.npz'):
            filepath = f"{folder_loc}/{file}"

            with np.load(filepath, allow_pickle=True) as data:  
 
                if data['vm'].shape[0]>0: 
                    nortraces = data['norTraces'] 
                    ROI_no = data['ROI_no']
                    if nortraces.shape[0] > ROI_no:
                        vm = data['vm']
                        main_ROI_trace = nortraces[ROI_no]
                        has_nan = any(math.isnan(x) for x in main_ROI_trace if isinstance(x, float))
                        if has_nan:
                            continue 
                        vm_len = len(vm)
                        main_ROI_trace = signal.resample(main_ROI_trace, vm_len)
                        ca_trace.append(main_ROI_trace)
                        vm_full.append(vm)
    

    return vm_full, ca_trace

def moving_avg_subtraction(smooth_vm):

    """
    Computes moving average over a window size, subtracts from original signal and 
    returns subtracted signal with boundaries (affected by subtraction) removed.
    
    """
    
    # fs = 8550 #Assuming approximate sampling rate of 8550 Hz or 8.55 kHz
    # window_size = 10 * fs  # 10 seconds window in samples

    avg_subtracted = [] 

    for exp, i in zip(smooth_vm, range(0, len(smooth_vm))):
        fs = len(exp)/100
    
        window_size = int(10 * fs)
        moving_avg = np.convolve(exp, np.ones(window_size)/window_size, mode='valid') 

        padded_moving_avg = np.pad(moving_avg, (window_size//2, len(exp) - len(moving_avg) - (window_size//2)), mode='constant', constant_values= 0)
      
        signal_corrected = exp - padded_moving_avg 
    
        #To remove the part of the signal that has boundary effects from moving avg subtraction
        #If window size is 10, then a full overlap of the window only occurs at the half of the 
        # lenght of the window from both the start and end of the signal

        start_pt = int(window_size/2) 
        end_pt = int(len(exp) - (window_size/2))
        avg_subtracted.append(np.array(signal_corrected[start_pt:end_pt]))
     
        #t = np.linspace(0, 90,855000)
        
        #To visualise the corrected signal 

        # plt.figure(figsize=(20, 6))
        # plt.plot(t[start_pt:end_pt], signal_corrected[start_pt:end_pt])
        # plt.title(f"Average Subtracted, {i}")
        
        # #plt.savefig(f"/Users/bhavikagopalani/Downloads/Npz files/plots/avg_subtracted/{i}.png")
        # plt.close()
       


    return avg_subtracted


def onset_end_func(avg_subtracted, peaks_comb):

    """
    Onset and end detection from moving avg subtracted signal using zero crossing 

    """
  
    onsets_comb = []
    ends_comb = []

   

    for exp, peaks, i in zip(avg_subtracted, peaks_comb, range(0, len(avg_subtracted))):
        zero_crossings = find_zero_crossings(exp)

        t = np.linspace(1, len(exp), len(exp))

        onsets = []
        ends = []
        temp = []

    
        for peak in peaks:


            onset_candidates = zero_crossings[zero_crossings < peak]

            if onset_candidates.size > 0:
            
                onset = onset_candidates[-1] + 1  
            else:
                onset = np.nan 
                #print(peak)

            end_candidates = zero_crossings[zero_crossings > peak]
            if end_candidates.size > 0:
            
                end = end_candidates[0] + 1  
            else:
                end = np.nan 
                #print(peak)

            
            # if onset> 8550:
            #     onset = onset - 6000
            
            # if end < 760000:
            #     end = end + 6000
            
            onsets.append(onset)
            ends.append(end)


        ##To visualise the onsets and ends overlayed on the signal 

        # t = np.linspace(0, 90, len(exp))
        # plt.figure(figsize=(20, 6))
        # plt.plot(t, exp)
        # plt.title(f"Onsets and Ends, {i}")
        
        # for on, end in zip(onsets, ends):
            
        #     if math.isnan(on) or math.isnan(end):
        #         continue

        #     plt.scatter(t[on], exp[on], color='green', marker='o')
        #     plt.scatter(t[end], exp[end], color='red', marker='o')

        # #plt.savefig(f"/Users/bhavikagopalani/Downloads/Npz files/plots/on_ends/{i}.png")
        # plt.close()

        onsets_comb.append(onsets)
        ends_comb.append(ends)


    return onsets_comb, ends_comb 


def smoothening_filter(vm_trace):

    """
    Apply Savitzky-Golay filter on original signal to smoothen it 

    """

    smooth_vm = []
    baseline_subtrated = []
    for vm, i in zip(vm_trace, range(0, len(vm_trace))):

        #Assuming signal frequency = vm_len/100
        fs = len(vm)/100 #sampling frequency
        #fs = 10000
        window_length = int(fs*0.05)  # 200 ms window length= fs*0.2
        polyorder = 2
        smoothed_signal = savgol_filter(vm, window_length, polyorder)

        smooth_vm.append(smoothed_signal)
        bs_ = vm - smoothed_signal
        baseline_subtrated.append(bs_)

        ##To visualise the smoothened signal 

        t = np.linspace(1, 100, len(vm))
        plt.figure(figsize=(20, 6))
        plt.plot(t, vm, color = 'black')
        plt.plot(t, smoothed_signal, color = 'blue', linewidth=3)
        plt.title("Original Signal")
        plt.savefig(f"/Users/bhavikagopalani/classifier/Basic/smooth_v_raw_2/{i}.png")
        #plt.show()
        plt.close()



    return smooth_vm, baseline_subtrated

def peak_sort(smooth_vm):


    """

    Find peaks in the smooth signal 

    """

    peaks_comb = []
 
    for exp in smooth_vm:
        # fs = 8550 
        # window_size = 10 * fs
        fs = len(exp)/100 
        #fs = 10000
        window_size = int(10 * fs)
        start_pt = int(window_size/2)
        end_pt = int(len(exp) - (window_size/2))

        exp = exp[start_pt:end_pt]
        w = fs*0.008
        #peaks, _ = find_peaks(exp, prominence= 8, width=w) #width= 8ms;  fs*0.008 time points
        min_distance_samples = int(0.005 * fs)
        min_width_samples = int(0.008 * fs)

        peaks, _ = find_peaks(exp,height=(-52, -20), width=min_width_samples, distance=min_distance_samples)

        dVm = np.diff(exp) * fs
        slope_threshold = 5000  # 5 V/s (adjust as needed)

        valid_peaks = []
        for p in peaks:
            # Look at slope just before the peak
            if p > 0:
                if dVm[p-1] < slope_threshold:
                    valid_peaks.append(p)
            else:
                valid_peaks.append(p)

        peaks_comb.append(valid_peaks)
    
    
    return peaks_comb

def plot_peaks(peaks_comb, smooth_vm):

    for exp, peak in zip(smooth_vm, peaks_comb):

        x = np.zeros_like(exp)
        x[peak] = 1 
        x[x==0] = np.nan
        t = np.linspace(0, 100, 855000)
        plt.scatter(t, x, marker='|')
        plt.ylim = (0.5, 1)
        #plt.show()
        
def find_zero_crossings(signal):

    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    #print(zero_crossings)
    return zero_crossings

        
def merge_events(onsets_comb, ends_comb):

    """
    Try function #1 to merge events 
    """

    threshold = 4275 #0.5*8550 // 500ms

    merged_onsets_comb = []
    merged_ends_comb = []

    for onsets, ends in zip(onsets_comb, ends_comb):
        merged_ends = []
        merged_onsets = [onsets[0]]
        for i in range(1, len(onsets)):

            gap = onsets[i] - ends[i-1]

            if gap > threshold:
               merged_onsets.append(onsets[i])
               merged_ends.append(ends[i-1])
                
            else:
                merged_ends.append(ends[i])
                merged_onsets.append(onsets[i-1])

        merged_onsets_comb.append(merged_onsets)
        merged_ends_comb.append(merged_ends)
    
    return merged_onsets_comb, merged_ends_comb

def merge_events_2(onsets_comb, ends_comb):

    """
    Try function #2 to merge events 
    """

    threshold = 4275 #0.5*8550 // 500ms

    merged_onsets_comb = []
    merged_ends_comb = []

    for onsets, ends in zip(onsets_comb, ends_comb):
        m_onsets = []
        m_ends = []
        diff_list = []
        

        for i in range(1, len(onsets)):
            if onsets[i] == np.nan or ends[i-1] == np.nan:
                diff_list.append(np.nan)
                continue
            diff = onsets[i] - ends[i-1]
            diff_list.append(diff)

        diff_list = np.array(diff_list)
        
        print(diff_list/8550)
        

def merge_events_3(onsets, ends, smooth_vm):

    """

    Main, working function to merge events 

    """
    

    merged_events_comb = []
    for i in range(len(onsets)):

        events = list(zip(onsets[i], ends[i]))
        
        events.sort(key=lambda x: x[0]) #Sort events by start time
        
        merged_events = []

        fs = len(smooth_vm[i])/100
        #fs = 10000
        threshold = int(0.5*fs)

        for current_start, current_end in events:
            #If merged_events is not empty and the current event is within the threshold
            if merged_events and current_start - merged_events[-1][1] <= threshold: #and (current_end - current_start) < 25000:
             
                merged_events[-1] = (merged_events[-1][0], max(merged_events[-1][1], current_end))
            else:
                merged_events.append((current_start, current_end))
        merged_events_comb.append(merged_events)
            
    return merged_events_comb


def extract_events_dep(smooth_vm, merged_events_comb):

    """
    Extract events from the smooth signal using merged events' onsets and ends 

    """

    #fs = 8550 
    fs = len(exp)/100
    #fs = 10000
    window_size = 10 * fs


    for exp in smooth_vm:

        start_pt = int(window_size/2)
        end_pt = int(len(exp) - (window_size/2))
        exp = exp[start_pt:end_pt]
        events = []
        for events_exp in merged_events_comb:
            for i in range(len(events_exp)):

                on = events_exp[i][0]
                end = events_exp[i][1]
                if math.isnan(on) or math.isnan(end):
                    continue
                event = exp[on:end]
                events.append(event)

    return events 

def extract_events(smooth_vm, merged_events_comb):

    """
    Extract events from the smooth signal using merged events' onsets and ends 

    """

    # fs = 8550 
    # window_size = 10 * fs

    events = []
    for exp, events_exp in zip(smooth_vm, merged_events_comb):
        fs = len(exp)/100 
        #fs = 10000
        window_size = int(10 * fs)

        start_pt = int(window_size/2)
        end_pt = int(len(exp) - (window_size/2))
        exp = exp[start_pt:end_pt]

    
    
        for i in range(len(events_exp)):
            on = events_exp[i][0] 
            end = events_exp[i][1]
            if math.isnan(on) or math.isnan(end):
                continue
                
            if len(exp[on:end]) < 1500:
                adj = int(0.5*fs)
                on = events_exp[i][0] - adj
                end = events_exp[i][1] + adj
                
            event = exp[on:end]
            # if len(event)>40000:
            #     continue

            if len(event)> 0:
                events.append(event)

    return events 

def extract_events_unmerged(smooth_vm, onsets, ends, type):

    """
    Extract events from the smooth signal using merged events' onsets and ends 

    """

    # fs = 8550 
    # window_size = 10 * fs

    
    events_comb = []
    for exp, on_co, end_co, p in zip(smooth_vm, onsets, ends, range(len(smooth_vm))):
        fs = len(exp)/100 
        #fs = 10000
        window_size = int(10 * fs)

        start_pt = int(window_size/2)
        end_pt = int(len(exp) - (window_size/2))
        exp = exp[start_pt:end_pt]
            
        events = []
        for i in range(len(on_co)):
            on = on_co[i]
            end = end_co[i]
            if np.isnan(on) or np.isnan(end):
                continue
   
            adj = int(0.5*fs)

            if on > adj:
                on_2 = on - adj
            else:
                on_2 = on 
                
            if len(exp) > (end + adj):
                end_2 = end + adj
            else:
                end_2 = end 
            
            #print(on, end)
            on = int(on)
            end = int(end)
            on_2 = int(on_2)
            end_2 = int(end_2)
            event = exp[on:end]
            event_2 = exp[on_2:end_2]
            # if len(event)>40000:
            #     continue

            if len(event) > 0:
                if type == 'vm':
                    events.append(event)
                if type == 'ca':
                    events.append(event_2)
                    
            if len(events) == 0:
                print(f"No events in {p}")
                    
        events_comb.append(events)

    return events_comb


def extract_events_viz(smooth_vm, merged_events_comb):

    """
    Extract events from the signal using merged events' onsets and ends 

    """

    # fs = 8550 
    # window_size = 10 * fs

    events = []
    counter = 0
    for exp, events_exp in zip(smooth_vm, merged_events_comb):
        fs = len(exp)/100 
        #fs = 10000
        window_size = int(10 * fs)

        start_pt = int(window_size/2)
        end_pt = int(len(exp) - (window_size/2))
        exp = exp[start_pt:end_pt]

        adj = int(0.5*fs)
    
        for i in range(1, len(events_exp)):
            
            on = events_exp[i][0] - adj
            end = events_exp[i][1] + adj
            if math.isnan(on) or math.isnan(end):
                continue
            event = exp[on:end]
            # if len(event)>40000:
            #     continue
                
            events.append(event)
            counter = counter + 1 
            t = np.linspace(0, len(event),len(event))
            plt.figure(figsize=(20, 15))
            plt.plot(t, event)
            plt.axvline(x=adj, color='green', linestyle='-', linewidth=1)
            plt.axvline(x= len(event) - adj, color='red', linestyle='-', linewidth=1)
            plt.savefig(f"/Users/bhavikagopalani/classifier/Basic/events/{counter}.png")
            plt.close()

    return events 

def svd_apply(events):

    """
    Apply SVD on events 
    """

    max_length = max(signal.size for signal in events)

    padded_signals = np.array([np.pad(signal, (0, max_length - signal.size), 'constant') for signal in events])

    svd = TruncatedSVD(n_components=100)

    U = svd.fit_transform(padded_signals)
    Sigma = svd.singular_values_
    VT = svd.components_
    
    return U, Sigma, VT


def perform_kmeans(U, n):

    """
    Perform K-means on features 
    """
    
    kmeans = KMeans(n_clusters=n, random_state=42, n_init="auto")
    kmeans.fit(U)
    labels = kmeans.labels_

    return labels
       
def k_means_analysis(U, dir):
    
    score = []
    for i in range(2, 11):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init="auto")
        kmeans.fit(U)
        labels = kmeans.labels_
        score.append(silhouette_score(U, labels))
    
    plt.plot(range(2, 11), score)
    plt.title("Silhouette Plot")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    #plt.savefig(dir)
            
    plt.show()
    
    
def cluster_analysis(events):
    
    range_n_clusters = range(2,10)
    
    for n_clusters in range_n_clusters:
     
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        
        ax1.set_ylim([0, len(events) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(events)

        silhouette_avg = silhouette_score(events, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(events, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()

def plot_events(smooth_vm, merged_onsets, merged_ends):

    """
    Visualise events with ends and onsets on original smooth signal 

    """

    #fs = 8550
    #fs = 10000
    fs = len(exp)/100
    window_size = int(fs*10)
    for exp in smooth_vm:
        start_pt = int(window_size/2)
        end_pt = int(len(exp) - (window_size/2))
        exp = exp[start_pt:end_pt]
        plt.figure(figsize=(10, 6))
        plt.plot(exp[:100000], label='Signal')
        for onsets, ends in zip(merged_onsets, merged_ends):
            for i in range(len(onsets)-1):
                if onsets[i] < 100000 and ends[i] < 100000: 
                    plt.axvline(x=onsets[i], color='g', linestyle='--', linewidth = 0.1)
                    plt.axvline(x=ends[i], color='r', linestyle='--', linewidth = 0.1)
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.title('Signal with Event Onsets and Ends')
        plt.legend()
        plt.show()


def visualize_clusters_svd(U, labels):


    """
    Trial funtion // Visualise clusters formed by k-means on U 
    """
    
    # Assuming U has been dimensionality-reduced to at least 2 components
    x = U[:,0]  # First principal component
    y = U[:,1]  # Second principal component
    
    # Create a scatter plot
    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, c=labels, cmap='viridis', marker='o', edgecolor='k', s=50, alpha=0.7)
    
    plt.title('Cluster Visualization')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.colorbar(label='Cluster Label')
    plt.show()


def viz_events(merged_events, smooth_vm, ca_trace):

    """
    Visualise events (plots)

    """

    # fs = 8550
    # window_size = 10*fs 
    
    for i in range(0, len(smooth_vm)):

        fs = len(smooth_vm[i])/100
        #fs = 10000
        window_size = int(10*fs)

        start_pt = int(window_size/2)
        end_pt = int(len(smooth_vm[i]) - (window_size/2))
        exp = smooth_vm[i][start_pt:end_pt]
        ca_exp = ca_trace[i][start_pt:end_pt]
        t = np.linspace(0, 90, len(exp))

       

        
        
        # Save the DataFrame to an Excel file
        # if i%2 == 0 or i==79:
             # df = pd.DataFrame({
             #    'Time (s)': t,
             #    'Vm': exp,
             #    'Calcium': ca_exp
              
             #    })
             # filename = f"/Users/bhavikagopalani/classifier/events_xl/{i}.xlsx"
             # df.to_excel(filename, index=False, engine='openpyxl')
        plt.figure(figsize=(20, 6))
        plt.plot(t, exp, color= 'blue')
        plt.title(f"Merged Events, {i}")
        for on, end in merged_events[i]:
            
            if math.isnan(on) or math.isnan(end):
                continue

            plt.scatter(t[on]-0.8, exp[on], color='green', marker='o')
            plt.scatter(t[end]+0.8, exp[end], color='red', marker='o')
            plt.scatter(t[on], exp[on], color='green', marker='o')
            plt.scatter(t[end], exp[end], color='red', marker='o')
            plt.axvline(x=t[on], color='green', linestyle='-', linewidth=1)
            plt.axvline(x=t[end], color='red', linestyle='-', linewidth=1)
            
        # plt.show()
           
                
        plt.savefig(f"/Users/bhavikagopalani/classifier/events_plot/vm_{i}.png")
        plt.close()

        plt.figure(figsize=(20, 6))
        plt.plot(t, ca_exp, color = 'orange')
        for on, end in merged_events[i]:
            
            if math.isnan(on) or math.isnan(end):
                continue

            # plt.scatter(t[on]-0.8, exp[on], color='green', marker='o')
            # plt.scatter(t[end]+0.8, exp[end], color='red', marker='o')
            # plt.scatter(t[on], exp[on], color='green', marker='o')
            # plt.scatter(t[end], exp[end], color='red', marker='o')
            plt.axvline(x=t[on], color='green', linestyle='-', linewidth=1)
            plt.axvline(x=t[end], color='red', linestyle='-', linewidth=1)
            
        #plt.show()
           
                
        plt.savefig(f"/Users/bhavikagopalani/classifier/events_plot/ca_{i}.png")
        plt.close()

        


def plot_avg_events(events, labels):


    """
    Plot average of a clustered event
    
    """

    events.tolist()
    data = {'signals': events, 'labels': labels}

    print(len(events))
    print(len(labels))

    df = pd.DataFrame(data)

    grouped = df.groupby('labels')['signals']

    for name, group in grouped:
   
        mean_signal = np.mean(np.stack(group.values), axis=0)
        std_signal = np.std(np.stack(group.values), axis=0)
        
     
        time_points = np.arange(mean_signal.size)
        

        plt.plot(time_points, mean_signal, label=f"Mean Signal {name}")
        
        
        plt.fill_between(time_points, mean_signal - std_signal, mean_signal + std_signal, alpha=0.2)


    
    plt.title("Average Signal Plots with Shaded Area")
    plt.xlabel("Time")
    plt.ylabel("Signal Strength")
    plt.legend()
    plt.show()

def extract_ca_events(ca_trace, merged_events):

    """
    Extract calcium events using merged events coordinates (from Vm)
    """


    #define start pt end pt to remove boundary effect region
    #use merged events coordinates to extract events
    #resample back to 50 Hz (maybe unnecesary?)

    #fs = 8550
    #fs = 10000
    fs = len(exp)/100
    window_size = 10*fs 

    merged_ca_events = []

    print("here")
    for exp, events_exp in zip(ca_trace, merged_events):

        start_pt = int(window_size/2)
        end_pt = int(len(exp) - (window_size/2))
        exp = exp[start_pt:end_pt]

        for event in events_exp:
            onset = event[0]
            end = event[1]
            if math.isnan(onset) or math.isnan(end):
                continue
            extracted_event = exp[onset:end]
            #print(len(extracted_event))
            merged_ca_events.append(extracted_event)

    return merged_ca_events

def viz_features(features):

    """"
    
    Make plots of the features extracted from events
    
    """

    stats = ['Mean', 'Min', 'Max', 'Var', 'Std', 'RMS', 'ZeroCrossing', 'Skewness', 'Kurtosis']
    features = np.array(features)

    for i in range(0,9):
        
        events = np.linspace(1, 1587, 1587)
        plt.scatter(events, features[:,i])
        plt.title(stats[i])
        plt.show()


def viz_labels(events, labels, dir, clus, num):

    """
    Plots for labels 
    """



    max_length = max(signal.size for signal in events)

    events = np.array([np.pad(signal, (0, max_length - signal.size), 'constant', constant_values= np.nan) for signal in events])

    unique_labels = set(labels)
    signals_dict = {label: [] for label in unique_labels}

    for signal, label in zip(events, labels):
        signals_dict[label].append(signal)
        
    for labels, events in signals_dict.items():
        print(f"Label {labels}, N= {len(events)}")
        
  

    avg_signals = {label: np.nanmean(signals_dict[label], axis=0) for label in signals_dict}

    #plt.figure(figsize=(10, 7))
    for label, avg_signal in avg_signals.items():

        # plt.figure(figsize=(20, 6))
        cmap = plt.cm.Grays
        plt.figure(figsize=(20, 6))
        for sig, i in zip(signals_dict[label], range(len(signals_dict[label]))):
            colour = cmap(i/700)
            plt.plot(sig, color = colour)
            #plt.savefig(f"/Users/bhavikagopalani/classifier/Basic/events_labels_freq/{label}/{i}.png")
        plt.show()
        plt.close()
        
        #plt.plot(avg_signal, label=f'Average Signal {label}', color = 'purple')
    #     plt.title('Average Signal by Label')
    #     plt.xlabel('Time Point')
    #     plt.ylabel('Signal Value')
    #     plt.legend()
    #     plt.show()
    # # #     plt.savefig(f'{dir}/Onset-End[0.7s]_[4.5s]/non_ca_pop/{clus}/{num}/{label}.png')
    # # #     plt.savefig(f'{dir}/Onset-End[0.7s]_[4.5s]/ca_pop/{clus}/{num}/{label}.png')
    #     plt.close()

def viz_labels_freq(events, labels, dir, clus, num):

    """
    Plots for labels 
    """



    max_length = max(signal.size for signal in events)

    #events = np.array([np.pad(signal, (0, max_length - signal.size), 'constant', constant_values= np.nan) for signal in events])

    unique_labels = set(labels)
    signals_dict = {label: [] for label in unique_labels}

    for signal, label in zip(events, labels):
        signals_dict[label].append(signal)
        
    # for labels, events in signals_dict.items():
    #     print(f"Label {labels}, N= {len(events)}")
        
  

    # avg_signals = {label: np.nanmean(signals_dict[label], axis=0) for label in signals_dict}

    #plt.figure(figsize=(10, 7))
    for label in range(0,5):


        cmap = plt.cm.Grays
        plt.figure(figsize=(20, 6))
      
        
        for sig, i in zip(signals_dict[label], range(len(signals_dict[label]))):
            #colour = cmap(i/700)
            
            #plt.savefig(f"/Users/bhavikagopalani/classifier/Basic/events_labels_freq/{label}/{i}.png")
            fs = 8550  # Sampling frequency (Hz)
            t = np.linspace(0, 1, fs)  # Time vector
          
            
            # FFT Power Spectrum
            fft_vals = fft(sig)
            fft_freq = np.fft.fftfreq(len(sig), 1/fs)
            fft_power = np.abs(fft_vals)**2  # Power spectrum
            
            # Plot FFT Power Spectrum
            # plt.figure(figsize=(12, 5))
            plt.plot(fft_freq[:len(fft_freq)//2], fft_power[:len(fft_freq)//2])
            # plt.xlim(0, 5)
           
            
            wavelet = 'db4'  # Daubechies wavelet
            coeffs = pywt.wavedec(sig, wavelet)  # Decompose signal into wavelet coefficients
            power_dwt = [np.sum(np.abs(c)**2) for c in coeffs]  # Power at each level
            
            # Compute approximate frequencies for each level in the DWT
            freqs_dwt = [fs / (2**(k + 1)) for k in reversed(range(len(coeffs)))]
            # print(coeffs, power_dwt)
            # Plot DWT Power Spectrum
            #plt.subplot(1, 2, 2)
            plt.plot(freqs_dwt, power_dwt, marker='o')
            plt.title("DWT Power Spectrum")
            plt.xlabel("Approximate Frequency (Hz)")
            plt.ylabel("Power")
            plt.xlim(0, 5)
             # Invert x-axis for frequency from high to low
            # plt.tight_layout()
            # plt.show()
        
        # plt.title("FFT Power Spectrum")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Power")
        # plt.gca().invert_xaxis() 
        plt.show()
        plt.close()



def signal_al(labels, events):
    
    
    #max_length = max(signal.size for signal in events)

    #events = np.array([np.pad(signal, (0, max_length - signal.size), 'constant', constant_values= np.nan) for signal in events])

    unique_labels = set(labels)
    signals_dict = {label: [] for label in unique_labels}

    
    
    for signal, label in zip(events, labels):
        signals_dict[label].append(signal)
    
        
    for labels, events in signals_dict.items():
        
        ev_len = [len(ev) for ev in events]
        min_len = min(ev_len)  
        min_index = ev_len.index(min_len)
        #print(min_len)
        
        #events = np.array([np.pad(signal, (0, max_length - signal.size), 
                                  #'constant', constant_values= np.nan) for signal in events])
        
        peak, _ = find_peaks(events[min_index], prominence= 8, width=69)
        
        if len(peak) == 0:
            center = len(events[min_index])//2
            range_roi = [center - 100, center + 100]
        else:
            range_roi = [peak[0] - 100, peak[0] + 100]

        print(range_roi)
        
   
        plt.figure(figsize=(10, 6))
    
        
        for i in range(0, len(events)):
            
            if not i == min_index:
                    
                print(len(events[min_index]))
                print(len(events[i]))
                s_1 = phase_align(events[min_index], events[i], range_roi)
        
                aligned_sig = shift(events[i],s_1,mode='nearest')
                plt.plot(aligned_sig, color = 'orange', ls='--')
                
        
        plt.plot(events[min_index], color = 'black')
        plt.show()
        plt.close()
               

def viz_vm_ca_trace(vm_trace, ca_trace):

    for i in range(0, len(vm_trace)):
        plt.figure(figsize=(20, 15))
        ca_trace[i] = signal.resample(ca_trace[i], len(vm_trace[i]))
        t = np.linspace(0, 100, len(vm_trace[i]))
        plt.plot(t, ca_trace[i], color = 'green')
        plt.plot(t, vm_trace[i], color = 'blue')
        plt.savefig(f"/Users/bhavikagopalani/classifier/data_plots/{i}.png")

def save_labels_mat(labels):

    df = pd.DataFrame(labels, columns=['Label'])

    one_hot_df = pd.get_dummies(df, columns=['Label'])
    
    one_hot_matrix = one_hot_df.to_numpy()
    
    file_path = '/Users/bhavikagopalani/Downloads/GNN/Event-Labels.mat'
    savemat(file_path, {'one_hot_matrix': one_hot_matrix})


def gmm_clus_analysis(X):

    n_clusters = np.arange(2, 9)
    bics = []
    aics = []

    for n in n_clusters:
        print(n)
        gmm = GaussianMixture(n_components=n, random_state=0)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    # Plotting the results
    plt.figure(figsize=(6, 5))
    plt.plot(n_clusters, bics, label='BIC', marker='o', color = 'blue')
    plt.plot(n_clusters, aics, label='AIC', marker='o', color= 'red')

    plt.xlabel('Number of clusters')
    plt.ylabel('Criteria Score')
    plt.legend()
    #plt.title('BIC and AIC Scores by Number of Clusters')
    #plt.grid(True)
    plt.show()

def assign_labels_with_threshold(probs, threshold=0.5):
    labels = []
    for prob in probs:
        # Find the index with the maximum probability
        max_prob_index = np.argmax(prob)
        # Check if the maximum probability exceeds the threshold
        if prob[max_prob_index] > threshold:
            labels.append(max_prob_index)
        else:
            labels.append(-1)
    return np.array(labels)
    
def compute_orthogonality(features):
  
    norms = np.linalg.norm(features, axis=0, keepdims=True)
    normalized_features = features / norms
    
    print(normalized_features)
    dot_product_matrix = np.dot(normalized_features.T, normalized_features)
    
    #orthogonality_matrix = 1 - np.abs(dot_product_matrix)
    f_2 = ['MIN_f',
            'MAX_f','MEAN_f','VAR_f','SKEW_f','KURTOSIS_f', 'Mean (DWT coeffs)', 'Power (DWT coeffs)', 'STD (DWT coeffs)', 'Skew (DWT coeffs)'
            , 'Kurtosis (DWT coeffs)']

    dot_product_df = pd.DataFrame(np.abs(dot_product_matrix), columns= f_2, index= f_2)

    avg_ortho = np.mean(np.abs(dot_product_matrix), axis=0)
    
    plt.figure(figsize=(25, 8))
    sns.heatmap(dot_product_df, annot=True, cmap='coolwarm', center=0.5, cbar=True)
    plt.title("Orthogonality matrix")
    plt.xlabel('Feature Index')
    plt.ylabel('Feature Index')
    plt.show()
    
    return avg_ortho, np.abs(dot_product_matrix)


def remove_ap(vm_events, ca_events, vm_features, ca_features):

    vm_events_new = []
    ca_events_new = []
    vm_features_new = []
    ca_features_new = []
    for i in range(len(vm_events)):
        if np.all(vm_events[i] <= 0):
            vm_events_new.append(vm_events[i])
            ca_events_new.append(ca_events[i])
            vm_features_new.append(vm_features[i])
            ca_features_new.append(ca_features[i])

    return vm_events_new, ca_events_new, vm_features_new, ca_features_new 
            

def create_labels(events):

    labels = []
    num_events = []
    counter = 0
    for ev in events:
        num_events.append(len(ev))
        for i in range(len(ev)):
            labels.append(counter)
        counter = counter + 1 
    return labels, num_events

def count_digits(unique_events, labels):
    result = []
    count = 0
    for sublist in unique_events:
        digit_counts = [0] * 5  # Initialize counts for digits 0-4
        for i in range(len(sublist)):
            num = labels[count+i]
            digit_counts[num] += 1
        result.append(digit_counts)
        count = count + 1
    return result
        

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch


def average_psd(psd_list):
    # Find the maximum length of the PSD arrays
    max_length = max(len(psd) for psd in psd_list)
    
    # Initialize an array to store the sum and a count array for averaging
    sum_psd = np.zeros(max_length)
    count = np.zeros(max_length)
    
    # Sum up all PSDs with proper alignment
    for psd in psd_list:
        length = len(psd)
        sum_psd[:length] += psd
        count[:length] += 1  # Increment count for valid indices
    
    # Calculate the mean by dividing sum by count
    avg_psd = sum_psd / count
    return avg_psd

def plot_average_psd(signals, labels, fs):
    # Find unique labels/groups
    unique_labels = np.unique(labels)
    
    # Dictionary to store PSDs for each group
    psd_dict = {label: [] for label in unique_labels}
    
    # Calculate PSD for each signal and store in the corresponding group
    for signal, label in zip(signals, labels):
        #f, Pxx = welch(signal, fs, nperseg=1024)  # You can adjust nperseg as needed
        ft = fft(signal) 
        S = np.abs(ft**2)/len(signal)
        #f, Pxx = welch(signal, fs, nperseg=50)
        psd_dict[label].append(S)

    for key, value in psd_dict.items(): 
        # avg_psd = average_psd(value)
        plt.figure(figsize=(10, 6))
        # plt.plot(avg_psd)
        for sig in value:
            plt.plot(sig)
        plt.xlim(left = 100 )
        plt.show()
        
            
def are_arrays_equal(array1_, array2_):

    for (array1, array2) in zip(array1_, array2_):
        array1 = np.array(array1)
        array2 = np.array(array2)
        if len(array1) != len(array2):
            return "len"
        
        for i in range(len(array1)):
            if array1[i] != array2[i]:
                return False 
                
       
            
    return True


def feature_matrix_pop(main_dir):

    #Extracts features from the set of signals for population analysis

    """
    
    #FEATURES = ['MIN_f','MAX_f','MEAN_f','VAR_f','SKEW_f','KURTOSIS_f',  
              'Mean (DWT coeffs)', 'Power (DWT coeffs)', 'STD (DWT coeffs)', 'Skew (DWT coeffs)'
            , 'Kurtosis (DWT coeffs)']
    """

    folder_loc = f"{main_dir}/net_events"

    items = os.listdir(folder_loc)
    sorted_files = sorted(items)


    k = 0 
    
    for file, i in zip(sorted_files, range(0, len(sorted_files))):
        if file.endswith('.npz'):
            features_comb = []
            filepath = f"{folder_loc}/{file}"
            print(file)
            with np.load(filepath, allow_pickle=True) as data:
                events = data['df_array'] #events should be a matrix of size 10*(num of events) 
                
                for ev in events: #ev is events in each neuron
                    features_ = []
                    for signal in ev: #signal is each event
                        if len(signal)>0:
                            ft = fft(signal) ##The resulting array ft contains 
                            ##complex numbers that represent the amplitude and phase of each frequency component present in the original signal.
                            #fs = 10000
                            #f, S = welch(signal, fs, nperseg=10)
                            S = np.abs(ft**2)/len(signal) ##This line calculates the power spectrum of the signal, 
                            ##which is a way to represent the distribution of power into frequency components making up the signal.
                            
                            normalized_signal = (signal - np.mean(signal)) / np.std(signal)
                            
                            wavelet_name = 'db4'  # Daubechies wavelet
                            wavelet = pywt.Wavelet(wavelet_name)
                            coeff = pywt.wavedec(signal, wavelet, mode='symmetric')
                            coeff = np.concatenate(coeff)
                            features = [np.min(S), np.max(S), np.mean(S), np.var(S), stats.skew(S), stats.kurtosis(S), np.min(coeff), np.max(coeff),
                                        np.mean(coeff), np.mean(coeff**2), 
                                       np.std(coeff), stats.skew(coeff), stats.kurtosis(coeff)]
                            
                            features_.append(np.array(features)) #appends features for each event in a neuron
                    #print("features_ len:", len(features_)) 
                    features_comb.append(features_) #appends features for all neurons
                    if len(features_) == 0:
                        print(file)
                        print(events)
                        
                    #print("features_comb len:",len(features_comb))
                df = pd.DataFrame(features_comb)
                print("k:", k)
                
                df_array = df.values
                np.savez(f'{main_dir}/net_events_features/net_features_{k:04d}.npz', df_array=df_array)
                k = k + 1 



                            
def pop_labels(main_dir,rf_clf):

    folder_loc = f"{main_dir}/net_events_features"

    items = os.listdir(folder_loc)
    sorted_files = sorted(items)
    labels_comb = []
    #labels_full = []
    for file, i in zip(sorted_files, range(0, len(sorted_files))):
        if file.endswith('.npz'):
            filepath = f"{folder_loc}/{file}"
            print(file)
            with np.load(filepath, allow_pickle=True) as data:

                features_ = data['df_array']
                labels = []
               
                for neu in features_:
                    if len(neu) > 0:
                        neu = np.vstack(neu)
                        #print(neu.shape)
                        pred_labels = rf_clf.predict(neu)
                        labels.append(pred_labels)
                    else:
                        pred_labels = []
                        labels.append(pred_labels)
                        
                labels_comb.append(labels)


    return labels_comb

def flatten_network(network):
    # Concatenate all arrays in the network list into a single 1D array
    concatenated = np.concatenate(network)
    return concatenated.flatten()
    
def analyze_network_states(networks):
    summary = []
    
    for i, network in enumerate(networks):
        all_states = flatten_network(network)
        state_counts = Counter(all_states)
        total_count = sum(state_counts.values())

        if total_count == 0: 
            summary.append({
            'Network': i + 1,
            'Top 2 States %': 0,
            'Top 3 States %': 0,
            'Top 4 States %': 0})
        else: 
            
            # Calculate the percentage for the top 2, 3, and 4 states
            top_2_states = sum([count for value, count in state_counts.most_common(2)]) / total_count
            top_3_states = sum([count for value, count in state_counts.most_common(3)]) / total_count
            top_4_states = sum([count for value, count in state_counts.most_common(4)]) / total_count
            
            summary.append({
                'Network': i + 1,
                'Top 2 States %': top_2_states * 100,
                'Top 3 States %': top_3_states * 100,
                'Top 4 States %': top_4_states * 100
            })
        
    return pd.DataFrame(summary)      


def calculate_percentages(state_summary):
    total_networks = len(state_summary)
    
    conditions = [
        (100, 100), (90, 90), (80, 80), (70, 70), (60, 60)
    ]
    
    percentages = []
    
    for two_state_threshold, three_state_threshold in conditions:
        two_state_count = len(state_summary[state_summary['Top 2 States %'] >= two_state_threshold])
        three_state_count = len(state_summary[state_summary['Top 3 States %'] >= three_state_threshold])
        
        percentages.append({
            'Condition': f'{two_state_threshold}% for 2 states, {three_state_threshold}% for 3 states',
            '% Networks with 2 States': (two_state_count / total_networks) * 100,
            '% Networks with 3 States': (three_state_count / total_networks) * 100
        })
    
    return pd.DataFrame(percentages)