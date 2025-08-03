from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import kurtosis, skew
import numpy as np
import scipy.signal
import math

fs = 10
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y
def movingaverage(data, periods=4):
    result = []
    data_set = np.asarray(data)
    weights = np.ones(periods) / periods
    result = np.convolve(data_set, weights, mode='valid')
    return result
def valley_detection(dataset, fs):
    window = []
    valleylist = []
    ybeat = []
    listpos = 0
    TH_elapsed = np.ceil(0.36 * fs)
    nvalleys = 0
    valleyarray = []
    localaverage = np.average(dataset)
    for datapoint in dataset:

        if (datapoint > localaverage) and (len(window) < 1):
            listpos += 1
        elif (datapoint <= localaverage):
            window.append(datapoint)
            listpos += 1
        else:
            minimum = min(window)
            beatposition = listpos - len(window) + (window.index(min(window)))
            valleylist.append(beatposition)
            window = []
            listpos += 1
    for val in valleylist:
        if nvalleys > 0:
            prev_valley = valleylist[nvalleys - 1]
            elapsed = val - prev_valley
            if elapsed > TH_elapsed:
                valleyarray.append(val)
        else:
            valleyarray.append(val)
            
        nvalleys += 1    

    return valleyarray


def pair_valley(valley):
    pair_valley=[]
    for i in range(len(valley)-1):
        pair_valley.append([valley[i], valley[i+1]])
    return pair_valley
def statistic_detection(signal, fs):
    
    valley = pair_valley(valley_detection(signal, fs))
    stds=[]
    kurtosiss=[]
    skews=[]

    for val in valley: 
        stds.append(np.std(signal[val[0]:val[1]]))
        kurtosiss.append(kurtosis(signal[val[0]:val[1]]))
        skews.append(skew(signal[val[0]:val[1]])) 

    return stds, kurtosiss, skews, valley
def eliminate_noise_in_time(data, fs, ths,cycle=1):
    stds, kurtosiss,skews, valley = statistic_detection(data, fs)
    
    stds_, kurtosiss_, skews_ = [], [], []
    stds_ = [np.mean(stds[i:i+cycle]) for i in range(0,len(stds)-cycle+1,cycle)]
    kurtosiss_ = [np.mean(kurtosiss[i:i+cycle]) for i in range(0,len(kurtosiss)-cycle+1,cycle)]
    skews_ = [np.mean(skews[i:i+cycle]) for i in range(0,len(skews)-cycle+1,cycle)]    
      
    eli_std = [stds_.index(x) for x in stds_ if x < ths[0]]
    eli_kurt = [kurtosiss_.index(x) for x in kurtosiss_ if x < ths[1]]
    eli_skew = [skews_.index(x) for x in skews_ if x > ths[2][0] and x < ths[2][1]]

    total_list = eli_std + eli_kurt + eli_skew

    dic = dict()
    for i in total_list:
        if i in dic.keys():
            dic[i] += 1
        else:
            dic[i] = 1
            
    new_list = []
    for key, value in dic.items():
        if value >= 3:
            new_list.append(key)
    new_list.sort()
    
    eliminated_data = []
    index = []
    for x in new_list:
        index.extend([x for x in range(valley[x*cycle][0],valley[x*cycle+cycle-1][1],1)])

    print(len(data), len(index))
    return len(data), len(index), index
# fs = 10  # Example: Sampling frequency in Hz

def threshold_peakdetection(dataset, fs):
    #print("dataset: ",dataset)
    ybeat = []
    mean = np.average(dataset)
    window = []
    peaklist = []
    listpos = 0
    TH_elapsed = np.ceil(0.36 * fs)
    npeaks = 0
    peakarray = []
    
    localaverage = np.average(dataset)
    for datapoint in dataset:
        if (datapoint < localaverage) and (len(window) < 1):
            listpos += 1
        elif (datapoint >= localaverage):
            window.append(datapoint)
            listpos += 1
        else:
            maximum = max(window)
            beatposition = listpos - len(window) + (window.index(max(window)))
            peaklist.append(beatposition)
            window = []
            listpos += 1
            
    ## Ignore if the previous peak was within 360 ms interval becasuse it is T-wave
    for val in peaklist:
        if npeaks > 0:
            prev_peak = peaklist[npeaks - 1]
            elapsed = val - prev_peak
            if elapsed > TH_elapsed:
                peakarray.append(val)
        else:
            peakarray.append(val)
            
        npeaks += 1    


    return peaklist

def determine_peak_or_not(prevAmp, curAmp, nextAmp):
    if prevAmp < curAmp and curAmp >= nextAmp:
        return True
    else:
        return False
    
def onoff_set(peak, sig):     # move peak from dy signal to original signal   
    onoffset = []
    for p in peak:
        for i in range(p, 0,-1):
            if sig[i] == 0:
                onset = i
                break
        for j in range(p, len(sig)):
            if sig[j] == 0:
                offset = j
                break
        if onset < offset:
            onoffset.append([onset,offset])
    return onoffset
def slope_sum_function(data,fs):
    dy = [0]
    
    dy.extend(np.diff(data))    
    w = fs // 8
    dy_ = [0] * w
    for i in range(len(data)-w):
        sum_ = np.sum(dy[i:i+w])
        if sum_ > 0:
            dy_.append(sum_)
        else:
            dy_.append(0)
    
    init_ths = 0.6 * np.max(dy[:3*fs])
    ths = init_ths
    recent_5_peakAmp = []
    peak_ind = []
    bef_idx = -300
    
    for idx in range(1,len(dy_)-1):
        prevAmp = dy_[idx-1]
        curAmp = dy_[idx]
        nextAmp = dy_[idx+1]
        if determine_peak_or_not(prevAmp, curAmp, nextAmp) == True:
            if (idx - bef_idx) > (300 * fs /1000):  
                if len(recent_5_peakAmp) < 100:  
                    if curAmp > ths:
                        peak_ind.append(idx)
                        bef_idx = idx
                        recent_5_peakAmp.append(curAmp)
                elif len(recent_5_peakAmp) == 100:
                    ths = 0.7*np.median(recent_5_peakAmp)
                    if curAmp > ths:
                        peak_ind.append(idx)
                        bef_idx = idx
                        recent_5_peakAmp.pop(0)
                        recent_5_peakAmp.append(curAmp)
                        
    onoffset = onoff_set(peak_ind, dy_)
    corrected_peak_ind = []
    for onoff in onoffset:
        segment = data[onoff[0]:onoff[1]]
        corrected_peak_ind.append(np.argmax(segment) + onoff[0])
                    
    return corrected_peak_ind
def seperate_division(data,fs):
    divisionSet = []
    for divisionUnit in range(0,len(data)-1,5*fs):  
        eachDivision = data[divisionUnit: (divisionUnit+1) * 5 * fs]
        divisionSet.append(eachDivision)
    return divisionSet
def first_derivative_with_adaptive_ths(data, fs):
    
    peak = []
    divisionSet = seperate_division(data, fs)
    selectiveWindow = 2 * fs
    block_size = 5 * fs
    bef_idx = -300
    
    for divInd in range(len(divisionSet)):
        block = divisionSet[divInd]
        ths = np.mean(block[:selectiveWindow]) # ths: 2 seconds mean in block
        
        firstDeriv = block[1:] - block[:-1]
        for i in range(1,len(firstDeriv)):
            if  firstDeriv[i] <= 0 and firstDeriv[i-1] > 0:
                if block[i] > ths:
                    idx = block_size*divInd + i
                    if idx - bef_idx > (300*fs/1000):
                        peak.append(idx)
                        bef_idx = idx
                                                
    return peak
def moving_average(signal, kernel='boxcar', size=5):
    size = int(size)
    window = scipy.signal.get_window(kernel, size)
    w = window / window.sum()
    
    x = np.concatenate((signal[0] * np.ones(size), signal, signal[-1] * np.ones(size)))
    
    smoothed = np.convolve(w, x, mode='same')
    smoothed = smoothed[size:-size]
    return smoothed

def moving_averages_with_dynamic_ths(signals,sampling_rate=10, peakwindow=.111, 
                                     beatwindow=.667, beatoffset=.02, mindelay=.3,show=False):
    
    signal = signals.copy()
    # ignore the samples with n
    signal[signal < 0] = 0
    sqrd = signal**2
    
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak = moving_average(sqrd, size=ma_peak_kernel)
    
    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat = moving_average(sqrd, size=ma_beat_kernel)
    
    thr1 = ma_beat + beatoffset * np.mean(sqrd)    # threshold 1

    waves = ma_peak > thr1
    
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]),
                                        waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1],
                                        np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    min_len = int(np.rint(peakwindow * sampling_rate))    # threshold 2
    min_delay = int(np.rint(mindelay * sampling_rate))
    peaks = [0]

    for i in range(num_waves):

        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        if len_wave < min_len: # threshold 2
            continue

        data = signal[beg:end]
        locmax, props = scipy.signal.find_peaks(data, prominence=(None, None))

        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between peaks(300ms)
            if peak - peaks[-1] > min_delay:
                peaks.append(peak)

    peaks.pop(0)

    peaks = np.asarray(peaks).astype(int)
    return peaks
def lmm_peakdetection(data,fs):
    
    peak_final = []
    peaks, _ = find_peaks(data,height=0)
    
    for peak in peaks:
        if data[peak] > 0:
            peak_final.append(peak)
        
    return peak_final
def ensemble_peak(preprocessed_data, fs, ensemble_ths=4):
    
    peak1 = threshold_peakdetection(preprocessed_data,fs)
    peak2 = slope_sum_function(preprocessed_data, fs)
    peak3 = first_derivative_with_adaptive_ths(preprocessed_data, fs)
    peak4 = moving_averages_with_dynamic_ths(preprocessed_data)
    peak5 = lmm_peakdetection(preprocessed_data,fs)
    
    peak_dic = dict()

    for key in peak1:
        peak_dic[key] = 1

    for key in peak2:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
    
    for key in peak3:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
        
    for key in peak4:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
        
    for key in peak5:
        if key in peak_dic.keys():
            peak_dic[key] += 1
        else:
            peak_dic[key] = 1
        
    peak_dic = dict(sorted(peak_dic.items()))

    count = 0
    cnt = 0
    bef_key = 0
    margin = 1

    new_peak_dic = dict()

    for key in peak_dic.keys():
        if cnt == 0:
            new_peak_dic[key] = peak_dic[key]
        else:
            if bef_key + margin >= key:  
                if peak_dic[bef_key] > peak_dic[key]: 
                    new_peak_dic[bef_key] += peak_dic[key]
                else:
                    #print("new peak dic: ",new_peak_dic)
                    new_peak_dic[key] = peak_dic[key] + peak_dic[bef_key]
                    del(new_peak_dic[bef_key])
                    bef_key = key
            else:
                new_peak_dic[key] = peak_dic[key]
                bef_key = key
        cnt += 1
    
    ensemble_dic = dict()
    for (key, value) in new_peak_dic.items():
        if value >= ensemble_ths:
            ensemble_dic[key] = value
            
    final_peak = list(ensemble_dic.keys())
    
    return final_peak


def calc_RRI(peaklist, fs):
    RR_list = []
    RR_list_e = []
    cnt = 0
    while (cnt < (len(peaklist)-1)):
        RR_interval = (peaklist[cnt+1] - peaklist[cnt]) 
        ms_dist = ((RR_interval / fs) * 1000.0)  
        cnt += 1
        RR_list.append(ms_dist)
    mean_RR = np.mean(RR_list)

    for ind, rr in enumerate(RR_list):
        if rr >  mean_RR - 300 and rr < mean_RR + 300:
            RR_list_e.append(rr)
            
    RR_diff = []
    RR_sqdiff = []
    cnt = 0
    while (cnt < (len(RR_list_e)-1)):
        RR_diff.append(abs(RR_list_e[cnt] - RR_list_e[cnt+1]))
        RR_sqdiff.append(math.pow(RR_list_e[cnt] - RR_list_e[cnt+1], 2))
        cnt += 1
        
    return RR_list_e, RR_diff, RR_sqdiff
def calc_heartrate(RR_list):
    HR = []
    heartrate_array=[]
    window_size = 10

    for val in RR_list:
        if val > 400 and val < 1500:
            heart_rate = 60000.0 / val 
        elif (val > 0 and val < 400) or val > 1500:
            if len(HR) > 0:
                heart_rate = np.mean(HR[-window_size:])
            else:
                heart_rate = 60.0
        else:
            
            print("err")
            heart_rate = 0.0

        HR.append(heart_rate)

    return HR
def calc_fd_hrv(RR_list):  
    
    rr_x = []
    pointer = 0
    for x in RR_list:
        pointer += x
        rr_x.append(pointer)
        
    if len(rr_x) <= 3 or len(RR_list) <= 3:
        print("rr_x or RR_list less than 5")   
        return 0
    
    RR_x_new = np.linspace(rr_x[0], rr_x[-1], int(rr_x[-1]))
    
   
    interpolated_func = UnivariateSpline(rr_x, RR_list, k=3)
    
    datalen = len(RR_x_new)
    frq = np.fft.fftfreq(datalen, d=((1/1000.0)))
    frq = frq[range(int(datalen/2))]
    Y = np.fft.fft(interpolated_func(RR_x_new))/datalen
    Y = Y[range(int(datalen/2))]
    psd = np.power(Y, 2)  

    lf = np.trapz(abs(psd[(frq >= 0.04) & (frq <= 0.15)])) 
    hf = np.trapz(abs(psd[(frq > 0.15) & (frq <= 0.5)])) 
    ulf = np.trapz(abs(psd[frq < 0.003]))
    vlf = np.trapz(abs(psd[(frq >= 0.003) & (frq < 0.04)]))
    
    if hf != 0:
        lfhf = lf/hf
    else:
        lfhf = 0
        
    total_power = lf + hf + vlf
    lfp = lf / total_power
    hfp = hf / total_power

    features = {'LF': lf, 'HF': hf, 'ULF' : ulf, 'VLF': vlf, 'LFHF': lfhf, 'total_power': total_power, 'lfp': lfp, 'hfp': hfp}
    bef_features = features
    
    return features

def calc_td_hrv(RR_list, RR_diff, RR_sqdiff, window_length): 
    
    # Time
    HR = calc_heartrate(RR_list)
    HR_mean, HR_std = np.mean(HR), np.std(HR)
    meanNN, SDNN, medianNN = np.mean(RR_list), np.std(RR_list), np.median(np.abs(RR_list))
    meanSD, SDSD = np.mean(RR_diff) , np.std(RR_diff)
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    
    NN20 = [x for x in RR_diff if x > 20]
    NN50 = [x for x in RR_diff if x > 50]
    pNN20 = len(NN20) / window_length
    pNN50 = len(NN50) / window_length
    
    
    bar_y, bar_x = np.histogram(RR_list)
    TINN = np.max(bar_x) - np.min(bar_x)
    
    RMSSD = np.sqrt(np.mean(RR_sqdiff))
    

    features = {'HR_mean': HR_mean, 'HR_std': HR_std, 'meanNN': meanNN, 'SDNN': SDNN, 'medianNN': medianNN,
                'meanSD': meanSD, 'SDSD': SDSD, 'RMSSD': RMSSD, 'pNN20': pNN20, 'pNN50': pNN50, 'TINN': TINN}

    return features
def get_window_stats_27_features(ppg_seg, window_length, ensemble, ma_usage):  
    
    fs = 10  
    
    if ma_usage:
        fwd = moving_average(ppg_seg, size=3)
        bwd = moving_average(ppg_seg[::-1], size=3)
        ppg_seg = np.mean(np.vstack((fwd,bwd[::-1])), axis=0)
    ppg_seg = np.array([item.real for item in ppg_seg])
    
    #peak = threshold_peakdetection(ppg_seg, fs)
    #peak = first_derivative_with_adaptive_ths(ppg_seg, fs)
    #peak = slope_sum_function(ppg_seg, fs)
    #peak = moving_averages_with_dynamic_ths(ppg_seg)
    peak = lmm_peakdetection(ppg_seg,fs)

        
    if ensemble:
        ensemble_ths = 3
        #print("one algorithm peak length: ", len(peak))
        peak = ensemble_peak(ppg_seg, fs, ensemble_ths)
        #print("after ensemble peak length: ", len(peak))
        
        if(len(peak) < 100):
            print("skip")
            return []

        
    RR_list, RR_diff, RR_sqdiff = calc_RRI(peak, fs)
    #print(RR_list)
    
    if len(RR_list) <= 3:
        return []
    
    td_features = calc_td_hrv(RR_list, RR_diff, RR_sqdiff, window_length)
    fd_features = calc_fd_hrv(RR_list)
    # if fd_features == 0:
    #     return []
    # nonli_features = calc_nonli_hrv(RR_list,label)
    total_features = {**td_features,**fd_features}
    return total_features