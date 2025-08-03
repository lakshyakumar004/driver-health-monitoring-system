#!/usr/bin/env python3
import serial
import time
import board
import busio
import numpy as np
import adafruit_mlx90640
from vmdpy import VMD
from scipy.signal import welch
import max30102_1
import hrcalc_1
import RPi.GPIO as GPIO
from ppg_helper import ensemble_peak, get_window_stats_27_features, bandpass_filter, movingaverage 
import matplotlib.pyplot as plt
import scipy.signal as signal
def tempearature_reading():
    # Setup I2C connection
    i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_2_HZ
    frame = np.zeros((32*24,))
    tot_time = 5
    time_period = 0.5
    temperatures = []
    while tot_time>=0:
        tot_time-=time_period
        try:
            mlx.getFrame(frame)
            curr_temp = np.mean(frame)
            #average_temp_c += curr_temp
            temperatures.append(curr_temp)
            #count_readings += 1
            time.sleep(time_period)
        except ValueError as e:
            time.sleep(time_period)
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            time.sleep(time_period)
    T_obs = np.array(temperatures)
    n = len(T_obs)
    mu_hat = np.mean(T_obs)
    sigma2_hat = np.mean((T_obs - mu_hat) ** 2)
    print("Estimated Variance (σ̂^2):", sigma2_hat)
    log_likelihood = - (n / 2) * np.log(2 * np.pi * sigma2_hat) - (1 / (2 * sigma2_hat)) * np.sum((T_obs - mu_hat) ** 2)
    print("Estimated log likelihood:", log_likelihood)
    return mu_hat

def get_ppg_gsr_reading():        
    ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1.0)
    time.sleep(1)
    ser.reset_input_buffer()
    print("Serial OK")
    gsrs = []
    ppg = []
    timestamps = []
    try:
        timeout = 60
        start_time = time.time()
        # Run the loop for up to 1 minute
        while time.time() - start_time < timeout:
            ser.reset_input_buffer()
            while ser.in_waiting<=0:
                time.sleep(0.1)
            response = ser.readline().decode('utf-8').rstrip()
            str_time = time.time()
            
            gsrs.append(int(response.split()[0]))
            ppg.append(int(response.split()[1]))
            
    except KeyboardInterrupt:
        print("Close Serial Communication.")
        ser.close()
    ser.close()
    return gsrs, ppg

def gsr_analysis(time_series):
    fs = 10  
    alpha = 2000       # Bandwidth constraint
    tau = 0.           # Noise-tolerance (no strict fidelity enforcement)
    K = 3              # Number of modes (noise, tonic, phasic)
    DC = 0             # No DC component imposed
    init = 1           # Initialize omegas uniformly
    tol = 1e-7         # Tolerance for convergence

    # Run VMD on the signal
    u, _, omega = VMD(time_series, alpha, tau, K, DC, init, tol)
    # Separate components based on frequency
    tonic_component = u[0]       # Slow-changing mode (tonic)
    phasic_component = u[1]      # Fast-changing mode (phasic)
    noise_component = u[2]       # High-frequency mode (noise >0.3 Hz)

    # Calculate Power Spectral Density (PSD) for tonic and phasic components
    _, noise_psd = welch(noise_component, fs, nperseg=len(time_series)-1)
    _, tonic_psd = welch(tonic_component, fs, nperseg=len(time_series)-1)
    _, phasic_psd = welch(phasic_component, fs, nperseg=len(time_series)-1)

    # Calculate power as the area under the PSD curve
    tonic_power = np.trapz(tonic_psd)
    phasic_power = np.trapz(phasic_psd)
    noise_power = np.trapz(noise_psd)
    #print(noise_power)
    # Calculate power ratio of phasic to tonic
    power_ratio = phasic_power / tonic_power
    return noise_power, tonic_power, phasic_power, power_ratio

def ppg_analysis(time_series):
    fs = 10
    filtered_ppg = bandpass_filter(time_series, 0.5, 4, fs)
    ff_ppg = movingaverage(filtered_ppg)
    peaks = ensemble_peak(ff_ppg,fs,3)
    #UNCOMMENT THESE LINES IF YOU WANT TO VISUALISE YOUR PPG DATA WITH ITS PEAK DETECTED
    #plt.figure(figsize=(10, 4))
    #plt.plot(ff_ppg)
    #plt.plot(peaks, ff_ppg[peaks], "x")
    #plt.title("Filtered PPG Signal")
    #plt.xlabel("Sample Index")
    #plt.ylabel("Amplitude")
    #plt.show()
    
    hrv = get_window_stats_27_features(ff_ppg,3,False,False)
    return hrv['SDNN']
    
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def filtering_for_spo2_data(data):
    fs = 10
    cutoff_freq = 0.01  # Cutoff at 0.01 Hz
    b, a = butter_lowpass(cutoff_freq, fs)
    filtered_signal = signal.filtfilt(b, a, data)
    return filtered_signal
    
def spo2_HR():
    m = max30102_1.MAX30102()
    hrarr = []
    spo2arr = []
    while len(hrarr)<100 and len(spo2arr)<100:
        red, ir = m.read_sequential()
        hr, boolhr, spo2, boolspo2 = hrcalc_1.calc_hr_and_spo2(ir, red)
        if boolhr and boolspo2:
            #print(f"{hr} {spo2}")
            hrarr.append(hr)
            spo2arr.append(spo2)
        time.sleep(0.1)
    filtered_hr = filtering_for_spo2_data(hrarr)
    filtered_spo2 = filtering_for_spo2_data(spo2arr)
    return filtered_hr, filtered_spo2
        
#Thermal Camera code
temperature = tempearature_reading()
print(f"temperature reading is {temperature}")

#PPG and GSR code
gsr_time_series, ppg_time_series = get_ppg_gsr_reading()
print(ppg_time_series)
print(gsr_time_series)
print(len(gsr_time_series))

#GSR analysis
noise_power, tonic_power, phasic_power, power_ratio = gsr_analysis(gsr_time_series)
print(noise_power)
print(tonic_power)
print(phasic_power)
print(f"phasic to tonic power ratio {power_ratio}")

#PPG analysis
SDNN_ppg = ppg_analysis(ppg_time_series)
print(f"SDNN from ppg signal is {SDNN_ppg}")

#MAX30102'S CODE
#GPIO.cleanup()
hrarr, spo2arr = spo2_HR()
print(hrarr)
print(spo2arr)
#print(f"SPO2 {round(spo2,2)}")
#print(f"HR {round(hr,2)}")