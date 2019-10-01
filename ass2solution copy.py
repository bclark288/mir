from scipy.io import wavfile
from scipy.signal import correlate, find_peaks,hann
import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
import math



def window_audio(xb):
    nbrBlocks = xb.shape[0]
    blkSize = xb.shape[1]
    rms = np.zeros(nbrBlocks)
    w_audio = []
    hwind = hann(blkSize)
    for i in range(nbrBlocks):
        wa = xb[i] * hwind
        #wa = 0.5*(1 - np.cos(2*np.pi*xb[i]/(xb.shape[0] - 1)))
        w_audio.append(wa)
    wa = np.asarray(w_audio)
    # plt.plot(wa_audio[0])
    # plt.plot(hann(blkSize))
    # plt.show()

    return wa
def block_audio(x,blockSize,hopSize,fs):
    numBlocks = math.ceil(x.size / hopSize)    
    xb = np.zeros([numBlocks, blockSize])       
    t = (np.arange(0, numBlocks) * hopSize) / fs    
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)    
    for n in range(0, numBlocks):       
        i_start = n * hopSize        
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])        
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]    
    return (xb,t)

def extract_spectral_centroid(xb, fs):
    wa = window_audio(xb)
    centroid = []
    for i in range(wa.shape[0]):
        spectrum = abs(np.fft.fft(wa[i][:wa.shape[1]//2]))
        #print("spgrm ",spgrm.shape)
        freq_indices = np.arange(len(spectrum))
        spectral_centroid = np.sum(freq_indices * spectrum)/np.sum(spectrum)
        spectral_centroid = spectral_centroid*(fs/wa[i].size)
        centroid.append(spectral_centroid)
    centroid = np.asarray(centroid)
    return centroid

def extract_rms(xb):
    nbrBlocks = xb.shape[0]
    rms = np.zeros(nbrBlocks)
    for i in range(nbrBlocks):
        rms[i] = np.sqrt(np.mean(np.power(xb[i],2)))
    eps = 0.00001
    for i in range(rms.shape[0]):
        if rms[i] < eps:
            rms[i] = eps
    rms = 20*np.log10(rms)
    for val in range(rms.shape[0]):
        if rms[val] < -100:
            rms[val] = -100
    return rms

def extract_zerocrossingrate(xb):
    nbrBlocks = xb.shape[0]
    zeroCrossingRate = np.zeros(nbrBlocks)
    for i in range(nbrBlocks):
        zeroCrossingRate[i] = 0.5*np.mean(np.abs(np.diff(np.sign(xb[i]))))
    return zeroCrossingRate

def extract_spectral_crest(xb):
    wa = window_audio(xb)
    crests = []
    for i in range(wa.shape[0]):
        spgrm = abs(np.fft.fft(wa[i]))
        crests.append(np.max(spgrm)/sum(spgrm))
    crests = np.asarray(crests)
    return crests

def extract_spectral_flux(xb):
    wa = window_audio(xb)
    
    fluxs = []
    for i in range(wa.shape[0]):
        spgrm = abs(np.fft.fft(wa[i]))
        delta = np.diff(spgrm)
        flux = np.sqrt(np.sum(np.power(delta,2)))/delta.shape[0]
        fluxs.append(flux)
    fluxs = np.asarray(fluxs)
    return fluxs

def extract_features(x, blockSize, hopSize, fs):
    xb,t = block_audio(x,blockSize,hopSize,fs)
    features = np.zeros([5,xb.shape[0]])
    features[0] = extract_spectral_centroid(xb,fs)
    features[1] = extract_rms(xb)
    features[2] = extract_zerocrossingrate(xb)
    features[3] = extract_spectral_crest(xb)
    features[4] = extract_spectral_flux(xb)
    return features

def aggregate_feature_per_file(features):
    agg_features = np.zeros(features.shape[0]*2)
    for i in range(features.shape[0]):
        agg_features[i * 2] = np.mean(features[i])
        agg_features[i * 2 + 1] = np.std(features[i])
    return agg_features

def  get_feature_data(path, blockSize, hopSize):
    featureData = []
    for file in os.listdir(path):
        if file == 'marlene.wav':   
            fs,data = wavfile.read(path + '/' + file)
            features = extract_features(data,blockSize,hopSize,fs)
            rosa_sig,rsr = librosa.core.load(path + '/' + file,sr=fs)
            rosa_centroid = librosa.feature.spectral_centroid(rosa_sig,sr=fs,n_fft=blockSize,hop_length=hopSize)
            rosa_rms = librosa.feature.rms(rosa_sig,frame_length=blockSize,hop_length=hopSize)
            rosa_zcr = librosa.feature.zero_crossing_rate(rosa_sig,frame_length=blockSize,hop_length=hopSize)
            print("diff ",features[0], rosa_centroid)
            print("diff rms",features[1],rosa_rms)
            print("diff zcr ",features[2],rosa_zcr)
            agg_features = aggregate_feature_per_file(features)
            featureData.append(agg_features)
        
    featureData = np.asarray(featureData)
    return(featureData)

def normalize_zscore(featureData):
    fdm = np.mean(featureData)
    fstd = np.std(featureData)
    normFeatures = (featureData-np.mean(featureData))/np.std(featureData)
    return normFeatures

def visualize_features(path_to_musicspeech):
    feature_files = []
    for file in os.listdir(path_to_musicspeech):
        if file.endswith('_wav'):
            print("dir ",path_to_musicspeech + '/' + file)
            feature_files.append(get_feature_data(path_to_musicspeech + '/' + file,1024,256))
    feature_files = np.asarray(feature_files)
    normalized_files = normalize_zscore(feature_files)
    print(normalized_files.shape) 
    xaxis_range = np.arange(normalized_files.shape[1]) 
    plt_vector = []
    fig, ax = plt.subplots()
    # sc mean
    plt.scatter(normalized_files[0,:,0],normalized_files[0,:,6],color='blue',label='Speech')
    #plt.scatter(xaxis_range,normalized_files[1,:,0],label='Music:SCmean')
    #scr mean
    plt.scatter(normalized_files[1,:,0],normalized_files[1,:,6],color='red',label='Music')
    #plt.scatter(xaxis_range,normalized_files[1,:,6],label='Music:SCRmean')
    plt.xlabel("Spectral Centroid Mean")
    plt.ylabel("Spectral Crest Mean")
    plt.legend()
    plt.show()
    # SF mean
    plt.scatter(normalized_files[0,:,8],normalized_files[0,:,4],color='blue',label='Speech')
    #plt.scatter(xaxis_range,normalized_files[1,:,8],label='Music:SFmean')
    # ZCR mean
    plt.scatter(normalized_files[1,:,8],normalized_files[1,:,4],color='red',label='Music')
    #plt.scatter(xaxis_range,normalized_files[1,:,4],label='Music:ZCRmean')
    plt.xlabel("Spectral Flux Mean")
    plt.ylabel("Zero Crossing Rate Mean")
    plt.legend()
    plt.show()
    # RMS mean
    plt.scatter(normalized_files[0,:,2],normalized_files[0,:,3],color='blue',label='Speech')
    #plt.scatter(xaxis_range,normalized_files[1,:,2],label='Music:RMSmean')
    # RMS std
    plt.scatter(normalized_files[1,:,2],normalized_files[1,:,3],color='red',label='Music')
    #plt.scatter(xaxis_range,normalized_files[1,:,3],label='Music:RMSstd')
    plt.xlabel("RMS Mean")
    plt.ylabel("RMS STD")
    plt.legend()
    plt.show()
    # ZCR std
    plt.scatter(normalized_files[0,:,5],normalized_files[0,:,7],color='blue',label='Speech')
    #plt.scatter(xaxis_range,normalized_files[1,:,5],label='Music:ZCRstd')
    # SCR std
    plt.scatter(normalized_files[1,:,5],normalized_files[1,:,7],color='red',label='Music')
    #plt.scatter(xaxis_range,normalized_files[1,:,7],label='Music:SCRstd')
    plt.xlabel("Zero Crossing Rate STD")
    plt.ylabel("Spectral Crest STD")
    plt.legend()
    plt.show()
    # SC std
    plt.scatter(normalized_files[0,:,1],normalized_files[0,:,9],color='blue',label='Speech')
    #plt.scatter(xaxis_range,normalized_files[1,:,1],label='Music:SCstd')
    # SF std
    plt.scatter(normalized_files[1,:,1],normalized_files[1,:,9],color='red',label='Music')
    #plt.scatter(xaxis_range,normalized_files[1,:,9],label='Music:SFstd')
    plt.xlabel("Spectral Crest STD")
    plt.ylabel("Spectral Flux STD")
    plt.legend()
    plt.show()
    #plt_vector = np.asarray(plt_vector)
    #for feat in range(plt_vector.shape[0]):
    # xaxis = range(normalized_files.shape[1])
    # for i in range(4):
    #     if i % 2 == 0:
    #         plt.scatter(plt_vector[i])
    #     else:
    #         plt.scatter(plt_vector[i])
    # plt.show()     

if __name__ == '__main__':
    # visualize_features('./music_speech')
    visualize_features('/Users/bclark66/Downloads/music_speech')

