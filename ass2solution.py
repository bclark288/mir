from scipy.io import wavfile
from scipy.signal import correlate, find_peaks
import numpy as np
import matplotlib.pyplot as plt
import os
import math



def window_audio(xb):
    nbrBlocks = xb.shape[0]
    blkSize = xb.shape[1]
    rms = np.zeros(nbrBlocks)
    w_audio = []
    for i in range(nbrBlocks):
        wa = 0.5*(1 - np.cos(2*np.pi*xb[i]/(xb.shape[0] - 1)))
        w_audio.append(wa)
    wa = np.asarray(w_audio)
    return wa
def block_audio(x,blockSize,hopSize,fs):
    # allocate memory    
    numBlocks = math.ceil(x.size / hopSize)    
    xb = np.zeros([numBlocks, blockSize])    
    # compute time stamps    
    t = (np.arange(0, numBlocks) * hopSize) / fs    
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)    
    for n in range(0, numBlocks):       
        i_start = n * hopSize        
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])        
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]    
    return (xb,t)

def extract_spectral_centroid(xb, fs):
    wa = window_audio(xb)
    #print("wa ",wa)
    
    centroid = []
    for i in range(wa.shape[0]):
        spgrm = abs(np.fft.fft(wa[i]))
        idxs = np.arange(spgrm.shape[0])
        ctrd = sum(idxs*spgrm)/sum(spgrm)

        normalized_spectrum = spgrm / sum(spgrm)  # like a probability mass function
        normalized_frequencies = np.linspace(0, 1, len(spgrm))
        spectral_centroid = sum(normalized_frequencies * normalized_spectrum)

        #print("ctrd ",spectral_centroid)
        #print('ws ',weighted_sum.shape)
        centroid.append(spectral_centroid)
        #centroid.append(sum(idxs*spgrm)/sum(spgrm))
    # avoid NaN for silence frames
    centroid = np.asarray(centroid)
    #centroid[sum(wa[0:-1] == 0)] = 0    
    # convert from index to Hz
    #centroid = centroid / (wa.shape[1]) * fs/2
    #print('centroid',centroid)
    return centroid

def extract_rms(xb):
    #np.sqrt(np.mean(xb^2))
    nbrBlocks = xb.shape[0]
    rms = np.zeros(nbrBlocks)
    for i in range(nbrBlocks):
        rms[i] = np.sqrt(np.mean(np.power(xb[i],2)))
    eps = 0.00001
    for i in range(rms.shape[0]):
        if rms[i] < eps:
            rms[i] = eps
    rms = 20*np.log10(rms)
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
    spgrm = np.fft.fft(wa)
    fluxs = []
    for i in range(spgrm.shape[0]):
        delta = np.diff(spgrm[i])
        flux = np.sqrt(np.sum(np.power(delta,2)))/delta.shape[0]
        fluxs.append(flux)
    fluxs = np.asarray(fluxs)
    return flux

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
        agg_features[i] = np.mean(features[i])
        agg_features[i * 2 + 1] = np.std(features[i])
    return agg_features

def  get_feature_data(path, blockSize, hopSize):
    featureData = []
    for file in os.listdir(path):
        fs,data = wavfile.read(path + '/' + file)
        features = extract_features(data,blockSize,hopSize,fs)
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
    plt_vector = []
    plt_vector.append(normalized_files[0,:,0])
    print('ctd ',normalized_files[0,:,0])
    plt_vector.append(normalized_files[0,:,3])
    plt_vector.append(normalized_files[1,:,0])
    print('ctd2 ',normalized_files[1,:,0])
    plt_vector.append(normalized_files[1,:,3])
    plt_vector = np.asarray(plt_vector)
    print("plt ",plt_vector.shape)
    #plt_vector = np.reshape(plt_vector,(64,4))
    print("plt ",plt_vector.shape)
    for feat in range(plt_vector.shape[0]):
        plt.plot(range(normalized_files.shape[1]),plt_vector[feat])
        plt.show()     

if __name__ == '__main__':
    visualize_features('/Users/bclark66/Downloads/music_speech')

