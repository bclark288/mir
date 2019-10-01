import numpy as np
from matplotlib import pyplot as plt
import scipy.signal as sig
import librosa



def block_audio(x, blockSize, hopSize, fs):

    # allocate memory
    numBlocks = np.ceil(x.size / hopSize).astype('int')
    xb = np.zeros([numBlocks, blockSize])

    # compute time stamps
    t = (np.arange(0, numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)),axis=0)

    for n in range(numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0,blockSize)] = x[np.arange(i_start, i_stop + 1)]

    return xb, t



def extract_spectral_centroid(xb, fs):
    blockSize = xb.shape[1]
    hannWindow = sig.hann(blockSize, True)

    spectralCentroids = []
    for block in xb:
        block *= hannWindow
        blockFFT = np.abs(np.fft.fft(block)[:blockSize//2+1])        
        spectralCentroids.append(np.sum(np.arange(blockFFT.size) * blockFFT) / np.sum(blockFFT))

    return np.array(spectralCentroids) * fs / (blockSize)


        
def extract_rms(xb):
    blockSize = xb.shape[1]        
    RMS = []
    
    for block in xb:
        blockRMS = np.sum(block**2)/blockSize
        RMSdB = 20 * np.log10(blockRMS)
        RMSdB = np.clip(RMSdB, a_min=-100, a_max=None)
        RMS.append(RMSdB)
    return np.array(RMS)


    
def extract_zerocrossingrate(xb):
    blockSize = xb.shape[1]
    zerocrossings = []
    block1 = xb[0]
    
    for block2 in xb[1:]:    
        zerocrossings.append(np.sum(np.abs(np.sign(block2) - np.sign(block1))) / (2*blockSize))
        block1 = block2.copy()

    return np.array(zerocrossings)


    
def extract_spectral_crest(xb):
    blockSize = xb.shape[1]
    hannWindow = sig.hann(blockSize, True)
    spectralCrests = []
    for block in xb:    
        block *= hannWindow
        blockFFT = np.abs(np.fft.fft(block)[:blockSize//2])        
        spectralCrests.append(np.max(blockFFT) / np.sum(blockFFT))
    return np.array(spectralCrests)
    

    
def extract_spectral_flux(xb):
    blockSize = xb.shape[1]
    hannWindow = sig.hann(blockSize, True)

    spectralFlux = []
    block1 = xb[0]
    block1 *= hannWindow
    block1FFT = np.abs(np.fft.fft(block1)[:blockSize//2])
    
    for block2 in xb[1:]:    
        block2 *= hannWindow
        block2FFT = np.abs(np.fft.fft(block2)[:blockSize//2])
        flux = np.sqrt(np.sum((block2FFT - block1FFT)**2)) / (blockSize//2)
        spectralFlux.append(flux)
        block1FFT = block2FFT.copy()

    return np.array(spectralFlux)



def extract_features(x, blockSize, hopSize, fs):
    blocks, times = block_audio(x, blockSize, hopSize, fs)
    centroids = extract_spectral_centroid(blocks, fs)
    rms = extract_rms(blocks)
    zeroCrossings = extract_zerocrossingrate(blocks)
    crest = extract_spectral_crest(blocks)
    flux = extract_spectral_flux(blocks)
    print(centroids.shape)
    print(rms.shape)
    print(zeroCrossings.shape)
    print(crest.shape)
    print(flux.shape)
    return np.stack((centroids, rms, zeroCrossings, crest, flux))
     


def aggregate_feature_per_file(features):
    print(features.shape)
    aggFeature = np.zeros(features.shape)
    aggFeatures[i * 2] = np.mean(features[i])
    aggFeatures[i * 2 + 1] = np.std(features[i])

    
def normalize_zscore(featureData):
    normalizedFeatureMatrix = featureData.copy()
