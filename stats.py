from bisect import bisect_left
import numpy as np

def histogram(samples, delta=1.):
    low, high = min(samples), max(samples)
    low = int(low // delta) * delta
    high = int(high // delta + 2) * delta
    histoLength = int((high - low) / delta)
    xs = [i*delta + low for i in range(histoLength)]
    ys = [0 for _ in range(histoLength)]
    for sample in samples:
        i = bisect_left(xs, sample)
        ys[i] += 1
    #change bin x from low part to mid of bin
    xs = [x + 0.5*delta for x in xs]
    return xs, ys

def histogramCDF(samples, delta=1.):
    xs, ys = histogram(samples, delta=delta)
    ysCDF = pdfToCDF(ys)
    return xs, ys, ysCDF

def pdfToCDF(pdf):
    cdf = [val for val in pdf]
    for i in range(1, len(cdf)):
        cdf[i] += cdf[i-1]
    return cdf

#means, pPos w/ over numGroups partitions
def computeSixFoldMeanViaSubsampling(sample, 
                                     numGroups=3, 
                                     nSubsamples=100, 
                                     comparisons=[1.,1.1,1.2,1.3,1.4,1.5]):
    #partition sample into numGroups partitons
    partitionIndices = np.random.randint(numGroups, size=len(sample))
    partition = [[] for _ in range(numGroups)]
    for i, index in enumerate(partitionIndices):
        partition[index].append(sample[i])
    means = [0. for _ in range(numGroups)]
    pPos = [[] for _ in range(numGroups)]
    #compute means, pPos
    for i in range(numGroups):
        means[i] = np.mean(partition[i])
        pPos[i] = computePPosViaSubsampling(partition[i], 
                                            comparisons, 
                                            nSubsamples=nSubsamples)
    meansSEM = [np.std(mean) / np.sqrt(numGroups) for mean in means]
    pPosSEM = [np.std(p) / np.sqrt(numGroups) for p in pPos]
    return means, meansSEM, pPos, pPosSEM

#pos hyp: do better than market (comparison == 1) or specific % above market (comparison > 1)
def computePPosViaSubsampling(sample, comparisons, nSubsamples=100):
    uniformInt = lambda upperBd, n: np.random.randint(upperBd, size=n)
    subsamples = (np.array([sample[i] for i in uniformInt(len(sample), len(sample))]) 
                  for _ in range(nSubsamples))
    subsampleMeans = [subsample.mean() for subsample in subsamples]
    ps = [0. for _ in comparisons]
    for i, comparison in enumerate(comparisons):
        ps[i] = sum(subsampleMean >= comparison 
                    for subsampleMean in subsampleMeans) / nSubsamples
    return ps

def computeSEM(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return mean, std / np.sqrt(len(data))
