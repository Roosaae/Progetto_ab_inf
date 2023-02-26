#!/usr/bin/python3

# esercizio per un multipolo

from astropy.io import fits
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

print('')
print('Which set do you want to try?') # scelta del set
print('')
test = int(input())
TEST_COVARIANCE = True
PLOTS = True

Nbins = 200
Nmeasures = 100  # il numero totale di misure è 10000
measures = []

for i in np.arange(Nmeasures)+1:
    fname = f'data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'

    file = fits.open(fname)
    table = file[1].data.copy()
    measures.append(table['XI0']) 
    
    if i==1:
        scale = table['SCALE']
    del table
    file.close()
    
measures = np.asarray(measures).transpose()
mean_xi = np.mean(measures,axis=1)
cov_meas = np.cov(measures) # covarianza misurata

if TEST_COVARIANCE: # confronto tra la funzione covarianza di numpy e la covarianza calcolata numericamente
    print('')
    print('Test to compare numpy covariance with measured covariance:')
    print('')

    AVE = np.zeros((Nbins,),dtype=float)
    COV = np.zeros((Nbins,Nbins),dtype=float)

    for i in range(Nmeasures):
        AVE += measures[:,i]
    AVE /= Nmeasures

    for i in range(Nbins):
        for j in range(Nbins):
            COV[i,j] = (np.sum(measures[i]*measures[j])-AVE[i]*AVE[j]*Nmeasures)/(Nmeasures-1) 

    print('Largest deviation: {}'.format(np.max(np.abs(COV-cov_meas)))) # la deviazione è trascurabile: utilizzo la funzione di numpy
    print('')
    
# matrice di correlazione (covarianza misurata normalizzata a uno sulla diagonale)
corr_meas = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        corr_meas[i,j] = cov_meas[i,j]/(cov_meas[i,i]*cov_meas[j,j])**0.5

# scelta dei parametri in base alla scelta del set
if test == 1:
    sigs = [0.02, 0.02, 0.02]
    ls = [25, 50, 75]
elif test == 2:
    sigs = [0.02, 0.01, 0.005]
    ls = [50, 50, 50]
else:
    sigs = [0.02, 0.01, 0.005]
    ls = [5, 5, 5]
    
# matrice di covarianza teorica
def covf(x1, x2, sig, l): # autocorrelazione
    return sig**2.*np.exp(-(x1-x2)**2./(2.*l**2.))
    
cov_th = np.zeros((Nbins,Nbins),dtype=float)
for i in range(Nbins):
    for j in range(Nbins):
        cov_th[i,j] = covf(scale[i],scale[j],sigs[0],ls[0])
        
        
if PLOTS:

    gratio = (1.+5.**0.5)/2.

    dpi = 300
    cmin = -np.max(cov_th)*0.05
    cmax =  np.max(cov_th)*1.05

    # grafico della covarianza misurata
    fig = plt.figure(figsize=(6,4))
    plt.title('Measured covariance matrix')
    plt.imshow(cov_meas, vmin=cmin, vmax=cmax)
    cbar = plt.colorbar(orientation="vertical", pad=0.02)
    plt.show()

    # grafico della covarianza teorica
    fig = plt.figure(figsize=(6,4))
    plt.title('Theoretical covariance matrix')
    plt.imshow(cov_th, vmin=cmin, vmax=cmax)
    cbar = plt.colorbar(orientation="vertical", pad=0.02)
    plt.show()

    # grafico dei residui tra covarianza teorica e misurata
    fig = plt.figure(figsize=(6,4))
    plt.title('Residuals')
    plt.imshow(cov_th-cov_meas, vmin=cmin, vmax=-cmin)
    cbar = plt.colorbar(orientation="vertical", pad=0.02)
    plt.show()
    
# calcolo dei residui normalizzati sulla varianza    
norm_residuals = np.zeros_like(cov_th)
for i in range(Nbins):
    for j in range(Nbins):
        corr_th = cov_th[i,j]**2./(np.sqrt(cov_th[i,i]*cov_th[j,j])**2.) # correlazione teorica
        norm_residuals[i,j] = (cov_th[i,j]-cov_meas[i,j])*np.sqrt((Nmeasures-1.)/((1.+corr_th)*cov_th[i,i]*cov_th[j,j]))

rms_deviation = np.std(norm_residuals.reshape(Nbins**2))

print('')
print(f"rms deviation of normalized residuals: {rms_deviation}")
print('')

# validazione

if rms_deviation < 1.1:
    print("**********")
    print("* PASSED *")
    print("**********")
else:
    print("!!!!!!!!!!")
    print("! FAILED !")
    print("!!!!!!!!!!")
































