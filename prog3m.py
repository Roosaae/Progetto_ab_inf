#!/usr/bin/python3

# esercizio per i tre multipoli pari

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

Nbins = 600
Nmeasures = 100  # il numero totale di misure è 10000
measures = []
measures_XI0 = []
measures_XI2 = []
measures_XI4 = []

for i in np.arange(Nmeasures)+1:
    fname = f'data/MockMeasures_2PCF_Test{test}/MockMeasures_2PCF_Correlation_MULTIPOLES_Test{test}_{i}.fits'

    file = fits.open(fname)
    table = file[1].data.copy()
    measures_XI0.append(table['XI0'])
    measures_XI2.append(table['XI2'])
    measures_XI4.append(table['XI4'])
    measures = np.concatenate((measures_XI0,measures_XI2,measures_XI4), axis=1) # matrice completa dei tre multipoli
    
    if i==1:
        scale = table['SCALE']
    del table
    file.close()
    
measures_XI0 = np.asarray(measures_XI0).transpose()
measures_XI2 = np.asarray(measures_XI2).transpose()
measures_XI4 = np.asarray(measures_XI4).transpose()
measures = np.asarray(measures).transpose()

mean_xi = np.mean(measures,axis=1)

cov_meas = np.cov(measures) # covarianza misurata


if TEST_COVARIANCE: # confronto tra la funzione covarianza di numpy e la covarianza calcolata numericamente
    print('')
    print('Test to compare numpy covariance to numeric covariance:')
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
    for j in range(Nbins): #okay, sarebbe R_ij
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

def covf1f2(x1, x2, sig1, l1, sig2, l2): # correlazione mista tra i multipoli
    return (np.sqrt(2.*l1*l2)*np.exp(-(np.sqrt((x1-x2)**2.)**2./(l1**2.+l2**2.)))*sig1*sig2)/np.sqrt(l1**2.+l2**2.)
    
cov_th = np.zeros((Nbins,Nbins),dtype=float)

cov_th_00 = np.zeros((200,200),dtype=float)
cov_th_22 = np.zeros((200,200),dtype=float)
cov_th_44 = np.zeros((200,200),dtype=float)

cov_th_02 = cov_th_20 = np.zeros((200,200),dtype=float)
cov_th_04 = cov_th_40 = np.zeros((200,200),dtype=float)
cov_th_24 = cov_th_42 = np.zeros((200,200),dtype=float)

for i in range(0,200):
    for j in range(0,200):
        cov_th_00[i,j] = covf(scale[i],scale[j],sigs[0],ls[0])
        cov_th_22[i,j] = covf(scale[i],scale[j],sigs[1],ls[1])
        cov_th_44[i,j] = covf(scale[i],scale[j],sigs[2],ls[2])
        
        cov_th_02[i,j] = cov_th_20[i,j] = covf1f2(scale[i],scale[j],sigs[0],ls[0],sigs[1],ls[1])
        cov_th_04[i,j] = cov_th_40[i,j] = covf1f2(scale[i],scale[j],sigs[0],ls[0],sigs[2],ls[2])
        cov_th_24[i,j] = cov_th_42[i,j] = covf1f2(scale[i],scale[j],sigs[1],ls[1],sigs[2],ls[2])
        
row_1 = np.concatenate((cov_th_00,cov_th_20,cov_th_40), axis=1) # matrice completa della covarianza teorica
row_2 = np.concatenate((cov_th_02,cov_th_22,cov_th_42), axis=1)
row_3 = np.concatenate((cov_th_04,cov_th_24,cov_th_44), axis=1)
cov_th = np.concatenate((row_1,row_2,row_3), axis=0)


if PLOTS:

    gratio = (1.+5.** 0.5)/2.

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

rms_deviation = np.std(norm_residuals.reshape(Nbins**2)) # differenza quadratica media dei residui normalizzati

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

