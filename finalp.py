# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:39:26 2019

@author: Maple

Data (data.npy) contains 399 neurons being observed, where we have 98 trials of mice watching the
same movie every single time. The neuron activity signal then has 2430 points in time where we record the signal. 

Signal is about ~80 seconds

The neurons each has an associated number, ranging from 0-399

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.linalg as la

Dat = np.load('data.npy')
Keys = pd.read_csv('key.csv')
Region = Keys['ccf']    # Index corresponds with Neuron Number

# 0-399 number of neurons 

neurons_num = np.shape(Dat)[0]
trials_num = np.shape(Dat)[1]
signal_len = np.shape(Dat)[2]
time = np.linspace(0, 80, signal_len)

# %% Count how many unique regions there are
regcount = {}
for r in Region:
    if r in regcount:
        regcount[r] = regcount[r] + 1
    else:
        regcount[r] = 0

print('We have ' + str(len(regcount)) + ' unique regions')

#%% We first see that our data contains some NaN entries
if np.any(np.isnan(Dat)):
    print('Contains NaN entrie(s)')
    
# %% Locate where NaN entries exist - we see that every 29th trial contains NaN values
for i in np.arange(neurons_num):
    for j in np.arange(trials_num):
        if np.any(np.isnan(Dat[i][j])):
            print('neuron ' + str(i) + ' and trial ' + str(j+1) + ' contains NaN values')

#%% Remove 29th trial from every neuron
beg = np.arange(0, 28)
en = np.arange(29, trials_num)

trials = np.concatenate((beg, en))
dat2 = Dat[:, trials, :]
trials_num = trials_num - 1

# %% 
signal = Dat[0][:]
plt.imshow(signal)
plt.colorbar()


#%% Confirming that there are no more NaN values in our dataset 
if np.any(np.isnan(dat2)):
    print('Contains NaN entrie(s)')
else:
    print('Contains no NaN entrie(s)')

#%% Plot an example neuron activity signal 
neuron = 120
trial = 40
signal1 = dat2[neuron, trial, :]
plt.plot(time, signal1)
plt.title('Activity of Neuron ' + str(neuron + 1) + ' during trial ' + str(trial + 1))
plt.xlabel('Time (sec)')
plt.ylabel('Activity')
plt.savefig('actneu' + str(neuron+1) + 'tr' + str(trial+1) + '.png')
plt.show()
    

#%% Create a Signal Matrix where each signal is put into a column
signals_all = np.zeros([signal_len, trials_num*neurons_num])
k = 0
for n in np.arange(neurons_num):
    for m in np.arange(trials_num):
        signals_all[:, k] = dat2[n][m][:]
        k = k+1

#%% Mean center signal matrix
means = np.mean(signals_all, axis = 0)
signals_mcen = signals_all-means
signals_mcen = signals_mcen*(1/((np.shape(signals_mcen)[0]-1)))  

#%% apply SVD
u,s,vh = la.svd(signals_mcen, full_matrices=False)
v = np.transpose(vh)

#%% We do not see a significant principal component in this case
plt.scatter(np.arange(len(s)), s**2/np.sum(s**2))
plt.ylim([0, 1])
plt.title('Amount of Variance in each Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.savefig('varsignals.png')


# %% Average the activity for all trials for each neuron
Activity_avg = np.zeros([neurons_num, signal_len])
for n in np.arange(neurons_num):
    aavg = np.zeros([1, signal_len])
    for m in np.arange(trials_num):
        aavg = aavg + dat2[n][m][:]
    aavg = (1/trials_num)*aavg
    Activity_avg[n] = aavg

#%% Mean Center
means = np.mean(Activity_avg, axis = 0)
signals_mcen = Activity_avg-means
signals_mcen = signals_mcen*(1/((np.shape(signals_mcen)[0]-1)))  

#%% apply SVD
u,s,vh = la.svd(signals_mcen, full_matrices=False)
v = np.transpose(vh)

#%% Plot variance in PCs, we see more variance in this case
plt.scatter(np.arange(len(s)), s**2/np.sum(s**2))
plt.ylim([0, 1])
plt.title('Amount of Variance in each Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Variance')
plt.savefig('avgvars.png')


# %% Average activity for 0 and 80 are pretty different looking
plt.plot(time, Activity_avg[80])    

#%% Create Spectrogram of Averaged signal 
width = 0.01
timeslide = np.arange(0,80, 0.1)
spc = np.zeros([len(timeslide), signal_len])
k = 0
n_num = 0
signal1 = Activity_avg[n_num]
for ts in timeslide:
    fil = np.exp(-width*(time-ts)**10)
    fs = signal1*fil
    fst = np.fft.fft(fs)
    spc[k,:] = np.abs(np.fft.fftshift(fst))
    k = k+1

x2 = np.arange(-signal_len/2, 0)
x1 = np.arange(0, signal_len/2)
kk = np.concatenate((x1, x2))

fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].plot(time, signal1)
ax[0].set_xlabel('Time (sec)')
ax[0].set_ylabel('Activity Level')
ax[0].set_title('Averaged Activity of Neuron ' + str(n_num + 1))
ax[0].set_xlim([0, 80])
im = ax[1].imshow(Activity_avg, aspect='auto')
ax[1].set_title('Average Activity Across all Neurons ')
ax[1].set_xlabel('Time (sec)')
ax[1].set_ylabel('Signal')
plt.tight_layout()
plt.colorbar(im, ax=ax[1])
plt.savefig('neuron' + str(n_num + 1) + 'fig.png')
plt.show()

#%% Reshape and store into large matrix for each neuron 
spch = np.shape(spc)[0]
spcw = np.shape(spc)[1]
All_spc = np.zeros([neurons_num, spch*spcw])

for n in np.arange(neurons_num):
    sig = Activity_avg[0]
    spc = np.zeros([len(timeslide), signal_len])
    k = 0
    for ts in timeslide:
        fil = np.exp(-width*(time-ts)**10)
        fs = signal1*fil
        fst = np.fft.fft(fs)
        spc[k,:] = np.abs(np.fft.fftshift(fst))
        k = k+1
    All_spc[n, :] = np.reshape(spc, [1, spch*spcw])

#%% 
from sklearn.model_selection import train_test_split

indx = np.arange(neurons_num)
X_train, X_test, Y_train, Y_test = train_test_split(indx, Region, test_size=0.2, random_state=42)

# %%
from sklearn.neighbors import KNeighborsClassifier
n = 8
clf = KNeighborsClassifier(n)
clf.fit(Activity_avg[X_train,:], Y_train)
score_kmean = clf.score(Activity_avg[X_test,:], Y_test)
print('Accuracy Score for Kmeans: ' + str(score_kmean))

# %% SVM
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(Activity_avg[X_train,:], Y_train)
score_SVM = clf.score(Activity_avg[X_test,:], Y_test)
print('Accuracy Score for SVM: ' + str(score_SVM))

# %% Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(Activity_avg[X_train,:], Y_train)
score_RF = clf.score(Activity_avg[X_test,:], Y_test)
print('Accuracy Score for DecisionTree: ' + str(score_RF))

# We do not see very good results with this - only about a ~20-30% accuracy rate

# %%
plt.bar(['K Neighbors Classifier', 'SVM', 'DecisionTree Classifier'], [score_kmean, score_SVM, score_RF], color=['red','blue','green'])
plt.title('Accuracy of Different Classification Algorithms')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy (%)')
plt.ylim([0, 1])
plt.savefig('acc.png')


# %%
'''


Here we attempt to see if removing the neurons with no activity spike makes a difference at all
when it comes to classification





'''
# See if neuron has spike of activity
nonact_neus = 0
to_rev = []
for i in np.arange(neurons_num):
    for j in np.arange(trials_num):
        if np.all(Dat[i][j] == 0):
            # print('neuron ' + str(i) + ' and trial ' + str(j+1) + ' has no activity spike')
            nonact_neus = nonact_neus+1
            to_rev.append(i)
            break

print('There are ' + str(nonact_neus) + ' trials with no activity spike')

# %% Remove neurons with no activity trials
neurons = np.arange(neurons_num)
keep_neu = [n for n in neurons if n not in to_rev]
dat3 = dat2[keep_neu, :, :]

region2 = Region[keep_neu]

neurons_num = neurons_num - nonact_neus
    
# %% Confirming that we have removed all neurons with no activity spike           
k = 0
for i in np.arange(neurons_num):
    for j in np.arange(trials_num):
        if np.all(dat3[i][j] == 0):
            k = k+1
            break

print('There are ' + str(k) + ' trials with no activity spike')
#%% 
from sklearn.model_selection import train_test_split

indx = np.arange(neurons_num)
X_train, X_test, Y_train, Y_test = train_test_split(indx, region2, test_size=0.2, random_state=42)

# %%
from sklearn.neighbors import KNeighborsClassifier
n = 8
clf = KNeighborsClassifier(n)
clf.fit(Activity_avg[X_train,:], Y_train)
score_kmean = clf.score(Activity_avg[X_test,:], Y_test)
print('Accuracy Score for Kmeans: ' + str(score_kmean))

# %% SVM
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(Activity_avg[X_train,:], Y_train)
score_SVM = clf.score(Activity_avg[X_test,:], Y_test)
print('Accuracy Score for SVM: ' + str(score_SVM))

# %% Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(Activity_avg[X_train,:], Y_train)
score_RF = clf.score(Activity_avg[X_test,:], Y_test)
print('Accuracy Score for DecisionTree: ' + str(score_RF))

# %%
plt.bar(['K Neighbors Classifier', 'SVM', 'DecisionTree Classifier'], [score_kmean, score_SVM, score_RF], color=['red','blue','green'])
plt.title('Accuracy of Different Classification Algorithms')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy (%)')
plt.ylim([0, 1])
plt.savefig('acc_removed_noactivity.png')

# %%
'''


Unsupervised methods using original data




'''
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=8, random_state = 0).fit(Activity_avg)
klabels = kmeans.labels_
labcount = {}
for l in klabels:
    if l in labcount:
        labcount[l] = labcount[l] + 1
    else:
        labcount[l] = 0


#%% 
k = 0
plt.figure(figsize=(12, 1))
for r in regcount.keys():
    AAA = [i for i, x in enumerate(Region) if x == r]
    neu = np.arange(k, k+regcount[r])
    plt.scatter(neu, np.zeros(regcount[r]), label=r)
    k = k + regcount[r]
plt.yticks([])
plt.xlabel('Neuron Number')
plt.legend(loc=(1.02,-1))
plt.title('Classification of Neurons Based on Location')
plt.savefig('uh.png')

#%% 
k = 0
plt.figure(figsize=(12, 1))
for r in regcount.keys():
    AAA = [i for i, x in enumerate(Region) if x == r]
    plt.scatter(AAA, np.zeros(len(AAA)), label=r)
    k = k + regcount[r]
plt.yticks([])
plt.xlabel('Neuron Number')
plt.legend(loc=(1.02,-1))
plt.title('Classification of Neurons Based on Location')
plt.tight_layout()
plt.savefig('w.png')

#%% 
k = 0
plt.figure(figsize=(12, 1))
for r in labcount.keys():
    AAA = [i for i, x in enumerate(klabels) if x == r]
    plt.scatter(AAA, np.zeros(len(AAA)), label=r)
    k = k + labcount[r]
plt.yticks([])
plt.xlabel('Neuron Number')
plt.legend(loc=(1.02,-1))
plt.title('Classification of Neurons Based on Unsupervised KMeans')
plt.savefig('u.png')






