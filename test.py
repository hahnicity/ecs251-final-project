#Mchine learning script
from sys import argv, exit
import csv
import os
import math
from glob import glob
from pandas import read_csv
import numpy as np
#using keywords data to do classification
#data_p = np.recarray((len(keywords),), names = keywords)--doesn't work
import time
import matplotlib.pyplot as plt

#------------------------------------------------------
from fitgaus import fitgausdist
#----------------------------------------

def normlize(vec):
    average = np.sum(vec)
    stddev = np.std(vec)
    vec = (vec-average)/stddev
    return vec



def get_cohort_files(cohort, keywords):
    """
        Read data and get parameters for processing
    """
    if cohort not in ["ardscohort", "controlcohort"]:
        raise Exception("Input must either be ardscohort or controlcohort")
    path = '/Users/shasha/Documents/courses/ECS251/project_final/data/' + cohort
    #dirs = os.listdir(path)
    dirs = glob(path+'/0*')
    data = []
    k = 0

    for dir in dirs:
        path_temp = dir
        name = glob(path_temp+'/*.csv')
        df = read_csv(name[0])
        #print 'Opening file: ', name
        k += 1

        #for each patient
        flag = 0
        paras = []
        m = 50
        a = df[keywords[0]]
        print len(a)
        for i in range(0, len(keywords)):
            a = df[keywords[i]]
            if len(a) <= m:
                print name, ' is an empty file'
                flag = 1
                break
            #fit to normal distribution
            #para = fitgausdist(a)
            #mean = np.sum(a)/float(len(a))
            #std = np.std(a)
            #para = [mean, std]
            partial = a[0:m]
            para = partial
            if math.isnan(para[0]):
                print name[0], keywords[i]
                flag = 1
                break
            paras.extend(para)
        if flag == 0:
            data.append(paras)
        k -= flag

    return data, k


keywords = ['I:E ratio','eTime','PIP', 'Maw', 'PEEP', 'ipAUC', 'epAUC'] #, 'minF_to_zero'
cohort = "ardscohort"
data_p, k1 = get_cohort_files(cohort, keywords)
print 'get %d files from patients' % k1

cohort = "controlcohort"
temp, k2 = get_cohort_files(cohort, keywords)
print 'get %d files from control group' % k2


labels = np.append(np.ones(k1-3), np.zeros(k2-3))
#data_p = np.transpose(data_p)

#print "Done reading data, get arrary of size", len(data_p)

#from sklearn import preprocessing
#normalized_X = preprocessing.normalize(np.array(data_p))
def normalize(x):
    x = np.dot(x, np.identity(len(x))-np.ones(len(x))/len(x))
    return x

#-------------FDA-----------------
datatrain = []
datatrain.extend(data_p[0:k1-3])
datatrain.extend(temp[0:k2-3])




datatest = data_p[-3:]
datatest.extend(temp[-3:])

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(datatrain, labels)

LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                           solver='svd', store_covariance=False, tol=0.0001)

print(clf.predict(datatest))


#should clear that one file


'''
    1. some blank files
    2. some files missing features
    3. some files too short, less than 1000 breath
    4. not using the min_f feature... the important one
    
    
    1. find out the collinear variables -- rank 70 of matrix 76, 
    2. use other partial data for testing - need to verify validity
    3. pull out the classification plot
    4. look for more features to use instead of just raw feature
    5. break one patien to multiple?
    6. kernalize?
    
    1. 300 too long
    2. 50 better than 100 or 30
    '''









#--------------------------------
'''
    for root, dirs, files in os.walk(path_ards):
    for name in files:
        if name.endswith((".csv")):

        for filename in os.listdir(path_ards):
        df = pd.read_csv(path_ards+filename)
        temp = []
        flag = 0
        print 'Patient ', filename, '  , number of breaths', len(df[keywords[0]])
        for i in range(0, len(keywords)):
        a = df[keywords[i]]    #column
        amean = np.sum(a)#/len(a)
        data_person.append(a)
        if amean == 0:
        print filename#, keywords[i]
        flag = 1
        break
        #print len(temp)
        temp = np.array(data_person, dtype = keywords)
        if flag == 0:
        data_raw.append(temp)
        

def mypca(x):
    result = la.eigh(np.dot(x, np.transpose(x)))
    plt.plot(result[0])
    plt.ylabel('Log of Residual History')
    plt.show()
    z = np.dot(np.dot(np.transpose(result[1]),result[1]), x)
    zcumu = np.dot(result[0], z)
    plt.plot(zcumu)
    plt.ylabel('Log of Residual History')
    plt.show()
    print zcumu
    return zcumu



'''













