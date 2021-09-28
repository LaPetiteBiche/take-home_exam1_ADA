import pandas as pd
import  numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
from matplotlib import pyplot as plt
from sklearn import preprocessing


##################
# a)
file = 'data/bankrupt.csv'
df = pd.read_csv(file)
corr = df.corr()
corr.to_csv(r'res/ex_2_corr.csv', index = True)
##################


##################
# b)
stats = df.describe(percentiles=[.5, .99])
stats.to_csv(r'res/ex_2_descr_1.csv', index = True)
##################

##################
# c)

df2 = df.loc[:, df.columns !='Bankrupt?']
df_bank = df['Bankrupt?']

#Normalize

bank_yes = df[df['Bankrupt?'] == 1]
bank_yes = bank_yes.loc[:, bank_yes.columns !='Bankrupt?']
x = bank_yes.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_norm_1 = pd.DataFrame(x_scaled)


bank_no = df[df['Bankrupt?'] == 0]
bank_no = bank_no.loc[:, bank_no.columns !='Bankrupt?']
x2 = bank_no.values 
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x2)
df_norm_2 = pd.DataFrame(x_scaled)




for n in df_norm_1.columns:
    
    x=df2.columns[n]
    x = x.replace(" ","_")
    x = x.replace("%","")
    x = x.replace("(","")
    x = x.replace(")","")
    x = x.replace("/","")
    x = x.replace("$","")
    x = x.replace("£","")
    x = x.replace("¥","")
    
    plt.hist(df_norm_1[n],bins=100, alpha =0.5, label='Bankrupt')
    plt.hist(df_norm_2[n],bins=100, alpha = 0.5, label='Not Bankrupt')
    plt.title(x)
    plt.legend()
    plt.show()
    plt.savefig(r'res/fig_c/{}.png'.format(x))
    plt.close()
    plt.savefig(sys.stdout.buffer)
    
    

   
    

##################

##################
# d)
#Split -> Test size 1-(6000/6820)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = df2.values
y = df['Bankrupt?'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.120234604,random_state=0)

min_max_scaler = preprocessing.MinMaxScaler() 
min_max_scaler.fit(X_train)  
X_train = min_max_scaler.transform(X_train) 
X_test = min_max_scaler.transform(X_test)

#kNN
neigh = KNeighborsClassifier(n_neighbors=3)
model = neigh.fit(X=X_train, y=y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]

#Naive Bayes
clf = GaussianNB()
model2 = clf.fit(X=X_train, y=y_train)
y_pred2 = model2.predict(X_test)
y_proba2 = model2.predict_proba(X_test)[:,1]

#Decision Tree
clf = tree.DecisionTreeClassifier()
model3 = clf.fit(X=X_train, y=y_train)
y_pred3 = model3.predict(X_test)
y_proba3 = model3.predict_proba(X_test)[:,1]

#Performance -> Could have use a function, for improvement next time :)
df_confusion = pd.crosstab(y_test, y_pred)
FP = df_confusion.sum(axis=0) - np.diag(df_confusion)
FN = df_confusion.sum(axis=1) - np.diag(df_confusion)
TP = np.diag(df_confusion)
TN = df_confusion.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# true positive rate
TPR = TP/(TP+FN)
# true negative rate
TNR = TN/(TN+FP) 
# Precision
PPV = TP/(TP+FP)
#Recall
REC = TP/(FN+TP)
# false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# F1
F1 = 2*(PPV*REC)/(PPV+REC)

params = pd.Series()
params['Recall']=REC
params['Precision']=PPV
params['F1 Score']=F1
params['TPR']=TPR
params['FPR']=FPR
params['TNR']=TNR
params['FNR']=FNR

df_confusion = pd.crosstab(y_test, y_pred2)
FP = df_confusion.sum(axis=0) - np.diag(df_confusion)
FN = df_confusion.sum(axis=1) - np.diag(df_confusion)
TP = np.diag(df_confusion)
TN = df_confusion.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# true positive rate
TPR = TP/(TP+FN)
# true negative rate
TNR = TN/(TN+FP) 
# Precision
PPV = TP/(TP+FP)
#Recall
REC = TP/(FN+TP)
# false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# F1
F1 = 2*(PPV*REC)/(PPV+REC)

params2 = pd.Series()
params2['Recall']=REC
params2['Precision']=PPV
params2['F1 Score']=F1
params2['TPR']=TPR
params2['FPR']=FPR
params2['TNR']=TNR
params2['FNR']=FNR

df_confusion = pd.crosstab(y_test, y_pred3)
FP = df_confusion.sum(axis=0) - np.diag(df_confusion)
FN = df_confusion.sum(axis=1) - np.diag(df_confusion)
TP = np.diag(df_confusion)
TN = df_confusion.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# true positive rate
TPR = TP/(TP+FN)
# true negative rate
TNR = TN/(TN+FP) 
# Precision
PPV = TP/(TP+FP)
#Recall
REC = TP/(FN+TP)
# false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# F1
F1 = 2*(PPV*REC)/(PPV+REC)

params3 = pd.Series()
params3['Recall']=REC
params3['Precision']=PPV
params3['F1 Score']=F1
params3['TPR']=TPR
params3['FPR']=FPR
params3['TNR']=TNR
params3['FNR']=FNR

print(pd.DataFrame(data={'kNN': params, 'NB': params2, 'DT': params3}))

#Save as CSV
pd.DataFrame(data={'kNN': params, 'NB': params2, 'DT': params3}).to_csv(r'res/ex_2_perf.csv', index = True)

# e)

def roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]

thresholds = []
for i in range(10000):
    thresholds.append(i/10000)
    
x, y = roc_curve(y_test, y_proba, thresholds)
plt.plot(x, y, label ="kNN")
x, y = roc_curve(y_test, y_proba2, thresholds)
plt.plot(x, y, label ="Naive Bayes")
x, y = roc_curve(y_test, y_proba3, thresholds)
plt.plot(x, y, label ="Tree")
plt.legend()
plt.savefig(r'res/ex_2_roc.png')
plt.close()
plt.savefig(sys.stdout.buffer) 

##################
