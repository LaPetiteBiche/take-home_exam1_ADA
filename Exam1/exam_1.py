import pandas as pd
import  numpy as np
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


##################
# a)

file = 'data/StudentsPerformance.csv'
df = pd.read_csv(file)

#Convert Binary
df['gender'] = (df['gender']=='male')*1
df['lunch'] = (df['lunch']=='standard')*1
df['test preparation course'] = (df['test preparation course']=='completed')*1

#Convert Category
dum = pd.get_dummies(df['race/ethnicity'])
df = pd.concat([df, dum], axis=1)
df['race/ethnicity'] = df['race/ethnicity'].astype('category')
df['race/ethnicity'] = df['race/ethnicity'].cat.codes
dum2 = pd.get_dummies(df['parental level of education'])
df = pd.concat([df, dum2], axis=1)
df['parental level of education'] = df['parental level of education'].astype('category')
df['parental level of education'] = df['parental level of education'].cat.codes

#Average math, reading an writing score
y = df[['math score', 'writing score', 'reading score']]
mean = y.mean(axis = 1)
df = pd.concat([df, mean], axis=1)
df.rename(columns={0:'y'}, inplace=True)

#Save as CSV

df[:5].to_csv(r'res/ex_1_a.csv', index = False)

print (df.head())


##################

##################
# b)

#First Graph
fig, axs = plt.subplots(3)
axs[0].scatter(df['math score'],df['reading score'])
axs[0].set(xlabel='Math')
axs[0].set(ylabel='Reading')
axs[1].scatter(df['reading score'],df['writing score'])
axs[1].set(xlabel='Reading')
axs[1].set(ylabel='Writing')
axs[2].scatter(df['writing score'],df['math score'])
axs[2].set(xlabel='Writing')
axs[2].set(ylabel='Math')
fig.tight_layout(pad=2)
fig.suptitle('Correlation between scores')
plt.savefig(r'res/ex_1_correlation_x.png')
plt.close()
plt.savefig(sys.stdout.buffer)


#Second Graph
fig, axs = plt.subplots(5)

ind = df['gender']==1
axs[0].scatter(df.loc[ind,'gender'],df.loc[ind,'y'], label='Male')
ind = df['gender']!=1
axs[0].scatter(df.loc[ind,'gender'],df.loc[ind,'y'], label='Female')

ind = df['group A']==1
axs[1].scatter(df.loc[ind,'race/ethnicity'],df.loc[ind,'y'], label='Group A')
ind = df['group B']==1
axs[1].scatter(df.loc[ind,'race/ethnicity'],df.loc[ind,'y'], label='Group B')
ind = df['group C']==1
axs[1].scatter(df.loc[ind,'race/ethnicity'],df.loc[ind,'y'], label='Group C')
ind = df['group D']==1
axs[1].scatter(df.loc[ind,'race/ethnicity'],df.loc[ind,'y'], label='Group D')
ind = df['group E']==1
axs[1].scatter(df.loc[ind,'race/ethnicity'],df.loc[ind,'y'], label='Group E')

ind = df["associate's degree"]==1
axs[2].scatter(df.loc[ind,'parental level of education'],df.loc[ind,'y'], label="associate's degree")
ind = df["bachelor's degree"]==1
axs[2].scatter(df.loc[ind,'parental level of education'],df.loc[ind,'y'], label="bachelor's degree")
ind = df['high school']==1
axs[2].scatter(df.loc[ind,'parental level of education'],df.loc[ind,'y'], label='high school')
ind = df["master's degree"]==1
axs[2].scatter(df.loc[ind,'parental level of education'],df.loc[ind,'y'], label="master's degree")
ind = df['some college']==1
axs[2].scatter(df.loc[ind,'parental level of education'],df.loc[ind,'y'], label='some college')
ind = df['some high school']==1
axs[2].scatter(df.loc[ind,'parental level of education'],df.loc[ind,'y'], label='some high school')

ind = df['lunch']==1
axs[3].scatter(df.loc[ind,'lunch'],df.loc[ind,'y'], label='Lunch Standard')
ind = df['lunch']!=1
axs[3].scatter(df.loc[ind,'lunch'],df.loc[ind,'y'], label='Lunch Free/Reduced')

ind = df['test preparation course']==1
axs[4].scatter(df.loc[ind,'test preparation course'],df.loc[ind,'y'], label='Test Completed')
ind = df['test preparation course']!=1
axs[4].scatter(df.loc[ind,'test preparation course'],df.loc[ind,'y'], label='None')

fig.tight_layout(pad=2)
axs[0].legend()
axs[1].legend()
box = axs[1].get_position()
axs[1].set_position([box.x0, box.y0, box.width*0.65, box.height])
legend_x = 1
legend_y = 0.5
axs[1].legend(bbox_to_anchor=(legend_x, legend_y))
box = axs[2].get_position()
axs[2].set_position([box.x0, box.y0, box.width*0.65, box.height])
axs[2].legend(bbox_to_anchor=(legend_x, legend_y))
axs[3].legend()
axs[4].legend()
fig.set_size_inches(10, 15)
fig.suptitle('Correlation with score mean')
plt.savefig(r'res/ex_1_correlation_xy.png')
plt.close()
plt.savefig(sys.stdout.buffer)

#Third graph
fig, axs = plt.subplots(4)
axs[0].hist(df['math score'], bins=100, label = "Math Score")
axs[0].legend()
axs[1].hist(df['reading score'], bins=100, label = "Reading Score")
axs[1].legend()
axs[2].hist(df['writing score'], bins=100, label = "Writing Score")
axs[2].legend()
axs[3].hist(df['y'], bins=100, label = "Mean Score (y)")
axs[3].legend()
fig.set_size_inches(10, 15)
fig.suptitle('Distribution over variable of interest')
plt.savefig(r'res/ex_1_dist.png')
plt.close()
plt.savefig(sys.stdout.buffer)

##################

##################
# c)

from sklearn import metrics, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Omitting a group and parent school to avoid collinearity problem

df2 = df[['gender','lunch','test preparation course','group A','group B', 'group C','group D',"associate's degree","bachelor's degree","high school","master's degree","some college"]]
X = df2.values
y = df['y'].values

#Split into training and testing 70%, 30% and run regression Linear and RF
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

model = LinearRegression()
model = model.fit(X=X_train, y=y_train)
params = pd.Series()

model2 = RandomForestRegressor(max_depth = 2, n_estimators = 100, random_state = 0)
model2 = model2.fit(X=X_train, y=y_train)
params2 = pd.Series()

#Measure performance on test sample
print(model.score(X_test, y_test))
print(model2.score(X_test, y_test))

#Measure mean absolute error, mean square error, and R^2
r_squared = metrics.r2_score(y_test, model.predict(X_test))
mean_absolute_error = metrics.mean_absolute_error(y_test, model.predict(X_test))
mean_squared_error = metrics.mean_squared_error(y_test, model.predict(X_test))
params['r_squared'] = r_squared
params['mean_absolute_error'] = mean_absolute_error
params['mean_squared_error'] = mean_squared_error

y_pred = model2.predict(X_test)
r_squared2 = metrics.r2_score(y_test, y_pred)
mean_absolute_error2 = metrics.mean_absolute_error(y_test, y_pred)
mean_squared_error2 = metrics.mean_squared_error(y_test, y_pred)
params2['r_squared'] = r_squared2
params2['mean_absolute_error'] = mean_absolute_error2
params2['mean_squared_error'] = mean_squared_error2

print(pd.DataFrame(data={'y-LR': params, 'y-RF': params2}))

#Save as CSV
pd.DataFrame(data={'y-LR': params, 'y-RF': params2}).to_csv(r'res/ex_1_c_oos.csv', index = False)

#Cross Validation
kf = model_selection.KFold(n_splits=3, shuffle=True)
mses = []
maes = []
rs = []

for i in range(100):
    for train_index, test_index in kf.split(X):
        #Split into training and test data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #Random Forest
        reg = RandomForestRegressor(max_depth = 2, n_estimators = 100, random_state = 0)
        reg.fit(X_train,y_train)

        # MSE
        mse = sum((y_test - reg.predict(X_test))**2.0)/len(y_test)
        mses.append(mse)

        # MAE
        mae = metrics.mean_absolute_error(y_test, reg.predict(X_test))
        maes.append(mae)

        # R^2
        r = metrics.r2_score(y_test, reg.predict(X_test))
        rs.append(r)


#MSE + MAE + R^2 average
results = pd.Series()
results['MSE (Average)'] = np.mean(mses, axis = 0)
results['MSE std'] = np.std(mses, axis = 0)
results['MSE min'] = np.min(mses, axis = 0)
results['MSE max'] = np.max(mses, axis = 0)
results['MSE 5%'] = np.percentile(mses, 5, axis = 0)
results['MSE 99%'] = np.percentile(mses, 99, axis = 0)
results['MSE median'] = np.median(mses, axis = 0)

results['MAE (Average)'] = np.mean(maes, axis = 0)
results['MAE std'] = np.std(maes, axis = 0)
results['MAE min'] = np.min(maes, axis = 0)
results['MAE max'] = np.max(maes, axis = 0)
results['MAE 5%'] = np.percentile(maes, 5, axis = 0)
results['MAE 99%'] = np.percentile(maes, 99, axis = 0)
results['MAE median'] = np.median(maes, axis = 0)

results['R2 (Average)'] = np.mean(rs, axis = 0)
results['R2 std'] = np.std(rs, axis = 0)
results['R2 min'] = np.min(rs, axis = 0)
results['R2 max'] = np.max(rs, axis = 0)
results['R2 5%'] = np.percentile(rs, 5, axis = 0)
results['R2 99%'] = np.percentile(rs, 99, axis = 0)
results['R2 median'] = np.median(rs, axis = 0)
print(results)

#Save as CSV
results.to_csv(r'res/ex_1_c_oos_CV.csv', index = True)
##################

