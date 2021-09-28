import pandas as pd
import numpy as np
import math




class KNN:
    def __init__(self,n_neighbors, dist_type = 1):
        ##################
        # a)

        self.n_neighbors = n_neighbors
        self.dist_type = dist_type
        
    def dist(self, a, b):
        if self.dist_type == 2:
            #Perform euclidiean algorithm
            numb_vectors_a = a.shape[0]
            numb_vectors_b = b.shape[0]
            numb_dims = a.shape[1]
            distances = np.zeros([numb_vectors_b, numb_vectors_a])
            for i in range(numb_vectors_b):
                for j in range(numb_vectors_a):
                    distance = 0.0
                for k in range(numb_dims):
                    temp = a[j, k] - b[i]
                    distance += (temp*temp)

            distances[i, j] = math.sqrt(distance)
            return distances

        if self.dist_type == 1:
            #Perform manathan algorithm
            numb_vectors_a = a.shape[0]
            numb_vectors_b = b.shape[0]
            numb_dims = a.shape[1]
            distances = np.zeros([numb_vectors_b, numb_vectors_a])
            for i in range(numb_vectors_b):
                for j in range(numb_vectors_a):
                    distance = 0.0
                for k in range(numb_dims):
                    distances[i, j] += abs(a[j, k] - b[i])
            return distances
        

    def fit(self, X, y):
        ##################
        # b)
        
        self.X = X
        self.y = y
        

    def predict_single_prb(self,x):
        ##################
        # c)
        distance_matrix = self.dist(X, y)
        k_smallest_indices = np.argpartition(distance_matrix, self.n_neighbors-1)[:, :self.n_neighbors]
        labels = self.X[k_smallest_indices]
        
        responses = self.X[k_smallest_indices]
        output_responses = np.mean(responses, axis=1)
        
        
        #normalize
        normalizer = output_responses.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        output_responses /= normalizer
        

        return output_responses

    def predict_multiple_prb(self,X):
        ##################
        # d)
        # I already implemented with a matrix in predict_single_prb
        return self.predict_single_prb(X)



##################
# e)
#Getting the data and Normalize + split -> Outside of class SKlearn OK
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
file = 'data/bankrupt.csv'
df = pd.read_csv(file)
df2 = df.loc[:, df.columns !='Bankrupt?']
df_bank = df['Bankrupt?']
X = df2.values
y = df['Bankrupt?'].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.120234604,random_state=0)
min_max_scaler = preprocessing.MinMaxScaler() 
min_max_scaler.fit(X_train)  
X_train = min_max_scaler.transform(X_train) 
X_test = min_max_scaler.transform(X_test)


#Initiate object
my_knn5 = KNN(5)
my_knn5.fit(X_train, y_train)
my_knn10 = KNN(10)
my_knn10.fit(X_train, y_train)
my_knn20 = KNN(20)
my_knn20.fit(X_train, y_train)

#Row 27 to 29 X_test
X_test = X_test[26:29]
y_prob5 = my_knn5.predict_single_prb(X_test)
y_prob10 = my_knn10.predict_single_prb(X_test)
y_prob20 = my_knn20.predict_single_prb(X_test)

param = pd.Series()
param['Probas'] = y_prob5
param2 = pd.Series()
param2['Probas'] = y_prob10
param3 = pd.Series()
param3['Probas'] = y_prob5
pd.DataFrame(data={'kNN5': param, 'kNN10': param2, 'kNN20': param3}).to_csv(r'res/ex_3_manathan.csv', index = False)


##################

##################
# f)
my_knn5 = KNN(5,2)
my_knn5.fit(X_train, y_train)
my_knn10 = KNN(10,2)
my_knn10.fit(X_train, y_train)
my_knn20 = KNN(20,2)
my_knn20.fit(X_train, y_train)

#Row 27 to 29 X_test
X_test = X_test[26:29]
y_prob5 = my_knn5.predict_multiple_prb(X_test)
y_prob10 = my_knn10.predict_multiple_prb(X_test)
y_prob20 = my_knn20.predict_multiple_prb(X_test)

param = pd.Series()
param['Probas'] = y_prob5
param2 = pd.Series()
param2['Probas'] = y_prob10
param3 = pd.Series()
param3['Probas'] = y_prob20
pd.DataFrame(data={'kNN5': param, 'kNN10': param2, 'kNN20': param3}).to_csv(r'res/ex_3_euclid.csv', index = False)
##################

##################
# g)
my_answer =  "No, the answer doesn't change as we can see by comparing both files, the reason is because a point will be closer or further apart to another one whatever the measure of distance we use. An easy exemple would be to compare a point that is 5 meters from the origin with one that is 7 meters away. The 5 meters will be closer. And it's the same to compare a point 16 feet and one from 23 feet. The 16 feet will be closer."
with open('res/ex_3_g.txt', 'a') as t:
    t.write(my_answer)

