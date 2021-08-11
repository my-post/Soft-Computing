import pandas as pd
import numpy as np
import math
import operator
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class fuzzyKNN():
    def __init__(self, k=5, m=2, plot=False):
        self.k = k
        self.plot = plot
        self.m = m

        
        if type(self.k) != int:
            raise Exception('"k" should have type int')
        elif type(self.plot) != bool:
            raise Exception('"plot" should have type bool')
    
    # Fit data to the model        
    def fit(self, X, Y):
        
        # Check data type. X and Y must be data frame
        if type(X) != pd.core.frame.DataFrame or type(Y) != pd.core.series.Series:
            raise Exception('X should be pandas dataframe and Y should be 1D pandas series')
        
        self.X = X
        self.Y = Y
        
        self.df = X.copy()
        self.df['Y'] = Y
        
        self.dim = len(X.columns)
        self.n = len(Y)
        
        classes = list(set(Y))
        classes.sort()
        self.classes = classes
        
        self.df['membership'] = self.cal_membership()
       
    # Predict
    def predict(self, new_x):
        if type(new_x) != pd.core.frame.DataFrame:
            raise Exception('X should be pandas dataframe.')

        pred = []
        for index, x in new_x.iterrows():
            print('Predict ',index,'th sample')
            # Get k neighbor
            k_neighbor = self.get_KNN(x.to_numpy(), self.df.copy())
            
            # Get memberships
            memberships = []
            for i in self.classes:
                
                den = 0
                num = 0
                for j in range(self.k):
                    dist = k_neighbor['distance'].iloc[j]
                    if dist != 0:
                        den = den + 1 / dist**(2/(self.m-1))
                        num = num + k_neighbor.iloc[j].membership[i] * 1 / dist**(2/(self.m-1))
                    else:
                        num = k_neighbor.iloc[j].membership[i]
                        den = 1
                        break

                membership = num/den
                memberships.append(membership)
            max_class = memberships.index(max(memberships))
            pred.append(max_class)
        return pred
          
    # Get kth nearest neighbor
    def get_KNN(self, x, dataframe):
        X = self.X.iloc[:,0:self.dim].to_numpy()
        dataframe['distance'] = [np.linalg.norm(Xi-x) for Xi in X]
        dataframe.sort_values(by='distance', ascending=True, inplace=True)
        return dataframe.iloc[0:self.k]
    #Calcualtion of accuracy of model
    def AccuracyCalculator(self,results, sol):
        temp=0
        for p, q in zip(results,sol):
            if p==q:
                temp=temp+1
        #print(temp)
        print('\nClassification accuarcy')
        temp1=temp/len(results)
        temp1=temp1*100
        print(temp1)
        return

    # Calculate membership of each data w.r.t each class
    def cal_membership(self):
        memberships = []
        # calculate μ-ij, the membership of jth sample w.r.t. ith class
        for j in range(self.n):
            #print("Start Generate sample ",j," membership")
            x = self.X.iloc[j]
            y = self.Y.iloc[j]
            neighbor = self.get_KNN(x.to_numpy(), self.df.copy())
            count = neighbor['Y'].value_counts().to_dict()
            sample_membership = dict()
            for i in self.classes:
                try:
                    uij = (count[i] / self.k) * 0.49
                    if i == y:
                        uij = uij + 0.51
                    sample_membership[i] = uij
                except Exception as e:
                    sample_membership[i] = 0
            memberships.append(sample_membership)
        return memberships

if __name__ == "__main__":
    data = pd.read_csv('ionosphere.csv', header=None)
    data=data.rename(columns={data.columns[34]: 'sol'})
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'species'.
    data['sol']= label_encoder.fit_transform(data['sol'])
    X = data.iloc[:,0:34]
    Y= data['sol']
    x, test_x, y, test_y = train_test_split(X, Y, test_size=0.3)
    results=[]
    for ty in test_y:
        results.append(ty)
    classifier = fuzzyKNN(k=3, m=2)
    classifier.fit(x, y)
    sol= classifier.predict(test_x)
    print('\nActual values of test set')
    print(results)
    print('\nPredicted Values of test set')
    print(sol)
    classifier.AccuracyCalculator(results,sol)
