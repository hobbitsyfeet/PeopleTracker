from sklearn import linear_model
from scipy.stats import pearsonr

import queue
import numpy as np

class rolling_regression():
    def __init__(self, window = 10, regression="Linear"):

        if regression == "Linear":
            self.regression = linear_model.LinearRegression()
        elif regression == "Lasso":
            self.regression = linear_model.Lasso(alpha=0.1)

        self.window = window
        self.data_window = []

    def predict(self, location):
        pred = None
        if len(self.data_window) <= self.window:
            self.data_window.append(location)
        else:
            self.data_window.append(location)
            self.data_window.pop(0)

        if len(self.data_window) >= 2:
            x = []
            y = []
            for index in range(len(self.data_window)):
                x.append(self.data_window[index][0])
                y.append(self.data_window[index][1])
            x = np.array(x).reshape(-1,1)
            y = np.array(y)
            self.regression.fit(x,y)
            
            # print("slope", self.regression.coef_)
            # print("intercept" ,self.regression.intercept_)
            x = x.reshape(y.shape)
            # covariance = np.cov(x, y)
            # print(all(x))
            if len(set(x)) > 1 and len(set(y)) > 1:
                corr, _ = pearsonr(x, y)
            else:
                corr = None

            return self.regression.coef_, self.regression.intercept_, corr
        else:
            return None, None, None
        
    def get_direction(self):
        """
        True if positive(Right), False if negative (Left)
        """
        Right = True
        Up = True

        if len(self.data_window) >= 2:
            x = []
            y = []
            # print(self.data_window)
            for index in range(len(self.data_window)):
                x.append(self.data_window[index][0])
                y.append(self.data_window[index][1])
            
        #     if max(x) - min(x)  >= 0:
        #         Right = True
        #     else:
        #         Right = False

        #     if max(y) - min(y) <= 0:
        #         Up = True
        #     else:
        #         Up = False
        # print(max(x) - min(x), max(y) - min(y), Up, Right)

        return (Right, Up)

def distance_2d(p1, p2):
    print(p1,p2)
    return abs( ((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)**(1/2) )

if __name__ == "__main__":
    locations = np.array([
       [ 0.,  0.],
       [ 1.,  3.],
       [ 2.,  5.],
       [ 3.,  7.],
       [ 4.,  9.],
       [ 5.,  16.],
       [ 7.,  1.],
       [ 8.,  35.],
       [ 9.,  52.],
       [ 10.,  1135.],
       [ 15.,  123.],
       [ 20.,  2.]
    ])

    reg = rolling_regression()


    

    for loc in locations:
        pred = reg.predict(loc)
        print(loc, pred)