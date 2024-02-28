import numpy as np

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from filterpy.kalman import ExtendedKalmanFilter

class KalmanPred():
    '''!
    @cite filterpy
    <a href="">https://github.com/rlabbe/filterpy</a>
    '''


    def __init__(self, white=False):
        self.reset(white)

    def reset(self, white=False):

        self.previous_location = np.array((float('inf'),float('inf')))

        self.k_filter = KalmanFilter(dim_x=4, dim_z=4)

        # initial state
        self.k_filter.F = np.array([[1.,    0.,     0.3,    0 ],
                                [0.,    1.,     0,      0.3],
                                [0,     0,      1,      0],
                                [0,     0,      0,      1]
                                ])    # state transition matrix
        # Measurement function
        self.k_filter.H = np.array([
                                [1.,    0.,     1,    0 ],
                                [0.,    1.,     0,      1],
                                [0,     0,      0,      0],
                                [0,     0,      0,      0]
                                ])
    
        # state transition matrix    
        self.k_filter.P *= 1000.                 # covariance matrix
        # self.k_filter.R = 0.1                     # state uncertainty
        self.k_filter.R = np.array([
                                [0.1,   0.,     0,        0],
                                [0.,    0.1,    0,        0],
                                [0,     0,      0.1,      0],
                                [0,     0,      0,      0.1]
                                ])
        if white:
            self.k_filter.R = np.array([
                        [0.1,   0.,     0,        0],
                        [0.,    0.1,    0,        0],
                        [0,     0,      10,      0],
                        [0,     0,      0,      10]
                        ])
        else:
            self.k_filter.Q = [[  0,  0,  0,  0],
                                [ 0,  0,  0,  0],
                                [ 0,  0,  0.1,0],
                                [ 0,  0,  0, 0.1]]


    def predict(self, location=None):

        # Indicates user set a new location or starting a new tracker
        if self.previous_location[0] == float('inf'):
            
            self.previous_location = np.array(location)

            self.k_filter.x = np.array([[location[0]],[location[1]],
                                    [0],[0]])     # state transition matrix


        # We are predicting new data and updating the model
        elif location is not None:
            self.previous_location = np.array(location)
            velocity_x = self.previous_location[0] - location[0]
            velocity_y = self.previous_location[1] - location[1]

            self.k_filter.predict()
            self.k_filter.update(np.array([[location[0]],[location[1]],
                                    [velocity_x],[velocity_y]]))
        
        #We are further predicting future steps
        else:
            self.k_filter.predict()
            
        return self.k_filter.x
    
# class KalmanPredBox():
#     def __init__(self):
#         # np.array([
#         #                         [1,  0,    0,   0,    0,  0,  0,  0],
#         #                         [0,  1,    0,   0,    0,  0,  0,  0],
#         #                         [0,  0,    1,   0,    0,  0,  0,  0],
#         #                         [0,  0,    0,   1,    0,  0,  0,  0],
#         #                         [0,  0,    0,   0,    1,  0,  0,  0],
#         #                         [0,  0,    0,   0,    0,  1,  0,  0],
#         #                         [0,  0,    0,   0,    0,  0,  1,  0],
#         #                         [0,  0,    0,   0,    0,  0,  0,  1]
#         #                         ])    # state transition matrix
#         # # Measurement function

#         previous_location = np.array((0,0,0,0))

#         self.k_filter = KalmanFilter(dim_x=8, dim_z=8)

#         # initial state
#         self.k_filter.F = np.array([
#                                 [1,  0,    0.3,     0,      0,      0,      0,      0  ],
#                                 [0,  1,    0,       0.3,    0,      0,      0,      0  ],
#                                 [0,  0,    1,       0,      0.3,    0,      0,      0  ],
#                                 [0,  0,    0,       1,      0,      0.3,    0,      0  ],
#                                 [0,  0,    0,       0,      1,      0,      0.3,    0  ],
#                                 [0,  0,    0,       0,      0,      1,      0,      0.3],
#                                 [0,  0,    0,       0,      0,      0,      1,      0  ],
#                                 [0,  0,    0,       0,      0,      0,      0,      1  ]
#                                 ])    # state transition matrix
#         # Measurement function
#         self.k_filter.H = np.array([
#                                 [1,  0,    1,   0,    1,  0,  1,  0],
#                                 [0,  1,    0,   1,    0,  1,  0,  1],
#                                 [0,  0,    0,   0,    0,  0,  0,  0],
#                                 [0,  0,    0,   0,    0,  0,  0,  0],
#                                 [0,  0,    0,   0,    0,  0,  0,  0],
#                                 [0,  0,    0,   0,    0,  0,  0,  0],
#                                 [0,  0,    0,   0,    0,  0,  0,  0],
#                                 [0,  0,    0,   0,    0,  0,  0,  0]
#                                 ])
    
#         # state transition matrix    
#         self.k_filter.P *= 1000.                 # covariance matrix
#         self.k_filter.R = 0.1                     # state uncertainty
#         self.k_filter.R = np.array([
#                                 [0.1,  0,    0,   0,    0,  0,  0,  0],
#                                 [0,  0.1,    0,   0,    0,  0,  0,  0],
#                                 [0,  0,    0.1,   0,    0,  0,  0,  0],
#                                 [0,  0,    0,   0.1,    0,  0,  0,  0],
#                                 [0,  0,    0,   0,    0.1,  0,  0,  0],
#                                 [0,  0,    0,   0,    0,  0.1,  0,  0],
#                                 [0,  0,    0,   0,    0,  0,  0.1,  0],
#                                 [0,  0,    0,   0,    0,  0,  0,  0.1]
#                                 ])

#         self.k_filter.Q = [[  0,  0,  0,  0],
#                             [ 0,  0,  0,  0],
#                             [ 0,  0,  0.1,0],
#                             [ 0,  0,  0, 0.1]]
#         # self.k_filter.Q = Q_discrete_white_noise(4, dt=1, var=100) # process uncertainty
#         print(self.k_filter.Q)


    # def predict(self, location=None):
    #     if location is not None:
    #         self.previous_location = np.array(location)
    #         velocity_x = self.previous_location[0] - location[0]
    #         velocity_y = self.previous_location[1] - location[1]

    #         self.k_filter.predict()
    #         self.k_filter.update(np.array([[location[0]],[location[1]],
    #                                 [velocity_x],[velocity_y]]))
    #     else:
    #         self.k_filter.predict()
            
    #     return self.k_filter.x
if __name__ == "__main__":

    locations = np.array([
       [ 0.,  0.],
       [ 1.,  1.],
       [ 2.,  2.],
       [ 3.,  3.],
       [ 4.,  4.],
       [ 5.,  5.],
       [ 7.,  7.],
       [ 8.,  8.],
       [ 9.,  9.],
       [ 10.,  10.],
       [ 15.,  15.],
       [ 20.,  20.]
    ])



    my_filter = KalmanPred()

   
    # while True:
    for index, loc in enumerate(locations):
        if index > 1:
            test = my_filter.predict(loc)
            print(test)

        # print(x)


