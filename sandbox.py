import scipy.fftpack as fftp
import torch
import torch.nn as nn
import numpy as np



"""
test = [[1,2,3], [4,5,6]]
test_1 = [1,2,3,4,5,6,0, 0, 0, 0, 0]
test_2 = [1,3,3,4,5,6,0, 0, 1, 0, 5]
test_3 = [1, -2, 3, 0]

print(fftp.idct(test_1, 3, 10))
print(fftp.idct(test_2, 3, 10))
print(fftp.idct(test_3, 2, 30, norm='ortho'))
print(fftp.dct([1,1,1,1,1,1,1], norm='ortho'))
print(fftp.idct(fftp.dct([1,1,1,1,1],n=10, norm='ortho'),n=10, norm='ortho'))
print(fftp.idct([0.5, -0.5, 0,  0, 0 ], n = 10))
#print(fftp.idct([1.2780193, 1.6505744,   1.22829917,  0.66765111,  0.12909944, -0.24940098], 2, 30, norm='ortho'))
#print(fftp.idct([1, 2, 3, 4], 3, 10))
#print(fftp.idct([1, 2, 3, 4, 1, 0], 3, 20))
"""

arr = [[1,2,3],[4,5,6]]
np_arr = np.array(arr)
print(np_arr.T.T)
test_torch = torch.rand((3,3))
linear_layer = nn.Linear(10, 5)

arr = [[1,2],[3,4]]
np_arr = np.array(arr)

print(test_torch.type())
for name, parameter in linear_layer.named_parameters():
    print(parameter)
    print(parameter.type())
print(torch.from_numpy(np_ar).type())
