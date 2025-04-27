# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 09:34:01 2024

@author: akaa
"""

import os as os
import numpy as np
# =============================================================================
# 1. Create a 3x3 NumPy array filled with random integers and print its shape and data
#  type.
# =============================================================================

a = np.random.randint(0, 100, (3, 3))
a
print("the shape is ", a.shape)
print('data type', a.dtype)

print('\n')

# =============================================================================
# 2. Convert a Python list of lists into a NumPy ndarray and print the resulting array.
# =============================================================================
a = [[1, 2], [3, 4], [5, 6]]
arr = np.array(a)
print('the array is', arr)
print('\n')
# =============================================================================
# 3. Create a 5x5 NumPy array of zeros and change the element at position (2, 2) to 5.
# =============================================================================
a = np.zeros((5, 5))
print('the 5x5 arary is\n', a)
print('\n')
a[2:3, 2:3] = 5
print(a)
print('\n')
# =============================================================================
# 4. Create a 4x4 identity matrix using NumPy.
# =============================================================================
a = np.identity(4)
print('the identity matrix is', a)
print('\n')

# =============================================================================
# 5. Create a NumPy array of shape (3, 3, 3) filled with random values and print the array.
# =============================================================================

a = np.random.randint(1, 100, (3, 3, 3))
print('the random array is', a)

# =============================================================================
# 6. Create a 10x10 NumPy array of ones and set the boundary values to zero (except the
# =============================================================================
a = np.ones((10, 10), dtype=int)
a
a[0, 0] = 0
a[0, -1] = 0
a[9, 9] = 0
a[9, 0] = 0
print(a)
print('\n')

# =============================================================================
# 7. Generate a NumPy array of 50 linearly spaced numbers between 0 and 100.
# =============================================================================

a = np.linspace(1, 100, 50, retstep=True)
print(a)
print('\n')


# =============================================================================
# 8. Convert a NumPy array of integers into a float array.
# =============================================================================
a = [1, 2, 3, 4, 5]
a
b = np.array(a, dtype=float)
print("the floart array is", b)

print('\n')

# =============================================================================
# 9. Create a NumPy array of size 20 and reshape it to (5, 4).
# =============================================================================

a = np.arange(20)
b = np.reshape(a, (5, 4))
print("the reshaped array is", b)
print('\n')


# =============================================================================
# 10. Create a 2D NumPy array of shape (5, 5) and calculate the sum of all elements.
# =============================================================================

a = np.arange(1, 26)
b = np.reshape(a, (5, 5))
summ = b.sum()
print('the sum of the matrix is ', summ)
print('\n')

# =============================================================================
# 11. Create a 6x6 NumPy array and extract the subarray of the first 3 rows and 3 columns.
# =============================================================================

a = np.arange(1, 37)
b = np.reshape(a, (6, 6))
b
print('the first 3 row and 3 col is\n', b[0:3, 0:3])
print('\n')

# =============================================================================
# 12. Create a 1D NumPy array of size 10 and extract every other element starting from
# index 0.
# =============================================================================

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(a[0:])
print('\n')
# =============================================================================
# 13. Reverse the elements of a 1D NumPy array using slicing.

# =============================================================================
a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(a[::-1])
print('\n')
# =============================================================================
# 14. Create a 5x5 array and replace all the odd elements with -1.
# =============================================================================
a = np.arange(1, 26)
b = np.reshape(a, (5, 5))
b
b[b % 2 != 0] = -1
print(b)
print('\n')


# =============================================================================
# 15. Select the elements in an array that are greater than a given threshold value.
# =============================================================================

a = 15
b = np.random.randint(1, 26, (2, 5))
print(b[b > a])


# =============================================================================
# 17. Create a 3D array and extract a subarray using multidimensional slicing.
# =============================================================================


a = np.arange(1, 28)
b = np.reshape(a, (3, 3, 3))
print(b)
print('\n')

print(b[0:3][..., 1][1, ...])


# =============================================================================
# 18. Modify the last row of a 4x4 NumPy array to contain all ones.
# =============================================================================


a = np.arange(16)
b = np.reshape(a, (4, 4))
b
b[2:, 2:] = 1
print(b)
print('\n')

# =============================================================================
# 19. Extract the diagonal elements from a 2D square NumPy array.
# ============================================================================
a = np.array([[1, 2], [4, 5]])
print(np.diag(a))

# =============================================================================
# 20. Using Boolean indexing, extract all elements of an array that are even numbers.
# =============================================================================

a = np.array([1, 2, 3, 4, 5, 6, 7, 8])

b = a % 2 == 0
print(b)
print('the even array is', a[b])

print('\n')
# =============================================================================
# 21. Create a 2D NumPy array and transpose it.
# ============================================================================

a = np.arange(4)
b = np.reshape(a, (2, 2))
print(np.transpose(b))

#
# =============================================================================
# 22. Create a matrix and compare its transpose with the original matrix.
# =============================================================================

a = np.arange(4)
b = np.reshape(a, (2, 2))
print('comparing the original matrix with transpose matrix\n',
      b, '\t', np.transpose(b))

print('\n')
# =============================================================================
# 23. Create a 3x3 matrix and swap the rows with columns using transposition.
# =============================================================================

a = np.arange(9)
b = np.reshape(a, (3, 3))
print(np.transpose(b))


print('\n')
# =============================================================================
# 24. Transpose a NumPy array of shape (4, 2) and reshape it to (2, 4).

# =============================================================================


a = np.arange(8)
b = np.reshape(a, (4, 2))

c = np.transpose(b)
print(c)
print('\n')

d = c.reshape(2, 4)
print(d)
print('\n')


# =============================================================================
# 25. Create a 5x5 matrix, swap the first and second rows, and print the result.
# =============================================================================

a = np.arange(25)
b = np.reshape(a, (5, 5))
print(b)
b[[0, 1]] = b[[1, 0]]
print(b)
print('\n')

# =============================================================================
# Write a function that accepts a NumPy array and returns its transpose using
# np.transpose().
# =============================================================================


def Transpose(x):
    return np.transpose(x)


a = np.arange(9)
b = np.reshape(a, (3, 3))
print('the orginal array is :\n', b)
c = Transpose(b)
print('the transpose array using fuction is\n', c)
print('\n')
print('\n')


# =============================================================================
# 27. Create a NumPy array and save it to a .npy file using np.save().
# =============================================================================

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
np.save('hello.npy', a)
print('file saved')
print('\n')

# =============================================================================
# 28. Create a random array and save it as a .csv file using np.savetxt().
# =============================================================================

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
np.savetxt('hello.csv', a)
print('file saved')
print('\n')


# =============================================================================
# 29. Load a NumPy array from a .npy file using np.load().
# =============================================================================

a = np.load("hello.npy")
print('the loadde npy file contains:', a)
print('\n')


# =============================================================================
# 30. Save multiple NumPy arrays into a single .npz file and then load them.
# =============================================================================

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.array([11, 22, 33, 44, 55, 66, 77, 88, 99])
np.savez("multiple_arr.npz", arr=a, arr1=b)
c = np.load("multiple_arr.npz")
print("the loaded multiple array is:")
print(c['arr'])
print(c['arr1'])
print('\n')

# =============================================================================
# 31. Create a 3x3 matrix and save it to a text file, then reload it and print the contents.
# =============================================================================

a = np.arange(9)
b = np.reshape(a, (3, 3))
c = np.matrix(b)

np.savetxt('matri.txt', c)
a = np.loadtxt('matri.txt', dtype=int)
print('the matrix is:', a)

print('\n')
# =============================================================================
# 32. Load a .csv file into a NumPy array and print the first 5 rows.
# =============================================================================

# a=np.loadtxt('Research_Project _March.csv',dtype=int)


# =============================================================================
# 33. Create two NumPy arrays, save them in binary format, and then reload and print
# them.
# =============================================================================


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
b = np.array([11, 22, 33, 44, 55, 66, 77, 88, 99])
np.save('multi_arr1', a)
np.save('mul_arr2', b)
d = np.load('multi_arr1.npy')
d1 = np.load('mul_arr2.npy')
print(d)
print(d1)
print('\n')
# =============================================================================
# 34. Save a NumPy array in compressed format using np.savez_compressed() and reload
# it.
# =============================================================================

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
np.savez_compressed('compress', c=a)

b = np.load('compress.npz')
print(b['c'])
print('\n')

# =============================================================================
# 35. Generate a large NumPy array, save it to a file, and measure the file size.
# =============================================================================
a = np.random.rand(1000, 1000)
np.savetxt('large_array', a)
size = os.path.getsize('large_array')
print(size)
print('\n')
# =============================================================================
# 36. Write a function that saves an array to a file and then loads it back.
# =============================================================================


def SaveLoad(name, arr):
    np.savetxt(name, arr, fmt='%d')

    a = np.loadtxt(f'{name}', dtype=int)
    return a


a = np.array([[1, 2], [2, 3], [3, 4]])
defi = SaveLoad('ary', a)
print(defi)
print('\n')


# =============================================================================
# 37. Create a NumPy array of random values and apply the np.sqrt() function to compute
# the square root of each element.
# =============================================================================
a = np.random.randint(1, 100, 10)
print("generated values is\n", a)
print('the square root of generated valuse is:\n', np.sqrt(a))
print('\n')
# =============================================================================
# 38. Generate a NumPy array of 100 random values and apply np.sin() to compute the sine
#  of each element.
# =============================================================================

a = np.random.randint(1, 100, 100)
print('the generated valuse is:\n', a)
print(np.sin(a*np.pi/180))
print('\n')

# =============================================================================
# 39. Given a NumPy array of floating-point numbers, apply the np.ceil() function to round
# each value up.
# =============================================================================

a = np.array([1.2, 2.7, 3.5, 4.1, 5.9])
print("given array", a)
print('array after float function', np.floor(a))
print('\n')

# =============================================================================
# 40. Create a NumPy array and apply np.exp() to compute the exponential of each
# element.
# =============================================================================

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print("the expo values are ", np.exp(a))
print('\n')

# =============================================================================
# 41. Create two arrays and use np.maximum() to compute the element-wise maximum of
# the two arrays.
# =============================================================================

a = np.array([1, 25, 4, 24, 76, 85, 3])
b = np.array([5, 42, 8, 64, 36, 84, 2])
print(np.maximum(a, b))
print('\n')

# =============================================================================
# 42. Create a 2D array of random integers and apply np.abs() to get the absolute values of
# all elements.
# =============================================================================
a = np.random.randint(-100, 100, (3, 3))
print('generated values are:\n', a)
print('\n')
print(np.abs(a))
print('\n')
# =============================================================================
# 43. Apply np.log() to a NumPy array of positive random values and print the results.
# =============================================================================

a = np.random.randint(1, 100, 10)
print('generated values are:\n', a)
print('\n')
print('log values are :\n', np.log(a))
print('\n')
# =============================================================================
# 44. Create a NumPy array of random floats and apply np.round() to round each value to 2
# decimal places.
# =============================================================================

a = np.random.rand(3,3,3)*10
print('generated values are:\n', a)
print('\n')
b = np.round(a, decimals=-1)
print('rounded values are :\n', b)
print('\n')
# =============================================================================
# 45. Use np.add() to add two NumPy arrays element-wise.
# =============================================================================

a = np.array([1, 25, 4, 24, 76, 85, 3])
b = np.array([5, 42, 8, 64, 36, 84, 2])
np.sum(a)
print('the added values are', np.add(a, b))
print('\n')

# =============================================================================
# 46.Apply np.power() to raise each element in an array to a given power.
# =============================================================================

a = np.array([2, 4, 6, 8, 10])
print("the original is:", a, 'the powered values are', np.power(a, 2))
print('\n')

# =============================================================================
# 47. Create a NumPy array of random integers and compute its mean using np.mean().
# =============================================================================
a = np.random.randint(1, 100, 10)
print("the orginal array is:\n", a, 'the mean of the array is:', np.mean(a))
print('\n')
# =============================================================================
# 48. Generate a 2D array and compute the sum of all its elements using np.sum().
# =============================================================================

a = np.array([[1, 2, 2],
              [3, 4, 5],
              [6, 5, 4],
              [1, 2, 3]])
print(np.sum(a))
print('\n')

# =============================================================================
# 50. Create a NumPy array and compute the median of its values using np.median().
# =============================================================================

a = np.array([1, 5, 7, 9, 3, 5, 7])
print('the median value is:', np.median(a))
print('\n')

# =============================================================================
# 51. Create a 3x3 matrix and compute the trace (sum of diagonal elements) using
# np.trace().
# =============================================================================

a = np.arange(9)
a = np.reshape(a, (3, 3))
a = np.asmatrix(a)
print("the matrix is:\n", a)
print("the sum of diagonal elements are:", np.trace(a))
print('\n')


# =============================================================================
# 52. Create a 2D array of random values and compute the row-wise and column-wise
# sums.
# =============================================================================

a = np.array([[1, 2], [5, 6]])
print("'2d array is:\n", a)
print('the row wise sum is ', np.sum(a, axis=1))
print('the column wise sum is ', np.sum(a, axis=0))


# =============================================================================
# 53. Given a 2D array, calculate the minimum and maximum values along both axes using
# np.min() and np.max().
# =============================================================================

a = np.array([[1, 2, 3], [5, 6, 7]])
print("'2d array is:\n", a)
print('the row wise min is ', np.min(a, axis=1))
print('the row wise max is ', np.max(a, axis=1))
print('the column wise min is ', np.min(a, axis=0))
print('the column wise max is ', np.max(a, axis=0))
print('\n')

# =============================================================================
# 54. Compute the variance of a NumPy array using np.var().
# =============================================================================

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
print('the array is', a)
print('the varianc of the array', np.var(a))








df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
s = pd.Series([1, 2], index=['A', 'B'])
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
s = pd.Series([1, 2], index=['A', 'B'])
df.mul(s, axis=0)


import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

df['newcol']=df['B'].map(lambda x: x+2)

df

import pandas as pd
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 , 4,1,2],
'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2014,2015,2017],
'Points':[876,789,863,673,741,812,756,788,694,701,804,690]}
df = pd.DataFrame(ipl_data)
grouped = df.groupby('Year')




import pandas as pd
from datetime import datetime
# Sample data
data = {
'EmployeeID': [1, 2, 3, 3],
'Name': ['alice johnson', 'bob smith', 'charlie brown','charlie brown'],
'Department': ['HR', 'IT', 'Finance','Finance'],
'Salary': [60000, 75000, 50000, 56000],
'JoinDate': ['2020-02-15', '2018-08-10', '2021-05-21','2021-05-21'],
'Gender': ['Female', 'Male', None, None],
'YearsAtCompany': [None, None, None, None]
}
df = pd.DataFrame(data).fillna



