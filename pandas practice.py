# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:44:32 2024

@author: Rohan
"""

import pandas as pd
import numpy as np


# =============================================================================
# 1. Create a pandas Series from a list of numbers and display it.
# =============================================================================

a = pd.Series([1, 2, 3, 4, 5, 6, 7])
print(a)
print('\n')


# =============================================================================
# 2. Create a pandas Series from a Python dictionary, where keys represent index values
# and values represent data.
# =============================================================================

a = {1: 'roha', 2: 'rohann', 3: 'ro', 4: 'a', 5: 'b'}
b = pd.Series(a)
print(b)
print('\n')


# =============================================================================
# 3. Given a pandas Series of exam scores, calculate the mean, median, and mode.
# =============================================================================
a = pd.Series({'math': 60, 'science': 70, 'english': 80, 'economics': 80})
print(a)
mean = a.mean()
median = a.median()
mode = a.mode()
print('the mean is', mean)
print('the meidan is', median)
print('the mode is', mode)
print('\n')


# =============================================================================
#  4. Create a DataFrame from a dictionary of lists and display the first five rows.
# =============================================================================


a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22]})
print("first 5 rows are:\n", a.head())
print('\n')


# =============================================================================
# 5. Create a DataFrame with custom row and column labels using numpy arrays.
# =============================================================================
a = np.random.randint(1, 45, (3, 3))
colm = np.array(['clo1', 'col2', 'col3'])
row = np.array(['row1', 'row2', 'row3'])
df = pd.DataFrame(a, index=row, columns=colm)
print('the custom dataframe is "\n', df)
print('\n')
print('\n')


# =============================================================================
# 6. Create a DataFrame where one of the columns is a pandas Series, and other columns
# are lists.
# =============================================================================

a = pd.Series([1, 2, 3, 4, 5])
b = [6, 7, 8, 9, 10]
c = [11, 22, 33, 44, 55]
df = pd.DataFrame({'series': a,
                  'list1': b,
                   'list2': c})
print(df)
print('\n')
print('\n')


# =============================================================================
# 7. Convert a pandas Series to a DataFrame with a custom column name.
# =============================================================================

a = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
b = pd.DataFrame(a, columns=['numbers'])
print(b)
print('\n')
print('\n')


# =============================================================================
# 8. Compare two Series element-wise and print the values that are equal in both.
# =============================================================================

a = pd.Series([6, 7, 8, 9, 10])
b = pd.Series([6, 22, 9, 44, 10])
print(a[a == b])
print('\n')
print('\n')


# =============================================================================
# 9. Create a DataFrame with NaN values and fill them with the mean of each column.
# =============================================================================
a = pd.DataFrame({'a': [10, 20, np.nan, 40, 50],
                  'b': [5, np.nan, 15, 20, 25],
                 'c': [np.nan, 2, 4, np.nan, 8]
                  })
print('the dataframe with na values:\n', a)
print('\n')
# Here b is jsut a random variable
df_mean = a.apply(lambda b: b.fillna(b.mean()))
print("fillinf the mean\n", df_mean)
print('\n')
print('\n')

# # =============================================================================
# 10. Create a DataFrame from a nested list, assign column names, and add a new column
# to the DataFrame.
# =============================================================================
a = [[1, 2, 3],
     [3, 4, 5],
     [5, 6, 7]]
b = pd.DataFrame(a, columns=["1st", '2nd', '3rd'])
print('data frame with nested lsit is:\n', b)
b['4th'] = [7, 8, 9]
print('\n')
print('adding the 4th column we will get:\n ', b)
print('\n')
print('\n')


# =============================================================================
# 11. Create a DataFrame with date ranges as the index and random numbers as values.
# =============================================================================

a = np.random.randint(1, 10, 5)

dat = pd.date_range('09/29/2024', periods=5)
print(pd.DataFrame(a, index=dat, columns=["ro"]))
print('\n')
print('\n')


# =============================================================================
# 12. Generate a DataFrame from a dictionary where the keys are column names and values
# are lists of numbers.
# =============================================================================

a = pd.DataFrame({'rohhh': [1, 2, 3, 4, 5], 'rohn': [6, 7, 8, 9, 10]})
print(a)
print('\n')
print('\n')


# =============================================================================
# 13. Create a DataFrame from two pandas Series as two columns.
# =============================================================================
a = pd.Series(['rohan', 'rohh', 'rhnn'])
b = pd.Series(['231BCADA31', '231', 'bcada'])
df = pd.DataFrame({'names': a,
                   'roll': b})
print(df)

print('\n')
print('\n')


# =============================================================================
# 14. Create a DataFrame with multiple indexes (MultiIndex) and retrieve data from it.
# =============================================================================


a = [['grp1', 'grp1', 'grp2', 'grp2'], ['one', 'two', 'one', 'two']]
indx = pd.MultiIndex.from_arrays(a, names=('Group', 'Number'))
df = pd.DataFrame({'val': [1, 2, 3, 4]}, index=indx)
print(df)


# =============================================================================
# 15. Access a specific column from a DataFrame by name and return it as a Series.
# =============================================================================
a = [[1, 2, 3],
     [3, 4, 5],
     [5, 6, 7]]
b = pd.DataFrame(a, columns=["1st", '2nd', '3rd'])
c = pd.Series(b['2nd'])
print("the extraced series is:\n", c)


# =============================================================================
# 16. Select multiple columns from a DataFrame and return them as a new DataFrame.
# =============================================================================

a = [[1, 2, 3],
     [3, 4, 5],
     [5, 6, 7]]
b = pd.DataFrame(a, columns=["1st", '2nd', '3rd'])
df = pd.DataFrame(b[['1st', '2nd']])
print(df)
print('\n')

# =============================================================================
# 17. Access a column from a DataFrame using dot notation (i.e., df.column_name).
# =============================================================================
a = [[1, 2, 3],
     [3, 4, 5],
     [5, 6, 7]]
b = pd.DataFrame(a, columns=["rohh", 'ronn', 'rohaan'])
print(b.rohaan)
print(b['rohh'])
print('\n')
print('\n')

# =============================================================================
# 18. Add a new column to a DataFrame using existing columns (e.g., sum two columns).
# =============================================================================

a = pd.DataFrame({'1st': [1, 2, 3, 4, 5],
                  '2nd': [2, 3, 4, 5, 6]})
a['added_col'] = a['1st']+a['2nd']
a.iloc[3]=np.nan
print("the 4th column after adding 1st and 2nd columns is:\n", a)
print('\n')
print('\n')


# =============================================================================
# 19. Rename a column in a DataFrame and display the updated DataFrame.
# =============================================================================

a = pd.DataFrame({'1st': [1, 2, 3, 4, 5],
                  '2nd': [2, 3, 4, 5, 6],
                 '3rd': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa']})
b = a.rename(index={0 : 'name'})
print("the updated column is:\n", b)
print('\n')
print('\n')


# =============================================================================
# 20. Check if a particular column exists in a DataFrame and print a message accordingly.
# =============================================================================


a = pd.DataFrame({'1st': [1, 2, 3, 4, 5],
                  '2nd': [2, 3, 4, 5, 6],
                 '3rd': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa']})
b = 3

if b in a.values:
    print(f"'{b}' element exists in the data frams")

else:
    print(f"'{b}' element do not exist")
print('\n')
print('\n')


# =============================================================================
# 21. Select only the numerical columns from a DataFrame and display their data types.
# =============================================================================


a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22]})

select = a.select_dtypes(include=int)
print(select)

print('\n')
print('\n')


# =============================================================================
# 22. Use the .iloc[] method to select the first column from a DataFrame.
# =============================================================================


a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                  'Age': [25, 30, 35, 40, 22]}, index=['student1', 'student2',
                                                       'student3', 'student4', 'student5'])
print(a.iloc[0:, 1])
print('\n')
print('\n')


# =============================================================================
# 23. Access the first row of a DataFrame using the .iloc[] method.
# =============================================================================

a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                  'Age': [25, 30, 35, 40, 22]}, index=['student1', 'student2',
                                                       'student3', 'student4', 'student5'])
print(a.iloc[0])
print('\n')
print('\n')


# =============================================================================
# 24. Use .loc[] to access rows of a DataFrame based on index labels.
# =============================================================================

a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                  'Age': [25, 30, 35, 40, 22]}, index=['student1', 'student2',
                                                       'student3', 'student4', 'student5'])
print(a.loc['student3',['Name','Age']])
print('\n')
print('\n')


# =============================================================================
# 25. Filter rows based on a condition (e.g., values in a specific column greater than 50).
# =============================================================================

a = [[11, 24, 39],
     [31, 42, 15],
     [5, 1, 17]]
b = pd.DataFrame(a, columns=["rohh", 'ronn', 'rohaan'])
b[b['ronn'] < 20]
print('\n')
print('\n')


# =============================================================================
# 26. Select the top 5 rows where a column contains a specific value (e.g., 'Male' in a
# gender column).
# =============================================================================
a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])
b = 'male'
gen = a[a['gender'] == 'male']

print('the data containing only male \n', gen)
print('\n')
print('\n')


# =============================================================================
# 27. Retrieve rows where multiple conditions are true (e.g., column A > 50 and column B
# < 20).
# =============================================================================
a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])

multi_condi = a[(a['Age'] >= 35) & (a['gender'] == 'female')]
print(multi_condi)
print('\n')
print('\n')


# =============================================================================
# 28. Select rows using a range of index labels (e.g., rows from index 'A' to 'D').
# =============================================================================

a = [1, 2, 3, 4, 5, 6, 7]
b = pd.DataFrame(a, index=['a', 'b', 'c', 'd', 'e',
                 'f', 'g'], columns=['alphebets'])
print(b['a':'e'])
print('\n')
print('\n')


# =============================================================================
# 29. Access rows of a DataFrame using a boolean mask and display the filtered
# DataFrame.
# =============================================================================


a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])

bool_mask = a['Age'] > 30
print(a[bool_mask])
print('\n')
print('\n')


# =============================================================================
# 30. Create a pandas Index object from a list of values and use it to index a Series.
# =============================================================================

indx = [9, 8, 7, 6, 5]
df = pd.Series(['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'], index=indx)
print(df)
print('\n')
print('\n')


# =============================================================================
# 31. Retrieve the index values of a DataFrame and print them.
# =============================================================================

a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])
print('the index values are :\n', a.index)
print('\n')
print('\n')


# =============================================================================
# 32. Set a column as the index of a DataFrame and display the updated DataFrame.
# =============================================================================

a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])
df = a.set_index('Name')
print('thenew df is :\n', df)

print('\n')
print('\n')


# =============================================================================
# 33. Reset the index of a DataFrame and convert the index into a regular column.
# =============================================================================
a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])
df = a.set_index('Name')
df1 = df.reset_index()
convert_index = a.reset_index()
print('conbverting the index to noemal column:\n', convert_index)
print('\n')
print('\n')


# =============================================================================
# 34. Create a pandas RangeIndex and explain its difference from a regular Index.
# =============================================================================

ind = pd.RangeIndex(0, 10, 2)
a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=ind)

print(a)
print('\n')
print('\n')

'''the difference between rangeindex and rgular index is that we can easily give
  index values for big data faster and based on our requirements'''


# =============================================================================
# 35. Rename the index of a DataFrame and show the updated DataFrame.
# =============================================================================

a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])

b = a.rename(index={'student1': 0, 'student2': 1,
             'student3': 2, 'student4': 3, 'student5': 4}, inplace=True)
print('renaming the index:\n', a)
print('\n')
print('\n')


# =============================================================================
# 36. Check if a given label exists in the index of a DataFrame.
# =============================================================================

a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])
b = 'student1'
if b in a.index:
    print(f"yes '{b}' exists")

else:
    print(f"no '{b}' do not exists")
    print('\n')
    print('\n')

# =============================================================================
# 37. Reindex a Series using a list of new labels and fill any missing values with 0.
# =============================================================================

a = pd.Series(['a', 'b', 'c', 'd'])
b = [0, 1, 2, 3, 4, 5]
c = a.reindex(b).fillna(0)
print(c)
print('\n')
print('\n')

# =============================================================================
# 38. Reindex a DataFrame with a new list of row labels and set a different order of
# columns.
# =============================================================================

a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['a', 'b', 'c', 'd', 'e'])
b = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
c = a.reindex(b)
print('the new data =frame that contains new lables are:\n', c)

print('\n')
print('\n')


# =============================================================================
# 39. Reindex a DataFrame using a method that forward fills missing values (e.g.,
# method='ffill').
# =============================================================================

a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['a', 'b', 'c', 'd', 'e'])
b = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
c = a.reindex(b).ffill()
print(c)
print('\n')
print('\n')


# =============================================================================
# 40. Reindex a DataFrame to match the index of another DataFrame.
# =============================================================================
a = pd.DataFrame([1, 2, 3, 4])
b = pd.DataFrame([9, 8, 7, 6, 5])
c = pd.concat([a, b], ignore_index=True)
print(c)
print('\n')
print('\n')

# =============================================================================
# 41. Align two DataFrames on their indexes and fill any missing values with NaN.
# =============================================================================
a = pd.DataFrame([[1, 2], [3, 4]], index=[1, 2], columns=['a', 'b'])
b = pd.DataFrame([[4, 5, 6], [8, 9, 1]], index=[1, 2], columns=['a', 'b', 'c'])
c = pd.concat([a, b], ignore_index=True).fillna(0)
print('Values of 2 data frames are:\n',a,'\n',b)
print('\n\n', c)

print('\n')
print('\n')


# =============================================================================
# 42. Reindex a DataFrame by reversing the index order and show the result.
# =============================================================================
a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['a', 'b', 'c', 'd', 'e'])
print(a[::-1])

print('\n')
print('\n')

# =============================================================================
# 43. Reindex a Series and assign default values for missing entries using the .fillna()
# method.
# =============================================================================

a = pd.Series([1, 2, 3, 4, 5, 6], index=[1, 2, 3, 4, 5, 6])
b = a.reindex([1, 2, 3, 4, 5, 6, 7, 8, 9]).fillna(0)
print('original value:\n', a, '\n\n')
print("Reindexing and filling using FillNA function:\n", b)

print('\n')
print('\n')

# =============================================================================
# 44. Drop a specific row from a DataFrame by label and display the result.
# =============================================================================


a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])
drop = a.drop('student1')
print('dropping row:\n', drop)


print('\n')
print('\n')

# =============================================================================
# 45. Drop multiple columns from a DataFrame by name using the .drop() method and
# display the result.
# =============================================================================


a = pd.DataFrame({'Name': ['rohan', 'roh', 'rohannn', 'roo', 'rohhhaa'],
                 'Age': [25, 30, 35, 40, 22],
                  'gender': ['male', 'female', 'male', 'female', 'male']}, index=['student1', 'student2',
                                                                                  'student3', 'student4', 'student5'])
drop = a.drop(['Age', 'gender'], axis=1)

print('the data after dropping columns is:\n', drop)
