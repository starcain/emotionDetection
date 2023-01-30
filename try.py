import pandas as pd
import numpy as np

arr = np.array([[2,2], [4,4], [6,6], [8,8], [10,10]])

# print(np.shape(arr))
ind = ['a', 'b', 'c', 'd', 'e']
df = pd.DataFrame(arr, index = ind)

# print(df)

df = df.drop(index=['a', 'e'])

print(df)

print(len(ind))