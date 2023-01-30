import pandas as pd
import numpy as np

arr = np.array([[2,2], [4,4]])

# print(np.shape(arr))

df = pd.DataFrame(arr, index = ['a', 'b'])

# print(df)

df = df.drop(index='a')

print(df)