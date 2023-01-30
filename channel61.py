#function imports..
from functions import modelTest

#parameters declaration..
index = 0
dropIndex = [3]
ratio = 0.75
batch = 32
epoch = 80

#function call..
model = modelTest(index, dropIndex, ratio, batch, epoch)