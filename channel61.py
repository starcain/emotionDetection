#function imports..
from functions import modelTest

#parameters declaration..
index = 0
dropIndex = [3]
ratio = 0.75
batch = 32
epoch = 80

#function call..
for i in range(62):
    print("REMOVED INDEX: ",i)
    print("REMOVED CHANNEL: ", channel[i])
    dropIndex = [i]
    model = modelTest(index, dropIndex, ratio, batch, epoch)