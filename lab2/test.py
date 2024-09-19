import numpy as np

data = np.loadtxt('test.txt', delimiter=',')
print(data)

data = data.reshape(-1,1)
print(data)

original_min = np.min(data)
original_max = np.max(data)
data = (data - original_min)/(original_max - original_min)
print(data)

restored_data = data * (original_max - original_min)+ original_min
print(restored_data)

