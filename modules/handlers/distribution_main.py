from BamHandler import BamHandler
from functools import partial
from Fast5Handler import H5_Handler
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

#
path = '/home/jacobgull/Desktop/TEST/Jacob/Nanopore_signals/rel3/signal_raw/fn/'
h5_handler = H5_Handler(path) #starts the mean read_id dictionary
print("1")
hist_Data = h5_handler.histogram_data()
hist_dict = defaultdict(list)
print("2")
key_list = []
first_iteration = True
test_key = []
for line in hist_Data:
    if(line[5] == 0 and first_iteration == False): #removes duplication of kmers
        pass
    else:
        # print(type(hist_dict[line[4]]))
        # print(type(float(line[0])))
        hist_dict[line[4]].append(float(line[0])) #  This needs to append a list to the list of values.
        first_iteration = False

# print(hist_dict)
# counter = 0
# for key in hist_dict:
#     if (counter == 20):
#         break
#     else:
#         counter+=1
#     test_list = hist_dict[key]
#     y, x = np.histogram(test_list, bins=np.arange(200,step= 1))
#     figure, axes = plt.subplots()
#     axes.plot(x[:-1], y)
#     figure.show()
#     if (counter != 20):
#         counter += 1
#     else:
#         break