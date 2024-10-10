import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

data = np.loadtxt("DataR_run_Black_02.csv", delimiter=",")
print(data)

first_row = data[30000]

x_values = np.arange(len(first_row)) * 4
plt.plot(x_values, first_row, marker='o')

# # Add labels and title
plt.title('Wave Form of One Shot')
plt.xlabel('Time (NanoSeconds)')
plt.ylabel('ADC Channels')
plt.legend()
plt.grid(True)
plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import numpy as np

# def shift_elements(arr, num, fill_value):
#     result = np.empty_like(arr)
#     if num > 0:
#         result[:len(arr)-num] = arr[num:]
#         result[len(arr)-num:] = fill_value
#     elif num < 0:
#         result[-num:] = arr[:num]
#         result[:-num] = fill_value
#     else:
#         result[:] = arr
#     return result

# data = np.loadtxt("Dark_Count.csv", delimiter=",")
# rows = data.shape[0]
# columns = data.shape[1]
# # print(data)
# # first_row = data[0]
# # add = data[0] + data[1]
# # print(add)


# # Locate where the largest value of a data point is at index m
# # move the whole array over to the left my m indexes to start at max, delete the rest of the points
# # points not included will be initalized to 0
# # coumt the number of 0 of each column of the data
# # divide by 1000-count_zero for each integrated column
# for i in range(rows):
#     max_index = np.argmax(data[i])
#     data[i] = shift_elements(data[i],max_index,0)

# print(data)


# zero_counts = np.zeros(columns, dtype=int)

# for j in range(columns):
#     zero_counts[j] = np.sum(data[j] == 0)

# print(zero_counts)

# column_sums = np.sum(data, axis=0)
# print(column_sums)

# y_values = np.zeros(columns, dtype=int)
# for k in range(columns):
#     y_values[k] = column_sums[k]/(1000-zero_counts[k])

# print(y_values)
# x_values = np.arange(len(y_values)) * 4
# plt.plot(x_values, y_values, marker='o')

# # # # Add labels and title
# plt.title('Average Waveform')
# plt.xlabel('Time (NanoSeconds)')
# plt.ylabel('ADC Channels')
# plt.legend()
# plt.grid(True)
# plt.show()