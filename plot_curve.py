import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def closest(lst, K): 
    lst = np.asarray(lst) 
    idx = (np.abs(lst - K)).argmin() 
    return lst[idx] 

filename = sys.argv[1]
fileDir = 'log/'

curr_power_list = []
avgr_power_list = []
time_list = []

time_list.append(float(0))

filepath = os.path.join(fileDir, filename)
with open(filepath) as f:
    str = f.read()

data = str.split()
end_time = []
end_time_index = [] # corresponding index in time list

# extract gpu power consumption and execution time
for index, line in enumerate(data):
    if 'GPU' == line:
        a, b = data[index+1].split("/")
        curr_power_list.append(float(a))
        avgr_power_list.append(float(b))
    elif 'time' in line:
        end_time.append(float(data[index+1]))

if not end_time:
    print("Log time error!")
    exit()
else:
    range_time = int(end_time[-1])
    increment = range_time / len(curr_power_list)

for x in range(1, len(curr_power_list)):
    time_list.append(float(x * increment))

for t in end_time:
    t = closest(time_list, t)
    end_time_index.append(time_list.index(t))

# did not include cooling down time
e_curr = np.trapz(curr_power_list[:end_time_index[-1]])
print("Energy consumption: {}".format(e_curr / 1000 / 1000))

plt.plot(time_list, curr_power_list)
# plt.plot(time_list, avgr_power_list)
for d in end_time_index[:-1]:
    plt.axvline(x = time_list[d], color='k', linestyle='dashed', linewidth=1)
plt.grid(b=True, which='major', color='#999999', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.title(filename)
plt.xlabel("Time (ms)")
plt.ylabel("Power Consumption (mWatt)")
plt.show()