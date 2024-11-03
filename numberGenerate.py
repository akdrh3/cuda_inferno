import random as r
import time
import math
from datetime import datetime

input_a = int(input("Enter the size of numbers: "))

# Get the current time
current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Current Time:", formatted_time)

start_time = time.time()

data_size = input_a * 1000000
file_name = f"numbers_{data_size}.txt"
parse = math.ceil(data_size/10)
with open(file_name, "w") as f:
    for i in range(0,data_size):
        num = r.randint(0,2147483647)
        f.write("%d\n"%num)
        if(i%parse == 0):
            print(f"just generated {i}th element: {num}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time for generating data set: {elapsed_time} sec")

current_time = datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Current Time:", formatted_time)