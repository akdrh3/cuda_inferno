import random as r
import time
import math
from datetime import datetime

input_a = int(input("Enter the size of numbers: "))
print("generating ... ")

start_time = time.time()

f = open("numbers.txt", "w")
data_size = input_a * 10

for i in range(0,data_size):
    num = r.randint(0,200)
    f.write("%d\n"%num)

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time for generating data set: {elapsed_time} sec")

# Get the current time
current_time = datetime.now()

# Format the time
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

# Print the formatted time
print("Current Time:", formatted_time)
