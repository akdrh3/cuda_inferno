import random as r
import time
import math
from datetime import datetime

input_a = int(input("Enter the size of numbers: "))
print("generating ... ")

start_time = time.time()

f = open("numbers.txt", "w")
data_size = input_a * 1000000
parse = math.ceil(data_size/5)
for i in range(0,input_a * 1000000):
    num = r.randint(0,2000000000)
    f.write("%d\n"%num)
    # if(i%parse == 0):
    #     print(f"just generated {i}th element: {num}")

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time for generating data set: {elapsed_time} sec")

# Get the current time
current_time = datetime.now()

# Format the time
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

# Print the formatted time
print("Current Time:", formatted_time)

# f = open("oteMillionNum.txt", "w")
# for i in range(0,128000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)


# f = open("tfsMillionNum.txt", "w")
# for i in range(0,256000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)


# f = open("fotMillionNum.txt", "w")
# for i in range(0,512000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)

# f = open("oztfMillionNum.txt", "w")
# for i in range(0,1024000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)

# f = open("tzfeMillionNum.txt", "w")
# for i in range(0,2048000000):
#     num = r.randint(0,100000000)
#     f.write("%d\n"%num)
