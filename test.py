import psutil, os
print(os.cpu_count()) # logical
print(psutil.cpu_count(logical=False)) # physical


