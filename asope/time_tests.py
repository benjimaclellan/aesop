import numpy as np
import time

n=2**13
a1 = np.random.uniform(0,1,(n))
a2 = np.random.uniform(0,1,(n))

N = 10000

t1 = time.time()
for i in range(N):
    val = np.max(np.abs(a1 - a2))
t2 = time.time()
print('For {} tests, the time is {}'.format(N, t2-t1))


t1 = time.time()
for i in range(N):
    val = np.sum( np.power(a1-a2,2) ) 
t2 = time.time()
print('For {} tests, the time is {}'.format(N, t2-t1))
