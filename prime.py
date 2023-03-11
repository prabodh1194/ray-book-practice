import time

import ray


@ray.remote
def isprime(_x):
    if _x > 1:
        for i in range(2, _x):
            if _x % i == 0:
                return 0
        else:
            return _x
    return 0


lower = 9000000
upper = 9003900

primes = []
objects = []

start_time = time.time()

for num in range(lower, upper + 1):
    x = isprime.remote(num)
    objects.append(x)

objs = ray.get(objects)
[primes.append(x) for x in objs if x > 0]

print(len(primes), primes[0], primes[-1])

print("Time taken =", (time.time() - start_time))
