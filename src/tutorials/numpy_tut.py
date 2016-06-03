import numpy as np

a = np.arange(15).reshape(3, 5)

print(a)
print(a.__class__)

print(a.shape)
print(a.ndim)
print(a.dtype)
print(a.itemsize)
print(a.size)
print(type(a))

b = np.array([6, 7, 8])
print(b)
print(type(b))

# Array Creation

a = np.array([2, 3, 4])
print(a)
print(a.dtype)

b = np.array([1.2, 3.5, 5])
print(b)
print(b.dtype)
print(b.itemsize)

c = np.array([[1, 2], [3, 4]], dtype=complex)
print(c)

print(np.zeros((3, 4)))
b = np.ones((2, 3, 4), dtype=np.int16)
print(b)
print(b.itemsize)

print(np.empty((2, 3)))

print(np.arange(0, 30, 5))
print(np.arange(0, 2, 0.3))

print(np.linspace(0, 2, 9))
from numpy import pi

x = np.linspace(0, 2 * pi, 100)
f = np.sin(x)

print("#" * 80 + "\n")
print("## Printing Array\n")
print("#" * 80 + "\n")

a = np.arange(6)
print(a)

a = np.arange(12).reshape(4, 3)
print(a)

a = np.arange(24).reshape(2, 3, 4)
print(a)

print(np.arange(10000))
print(np.arange(10000).reshape(100, 100))
print(np.get_printoptions())

print("#" * 80 + "\n")
print("## Basic Operations\n")
print("#" * 80 + "\n")

a = np.array([20, 30, 40, 50])
b = np.arange(4)
c = a - b
print(c)

print(b ** 2)
print(10 * np.sin(a))
print(a < 35)

A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])

print(A * B)
print(A.dot(B))
print(np.dot(A, B))

a = np.ones((2, 3), dtype=int)
b = np.random.random((2, 3))
a *= 3
print(a)
b += a
print(b)

# Doesn't work
# a += b

a = np.ones(3, dtype=np.int32)
print(a)
b = np.linspace(0, pi, 3)
print(b)
print(b.dtype)
c = a + b
print(c.dtype)
d = np.exp(c * 1j)
print(d)
print(d.dtype)

a = np.random.random((2, 3))
print(a)
print(a.sum())
print(a.min())
print(a.max())

b = np.arange(12).reshape(3, 4)
print(b)
print(b.sum(axis=0))
print(b.min(axis=1))
print(b.cumsum(axis=1))

a = np.arange(10) ** 3
print(a)
print(a[2])
print(a[2:5])
a[:6:2] = -1000
print(a)
print(a[::-1])


# for i in a:
#     print(i**(1/3.0))

def f(x, y):
    return 10 * x + y


b = np.fromfunction(f, (5, 4), dtype=int)

print(b)
print(b[2, 3])
print(b[0:5, 1])
print(b[:, 1])
print(b[1:3, :])

for row in b:
    print(row)

for row in b:
    for e in row:
        print(e)

for e in b.flat:
    print(e)

a = np.floor(10 * np.random.random((3, 4)))
print(a)
print(a.shape)
a.shape = (6, 2)
print(a)
print(a.T)
print(a.ravel())

a.resize((3, 4))
print(a)
print(a.reshape(2, -1))
print(a.reshape(2, -1, 3))

a = np.array([4, 3])
b = np.array([2, 8])
print(np.vstack((a, b)))
print(np.hstack((a, b)))
print(np.column_stack((a, b)))
print(np.column_stack((a[:, np.newaxis], b[:, np.newaxis])))
print(np.vstack((a[:, np.newaxis], b[:, np.newaxis])))

a = np.floor(10*np.random.random((2, 12)))
print(a)
print(np.hsplit(a, 3))
print(np.hsplit(a, (3, 4)))
print(np.vsplit(a, 2))

time = np.linspace(20, 145, 5)                 # time scale
data = np.sin(np.arange(20)).reshape(5,4)      # 4 time-dependent series
print(time)
print(data)
ind = data.argmax(axis=0)                   # index of the maxima for each series
print(ind)


time_max = time[ ind]                       # times corresponding to the maxima

data_max = data[ind, range(data.shape[1])] # => data[ind[0],0], data[ind[1],1]...

time_max

data_max


np.all(data_max == data.max(axis=0))