Prerequsites
============

* `numpy tutorial <https://docs.scipy.org/doc/numpy-dev/user/quickstart.html>`_
* `pandas tutorial <http://pandas.pydata.org/pandas-docs/version/0.18.1/tutorials.html>`_
* `matplotlib tutorial <http://matplotlib.org/users/pyplot_tutorial.html>`_

`numpy`
-------

:code:`ndarray` important attributes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`ndarray.ndim:` number of dimensions
* :code:`ndarray.shape:` return 3 by 5 matrix
* :code:`ndarray.size:` number of elements
* :code:`ndarray.dtype:` data type
* :code:`ndarray.itemsize:` number of bytes of the data type

Array creation
^^^^^^^^^^^^^^

* :code:`np.array()`, we can use :code:`dtype` argument like::

    c = np.array([[1, 2], [3, 4]], dtype=complex)

* We can also use the following methods to create arrays with sizes::

    zeros((3, 4))
    ones((3, 4))
    empty((3, 4))

* To create array with sequence of numbers::

    np.arange(0, 30, 5) # Note good for float number
    np.linspace(0, 2*pi, 100) # Get exact number we want

Printing Array
^^^^^^^^^^^^^^

* printing is truncated if array is too large, use the following to change the
  behaviour::

    np.get_printoptions()
    np.set_printoptions(threshold='nan')

Basic Operations
^^^^^^^^^^^^^^^^

* :code:`*` is elementwise product, :code:`A.dot(B)` or :code:`np.dot(A, B)` is
  the matrix multiplication.
* when doing :code:`*=` and :code:`+=`, the variable on the left side must be
  precise one
* :code:`sum(), min(), max()` can accept :code:`axis` argument. If :code:`axis=0`,
  it will do the calculation along the first axis.

Universal Functions
^^^^^^^^^^^^^^^^^^^

* :code:`sin, cos, exp, sqrt` are all elementwise.

Indexing, Slicing and Iterating
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* 1-d array can be indexed as the normal list.
* multiple dimensional array can have one index per axis, given by a tuple separated
  by comma::

    b[2, 3]
    b[0:5, 1]
    b[:, 1]
    b[1:3, :]

* We can use :code:`...` in indexing, assume :code:`x` has 5 axis::

    x[1,2,...] = x[1,2,:,:,:]
    x[...,3]   = x[:,:,:,:,3]
    x[4,...,5] = x[4,:,:,:,5]

* iterating first axis::

    for row in b:
        print(row)

* iterating all elemnts, the following are the same::

    for row in b:
      for e in row:
          print(e)

    for e in b.flat:
      print(e)

Summary of methods
^^^^^^^^^^^^^^^^^^

* Here is the summary of methods seen so far::

    np.linspace()
    np.arange()
    np.array()
    np.arange(12).reshape(3, 4)
    np.zeros()
    np.ones((2, 3))
    np.empty((2, 3))
    np.random.random((2, 3))
    np.exp(c*1j)
    np.get_printoptions()
    np.set_printoptions(threshold='nan')
    np.dot(A, B)
    A.dot(B)
    a.sum(), a.min(axis=0), a.max(axis=1), a.cumsum()
    np.fromfunction(f, (5, 4), dtype=int)
    b.flat # Arribute

Chapter 02 Training Machine Learning Algorithms for Classification
==================================================================
