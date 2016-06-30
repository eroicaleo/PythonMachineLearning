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

Shape Manipulation
^^^^^^^^^^^^^^^^^^

* :code:`a.shape:` return the shape of the array.
* :code:`a.shape = (6, 2)` directly modify the shape.
* :code:`a.T:` do the transpose.
* :code:`a.ravel():` make the array flat.
* :code:`a.resize((3, 4))` modify the array :code:`a` itself, while
  :code:`a.reshape(2, 6)` returns a copy. Note the arguments are different for the
  two methods. We can use :code:`a.reshape(3, -1)` to let it decide the size.

Stacking together different arrays
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`hstack(), vstack()`: stack array horizontally and vertically.
* :code:`column_stack()`: stack 1-d array to 2-d array, each 1-d array becomes
  one column.
* :code:`a[:, np.newaxis]`: To make a 1-d row array to 2-D column vector.

Splitting one array into several smaller ones
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :code:`hsplit, vsplit`
    * :code:`hsplit(a, 3)`, split the array into 3 parts along the horizontal axis
    * :code:`hsplit(a, (3, 4))`, split the array after 3rd and 4th column.

View and Copy
^^^^^^^^^^^^^

* :code:`a = b` doesn't create a copy
* :code:`a = b.view()`, only creates a view, they share the same data. Change the
  data in :code:`b` also changes data in :code:`a`.
* :code:`a = b.copy()` makes a deep copy.

Fancy Indexing
^^^^^^^^^^^^^^

We will take a case analysis approach, assume :code:`a` is the array, :code:`i`
and :code:`j` are the index array.

* If :code:`a` is one dimension, :code:`a[i]` is same shape as :code:`i`.
* If :code:`a` is multidimensional, think of it as an one dimension array along
  the 1st axis.
* If :code:`a` is 2-dimension, the :code:`i, j` must be the same shape. :code:`a[i, j]`
  is the same shape as :code:`i`.

    * :code:`a[i, 2]` is the same shape as :code:`i`.
    * :code:`a[:, j]` is stack of :code:`a[0, j], a[1, j]` and :code:`a[2, j]`.

* We can also do assignment: :code:`a[i] = 0`;
* We can even do increament: :code:`a[i] += 1`, but need to be careful if there
  is duplication in :code:`i`.

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
    a.argmax(), a.argmin()
    np.fromfunction(f, (5, 4), dtype=int)
    b.flat # Arribute
    ## Shape Manipulation
    a.reshape(3, 1)
    a.resize(3, 2)
    a.ravel()
    hstack((a, b))
    vstack((a, b))
    column_stack((a, b))
    a[:, newaxis] # To make 2-D column vector
    np.hsplit(a, )
    ## View and Copy
    a.view()
    a.copy()

pandas
------

Object creation
^^^^^^^^^^^^^^^

* series data: :code:`pd.Series([1, 3, 5, np.nan, 6, 8])`
* data frame:

    * :code:`pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))`
    * Dictionary like method::

        df2 = pd.DataFrame({
          'A': 1.,
          'B': pd.Timestamp('20130102'),
          'C': pd.Series(1, index=list(range(4)), dtype='float32'),
          'D': np.array([3] * 4, dtype='int32'),
          'E': pd.Categorical(["test", "train", "test", "train"]),
          'F': 'foo'
        })

Viewing Data
^^^^^^^^^^^^

* :code:`df.head()`: top data.
* :code:`df.tail(3)`: bottom data.
* :code:`df.index`: get the index.
* :code:`df.columns`: get the column name.
* :code:`df.values`: get the contents of the data frame.
* :code:`df.T`: transpose.
* :code:`df.sort_index(axis=1, ascending=False)`: sort column.
* :code:`df.sort_value(by='B')`: sort by values.

Selection
^^^^^^^^^

* :code:`df['A'], df.A`: select column A.
* :code:`df[0:3], df['20130102:20130104']`: slices the rows.
* :code:`df.loc[:, ['A', 'B']]`: select all rows with column 'A' and 'B'.
* :code:`df.at[dates[0], 'A']`: access a scalar, faster than :code:`df.loc`.
* :code:`df.iloc[3:5, 0:2]`: access column 3, 4 and row 0, 2 with integer index.

Boolean selection

* :code:`df[df['A'] > 0]`
* :code:`df[df > 0]`
* :code:`df2[df2['E'].isin(['two', 'four'])]`

Setting

* Adding a column to a dataframe:
    * :code:`s1 = pd.Series([1, 2, 3, 4, 5, 6], index=pd.date_range('20130102', periods=6))`
    * :code:`df['F'] = s1`
* Assign a numpy array to a column:
    * :code:`df.loc[:, 'D'] = np.array([5] * len(df))`

Missing Data
^^^^^^^^^^^^

Useful methos:

* :code:`df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])`, the new
  column will be filled with :code:`NaN`.
* :code:`df.dropna(how='any')`
* :code:`df.fillna(value=5)`
* :code:`df.isnall()`

Operations
^^^^^^^^^^

* :code:`df.mean()`
* :code:`df.sub(s, axis='index')`
* :code:`df.apply()`
* :code:`s = pd.Series(np.random.randint(0, 7, size=10))`, then :code:`s.value_counts()`
* :code:`s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])`
* :code:`s.str.lower()`

Merge
^^^^^

There are 3 ways to merge

* :code:`concat`, :code:`append`: straight forward, just concat and append two
  dataframe.
* :code:`join`: SQL like merge.

Grouping
^^^^^^^^

3 steps in grouping: :code:`f.groupby(['A', 'B']).sum()`

1. split the data
2. apply a function to each group separately
3. combining the results to a data structure


Summary of methods
^^^^^^^^^^^^^^^^^^

* Here is the summary of methods seen so far::

    pd.Series()
    pd.DataFrame()
    pd.date_range('20130101', periods=6)
    pd.head()
    pd.tail()
    df.index
    df.column
    df.values
    df.T
    df.sort_index()
    df.sort_value()
    df.loc['20130102':'20130104',['A','B']] # loc is property
    df.at[dates[0], 'A']
    df.iloc[]
    df.iat[]
    df.isin()
    df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
    df.dropna(how='any')
    df.fillna(value=5)
    df.isnall()
    pd.concat()
    df.append(s, ignore_index=True)
    pd.merge(left, right)
    f.groupby(['A', 'B']).sum()

Chapter 02 Training Machine Learning Algorithms for Classification
==================================================================
