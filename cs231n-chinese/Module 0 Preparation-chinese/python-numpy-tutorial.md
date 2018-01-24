---
layout: page
title: Python Numpy Tutorial
permalink: /python-numpy-tutorial/
---

<!--
Python:
  简单的数据类型
    integer, float, string
 复合数据类型
    tuple, list, dictionary, set
  流控制
    if, while, for, try, with
  解析器, 生成器
  函数
  库
  标准库
    json, collections, itertools

Numpy
-->

本教程最初由 [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) 撰写创立。现由[Qi (Lewis) Liu](https://github.com/liuqidev) 翻译整理。

在本门课程（CS231n）中， 我们将使用Python编程语言来提交所有的作业。Python就其本身而言是一门通用的编程语言，得益于众多受欢迎的开源库（numpy, scipy, matplotlib），其成为科学计算领域不可多得的强大工具。

我们希望你们对于Python和numpy的使用或多或少有些经验；如果你没有这方面的经验，那么本教程就是为你而生，你可以将其视作一个速成课程，你将快速学会Python编程语言以及Python在科学计算中的使用。

可能你们中有些人之前用过Mathlab，那么我们也推荐你去看 [numpy for Matlab users](http://wiki.scipy.org/NumPy_for_Matlab_Users) .

你也可以找到 [IPython notebook tutorial ](https://github.com/kuleshov/cs228-material/blob/master/tutorials/python/cs228-python-tutorial.ipynb) 早前版本的一个教程，由 [Volodymyr Kuleshov](http://web.stanford.edu/~kuleshov/) 和 [Isaac Caswell](https://symsys.stanford.edu/viewing/symsysaffiliate/21335) 撰写创立，用于先修课 [CS 228](http://cs.stanford.edu/~ermon/cs228/index.html).

目录:

- [Python](#python)
  - [基本数据类型 Basic data types](#python-basic)
  - [容器 Containers](#python-containers)
      - [列表 Lists](#python-列表s)
      - [字典 Dictionaries](#python-dicts)
      - [集合 Sets](#python-sets)
      - [元组 Tuples](#python-tuples)
  - [函数 Functions](#python-functions)
  - [类 Classes](#python-classes)
- [Numpy](#numpy)
  - [Arrays](#numpy-arrays)
  - [Array indexing](#numpy-array-indexing)
  - [Datatypes](#numpy-datatypes)
  - [Array math](#numpy-math)
  - [Broadcasting](#numpy-broadcasting)
- [SciPy](#scipy)
  - [Image operations](#scipy-image)
  - [MATLAB files](#scipy-matlab)
  - [Distance between points](#scipy-dist)
- [Matplotlib](#matplotlib)
  - [Plotting](#matplotlib-plotting)
  - [Subplots](#matplotlib-subplots)
  - [Images](#matplotlib-images)

<a name='python'></a>

## Python

Python是一门高级的，动态类型，多参数的编程语言。Python程序被誉为“可执行的伪代码”，因为你允许你使用极少的几行代码就可以实现强大的逻辑。作为示例，下面是使用Python编写的经典快速排序实现。

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
# 输出： "[1, 1, 2, 3, 6, 8, 10]"
```

### Python 版本
目前有两个官方支持的Python版本，一个是2.7版本，另一个是3.5的。

那么该选择哪个呢？Python 3.0 引入了很多向后不兼容的语法，所以使用Python 2.7编写的代码可能在3.5版本之下无法运行，反之亦然。

对于本门课程来说，所有代码将采用Python 3.5. (译者注：事实上并非如此)

你可以通过在命令行上输入以下代码来查看自己的Python 版本：
`python --version`.

<a name='python-basic'></a>

### 基本数据类型

和大多数编程语言一样，Python有一些列自己的基本数据类型，包括整型，浮点型，布尔型，字符串类型。其属性和其他编程语言类似。

**数字类型 Numbers:** 整型和浮点型可能如你所想和其他编程语言中的用法无异。

```python
x = 3
print(type(x)) # 输出 "<class 'int'>"
print(x)       # 输出 "3"
print(x + 1)   # 加; 输出 "4"
print(x - 1)   # 减; 输出 "2"
print(x * 2)   # 乘; 输出 "6"
print(x ** 2)  # 幂; 输出 "9"
x += 1
print(x)  # 输出 "4"
x *= 2
print(x)  # 输出 "8"
y = 2.5
print(type(y)) # 输出 "<class 'float'>"
print(y, y + 1, y * 2, y ** 2) # 输出 "2.5 3.5 5.0 6.25"
```
注意，和很多编程语言不一样的是，Python没有一元的增 (`x++`) 和一元减 (`x--`) 操作符。

对于更复杂的数字Python也提供了很多内置类型，具体细节参见[这篇文档](https://docs.python.org/3.5/library/stdtypes.html#numeric-types-int-float-complex).



**布尔类型 Booleans:** Python实现了所有的常用的布尔逻辑操作符，但是，不是符号（`&&`, `||`, 等），取而代之的是英语单词（`and`,`or`,`not`,等）:

```python
t = True
f = False
print(type(t)) # 输出 "<class 'bool'>"
print(t and f) # 逻辑 与; 输出 "False"
print(t or f)  # 逻辑 或; 输出 "True"
print(not t)   # 逻辑 非; 输出 "False"
print(t != f)  # 逻辑 异或; 输出 "True"
```

**字符串类型 Strings:** Python对于字符串类型有非常强大的支持性:

```python
hello = 'hello'    # 字符串可以使用单引号阔起来，
world = "world"    # 也可以使用双引号; 两者完全等效
print(hello)       # 输出 "hello"
print(len(hello))  # 字符串长度; 输出 "5"
hw = hello + ' ' + world  # 字符串级联
print(hw)  # 输出 "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf 风格的字符串格式化
print(hw12)  # 输出 "hello world 12"
```

字符串对象有许多有用的方法；例如：

```python
s = "hello"
print(s.capitalize())  # 将字符串首字母大写; 输出 "Hello"
print(s.upper())       # 将字符串转换成全大写; 输出 "HELLO"
print(s.rjust(7))      # 将字符串右对齐, 不够使用空格填充; 输出 "  hello"
print(s.center(7))     # 将字符串中间对其, 不够使用空白填充; 输出 " hello "
print(s.replace('l', '(ell)'))  # 将字符串中所有找到的字串使用另一个字串替换
                                # 输出 "he(ell)(ell)o"
print('  world '.strip())  # 跳过字符串首部和尾部的空字符串; 输出 "world"
```
想查看所有的字符串内置方法的列表，看 [这篇文档](https://docs.python.org/3.5/library/stdtypes.html#string-methods).

<a name='python-containers'></a>

### 容器 Containers
Python 中包含了好几种内置的容器类型：列表(lists), 字典(dictionaries), 集合(sets), 和元组(tuples).

<a name='python-lists'></a>

#### 列表 Lists
Python中列表(list) 相当于是数组(array), 但是Python列表是大小可变的

并且**可以存储不同类型的元素 **:

```python
xs = [3, 1, 2]    # 创建一个列表
print(xs, xs[2])  # 输出 "[3, 1, 2] 2"
print(xs[-1])     # 负的索引，将从列表的后边开始计数; 输出 "2"
xs[2] = 'foo'     # 列表可以包含不同类型的元素
print(xs)         # 输出 "[3, 1, 'foo']"
xs.append('bar')  # 向列表中最后增加一个元素
print(xs)         # 输出 "[3, 1, 'foo', 'bar']"
x = xs.pop()      # 移除并返回列表中最后一个元素
print(x, xs)      # 输出 "bar [3, 1, 'foo']"
```
关于列表使用凶残的细节，我不会说可以在[这篇文档](https://docs.python.org/3.5/tutorial/datastructures.html#more-on-lists)中找到的。

**切片 Slicing:**
除了每次访问列表元素的一个元素外，Python提供了一种简洁的语法来访问列表中的一部分；这种方式叫做切片：

```python
nums = list(range(5))     # range 为一个内置函数，用来创建一个整型的列表
print(nums)               # 输出 "[0, 1, 2, 3, 4]"
print(nums[2:4])          # 得到一个切片，元素为索引 2 到 4（开）的列表元素; 输出 "[2, 3]"
print(nums[2:])           # 得到一个切片，元素为索引 2 到 结尾的元素; 输出 "[2, 3, 4]"
print(nums[:2])           # 得到一个切片，元素为索引从开始(0) 到 2 (开)的元素; 输出 "[0, 1]"
print(nums[:])            # 得到一个切片，元素为整个列表的全部元素; 输出 "[0, 1, 2, 3, 4]"
print(nums[:-1])          # 切片的索引也可以为负值; 输出 "[0, 1, 2, 3]"
nums[2:4] = [8, 9]        # 将一个列表对切片进行赋值
print(nums)               # 输出 "[0, 1, 8, 9, 4]"
```
我们在介绍 numpy arrays 的时候还会见到切片。

**循环 Loops:** 你可以像下面这样遍历列表元素:

```python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)
# 输出 "cat", "dog", "monkey", 每个元素一行.
```

如果你想在循环体中获取其中元素的索引，需要使用内置函数`enumerate`I：

```python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# 输出 "#1: cat", "#2: dog", "#3: monkey", 每个输出一行
```

**列表解析 List comprehensions:**
编程工程中，我梦经常需要将一种类型的数据转换成另外一种类型的数据。

下面是计算数字平方数的例子:

```python
nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)   # 输出 [0, 1, 4, 9, 16]
```

可以使用**列表解析 list comprehension**来更加优雅地完成上述任务：

```python
nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)   # 输出 [0, 1, 4, 9, 16]
```

列表解析中也可以包含条件表达式：

```python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)  # 输出 "[0, 4, 16]"
```

<a name='python-dicts'></a>

#### 字典 Dictionaries
字典用来存储键值对——(key, value)，和Java中的`map` 或者Javascript中的对象类似。其用法如下：

```python
d = {'cat': 'cute', 'dog': 'furry'}  # 创建一个字典
print(d['cat'])       # 获取字典中的某项; 输出 "cute"
print('cat' in d)     # 检查字典中是否有某个给定的键; 输出 "True"
d['fish'] = 'wet'     # 修改字典中的某项
print(d['fish'])      # 输出 "wet"
# print(d['monkey'])  # 键的错误，KeyError: 'monkey' 不是字典 d 中的键
print(d.get('monkey', 'N/A'))  # 获取到一个默认的元素; 输出 "N/A"
print(d.get('fish', 'N/A'))    # 获取到一个默认的元素; 输出 "wet"
del d['fish']         # 从字典中移除一个元素
print(d.get('fish', 'N/A')) # "fish" 不再是其中的一个键; 输出 "N/A"
```
关于字典的更多的用法，请阅读 [这篇文档](https://docs.python.org/3.5/library/stdtypes.html#dict).

**遍历:** 字典很容易遍历其中的键:

```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal in d:
    legs = d[animal]
    print('A %s has %d legs' % (animal, legs))
# 输出 "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

如果你想在遍历字典过程中，同时获取字典中的键和键对应的值，使用`items` 方法：

```python
d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A %s has %d legs' % (animal, legs))
# 输出 "A person has 2 legs", "A cat has 4 legs", "A spider has 8 legs"
```

**字典解析 Dictionary comprehensions:**
字典解析和列表解析类似，能够以更快的方式建立字典。例如:

```python
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # 输出 "{0: 0, 2: 4, 4: 16}"
```

<a name='python-sets'></a>

#### 集合 Sets
集合是不同无序元素组成的集体。其基本的用法，参见下方示例：

```python
animals = {'cat', 'dog'}
print('cat' in animals)   # 检查某元素是否是集合中的元素; 输出 "True"
print('fish' in animals)  # 输出 "False"
animals.add('fish')       # 向集合中增加一个元素
print('fish' in animals)  # 输出 "True"
print(len(animals))       # 集合中元素的数量; 输出 "3"
animals.add('cat')        # 向集合中增加一个已经存在的元素
print(len(animals))       # 输出 "3"
animals.remove('cat')     # 从结合中移除一个元素
print(len(animals))       # 输出 "2"
```

同样，如果想知道关于集合的更多用法的介绍请阅读[这篇文档](https://docs.python.org/3.5/library/stdtypes.html#set).


**遍历:**
遍历集合和遍历列表的语法相同;
但是，由于集合是无序的，所以你没办法假设你访问的集合中元素就是按照显示的顺序访问的：

```python
animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#%d: %s' % (idx + 1, animal))
# 输出 "#1: fish", "#2: dog", "#3: cat"
```

**集合解析 Set comprehensions:**
和集合以及字典类似，你也可以使用集合解析来构建集合:

```python
from math import sqrt
nums = {int(sqrt(x)) for x in range(30)}
print(nums)  # 输出 "{0, 1, 2, 3, 4, 5}"
```

<a name='python-tuples'></a>

#### 元组 Tuples
元组是值不可更改的有序列表。

元祖在很多方面和列表一样, 和列表最大的不同点就是元祖可以作为字典以及集合中的键, 而列表是不可以这样的。

下面是一个简单的示例：

```python
d = {(x, x + 1): x for x in range(10)}  # 创建一个带有元组键值的字典
t = (5, 6)        # 创建一个元组
print(type(t))    # 输出 "<class 'tuple'>"
print(d[t])       # 输出 "5"
print(d[(1, 2)])  # 输出 "1"
```
[这个文档](https://docs.python.org/3.5/tutorial/datastructures.html#tuples-and-sequences) 有关于元组更多的信息。

<a name='python-functions'></a>

### 函数 Functions
Python 函数使用 `def` 关机字来构建. 举例:

```python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))
# 输出 "negative", "zero", "positive"
```

我们常常定义一些包含一些可选参数的函数，就像下面这个一样：

```python
def hello(name, loud=False):
    if loud:
        print('HELLO, %s!' % name.upper())
    else:
        print('Hello, %s' % name)

hello('Bob') # 输出 "Hello, Bob"
hello('Fred', loud=True)  # 输出 "HELLO, FRED!"
```
[这篇文档](https://docs.python.org/3.5/tutorial/controlflow.html#defining-functions) 中有关于Python 函数的更多用法.

<a name='python-classes'></a>

### 类 Classes

Python 用来定义类的语法很简单:

```python
class Greeter(object):

    # 构造器 Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # 实例方法 Instance method
    def greet(self, loud=False):
        if loud:
            print('HELLO, %s!' % self.name.upper())
        else:
            print('Hello, %s' % self.name)

g = Greeter('Fred')  # 构建一个 Greeter 类的实例
g.greet()            # 调用一个实例方法; 输出 "Hello, Fred"
g.greet(loud=True)   # 调用一个实例方法; 输出 "HELLO, FRED!"
```
关于Python类，更多请阅读 [这篇文档](https://docs.python.org/3.5/tutorial/classes.html).

<a name='numpy'></a>

## Numpy

[Numpy](http://www.numpy.org/) is the core library for scientific computing in Python.
It provides a high-performance multidimensional array object, and tools for working with these
arrays. If you are already familiar with MATLAB, you might find
[this tutorial useful](http://wiki.scipy.org/NumPy_for_Matlab_Users) to get started with Numpy.

<a name='numpy-arrays'></a>

### Arrays
A numpy array is a grid of values, all of the same type, and is indexed by a tuple of
nonnegative integers. The number of dimensions is the *rank* of the array; the *shape*
of an array is a tuple of integers giving the size of the array along each dimension.

We can initialize numpy arrays from nested Python lists,
and access elements using square brackets:

```python
import numpy as np

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a))            # 输出 "<class 'numpy.ndarray'>"
print(a.shape)            # 输出 "(3,)"
print(a[0], a[1], a[2])   # 输出 "1 2 3"
a[0] = 5                  # Change an element of the array
print(a)                  # 输出 "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)                     # 输出 "(2, 3)"
print(b[0, 0], b[0, 1], b[1, 0])   # 输出 "1 2 4"
```

Numpy also provides many functions to create arrays:

```python
import numpy as np

a = np.zeros((2,2))   # Create an array of all zeros
print(a)              # 输出 "[[ 0.  0.]
                      #          [ 0.  0.]]"

b = np.ones((1,2))    # Create an array of all ones
print(b)              # 输出 "[[ 1.  1.]]"

c = np.full((2,2), 7)  # Create a constant array
print(c)               # 输出 "[[ 7.  7.]
                       #          [ 7.  7.]]"

d = np.eye(2)         # Create a 2x2 identity matrix
print(d)              # 输出 "[[ 1.  0.]
                      #          [ 0.  1.]]"

e = np.random.random((2,2))  # Create an array filled with random values
print(e)                     # Might print "[[ 0.91940167  0.08143941]
                             #               [ 0.68744134  0.87236687]]"
```
You can read about other methods of array creation
[in the documentation](http://docs.scipy.org/doc/numpy/user/basics.creation.html#arrays-creation).

<a name='numpy-array-indexing'></a>

### Array indexing
Numpy offers several ways to index into arrays.

**Slicing:**
Similar to Python lists, numpy arrays can be sliced.
Since arrays may be multidimensional, you must specify a slice for each dimension
of the array:

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # 输出 "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # 输出 "77"
```

You can also mix integer indexing with slice indexing.
However, doing so will yield an array of lower rank than the original array.
Note that this is quite different from the way that MATLAB handles array
slicing:

```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # 输出 "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # 输出 "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # 输出 "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # 输出 "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"
```

**Integer array indexing:**
When you index into numpy arrays using slicing, the resulting array view
will always be a subarray of the original array. In contrast, integer array
indexing allows you to construct arbitrary arrays using the data from another
array. Here is an example:

```python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # 输出 "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # 输出 "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # 输出 "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # 输出 "[2 2]"
```

One useful trick with integer array indexing is selecting or mutating one
element from each row of a matrix:

```python
import numpy as np

# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a)  # 输出 "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # 输出 "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # 输出 "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])
```

**Boolean array indexing:**
Boolean array indexing lets you pick out arbitrary elements of an array.
Frequently this type of indexing is used to select the elements of an array
that satisfy some condition. Here is an example:

```python
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # 输出 "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # 输出 "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # 输出 "[3 4 5 6]"
```

For brevity we have left out a lot of details about numpy array indexing;
if you want to know more you should
[read the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).

<a name='numpy-datatypes'></a>

### Datatypes
Every numpy array is a grid of elements of the same type.
Numpy provides a large set of numeric datatypes that you can use to construct arrays.
Numpy tries to guess a datatype when you create an array, but functions that construct
arrays usually also include an optional argument to explicitly specify the datatype.
Here is an example:

```python
import numpy as np

x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # 输出 "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # 输出 "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # 输出 "int64"
```
You can read all about numpy datatypes
[in the documentation](http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html).

<a name='numpy-math'></a>

### Array math
Basic mathematical functions operate elementwise on arrays, and are available
both as operator overloads and as functions in the numpy module:

```python
import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))
```

Note that unlike MATLAB, `*` is elementwise multiplication, not matrix
multiplication. We instead use the `dot` function to compute inner
products of vectors, to multiply a vector by a matrix, and to
multiply matrices. `dot` is available both as a function in the numpy
module and as an instance method of array objects:

```python
import numpy as np

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
```

Numpy provides many useful functions for performing computations on
arrays; one of the most useful is `sum`:

```python
import numpy as np

x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; 输出 "10"
print(np.sum(x, axis=0))  # Compute sum of each column; 输出 "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; 输出 "[3 7]"
```
You can find the full list of mathematical functions provided by numpy
[in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.math.html).

Apart from computing mathematical functions using arrays, we frequently
need to reshape or otherwise manipulate data in arrays. The simplest example
of this type of operation is transposing a matrix; to transpose a matrix,
simply use the `T` attribute of an array object:

```python
import numpy as np

x = np.array([[1,2], [3,4]])
print(x)    # 输出 "[[1 2]
            #          [3 4]]"
print(x.T)  # 输出 "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1,2,3])
print(v)    # 输出 "[1 2 3]"
print(v.T)  # 输出 "[1 2 3]"
```
Numpy provides many more functions for manipulating arrays; you can see the full list
[in the documentation](http://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html).


<a name='numpy-broadcasting'></a>

### Broadcasting
Broadcasting is a powerful mechanism that allows numpy to work with arrays of different
shapes when performing arithmetic operations. Frequently we have a smaller array and a
larger array, and we want to use the smaller array multiple times to perform some operation
on the larger array.

For example, suppose that we want to add a constant vector to each
row of a matrix. We could do it like this:

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)
```

This works; however when the matrix `x` is very large, computing an explicit loop
in Python could be slow. Note that adding the vector `v` to each row of the matrix
`x` is equivalent to forming a matrix `vv` by stacking multiple copies of `v` vertically,
then performing elementwise summation of `x` and `vv`. We could implement this
approach like this:

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # 输出 "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # 输出 "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

Numpy broadcasting allows us to perform this computation without actually
creating multiple copies of `v`. Consider this version, using broadcasting:

```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # 输出 "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"
```

The line `y = x + v` works even though `x` has shape `(4, 3)` and `v` has shape
`(3,)` due to broadcasting; this line works as if `v` actually had shape `(4, 3)`,
where each row was a copy of `v`, and the sum was performed elementwise.

Broadcasting two arrays together follows these rules:

1. If the arrays do not have the same rank, prepend the shape of the lower rank array
   with 1s until both shapes have the same length.
2. The two arrays are said to be *compatible* in a dimension if they have the same
   size in the dimension, or if one of the arrays has size 1 in that dimension.
3. The arrays can be broadcast together if they are compatible in all dimensions.
4. After broadcasting, each array behaves as if it had shape equal to the elementwise
   maximum of shapes of the two input arrays.
5. In any dimension where one array had size 1 and the other array had size greater than 1,
   the first array behaves as if it were copied along that dimension

If this explanation does not make sense, try reading the explanation
[from the documentation](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
or [this explanation](http://wiki.scipy.org/EricsBroadcastingDoc).

Functions that support broadcasting are known as *universal functions*. You can find
the list of all universal functions
[in the documentation](http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs).

Here are some applications of broadcasting:

```python
import numpy as np

# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:
# [[ 4  5]
#  [ 8 10]
#  [12 15]]
print(np.reshape(v, (3, 1)) * w)

# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:
# [[2 4 6]
#  [5 7 9]]
print(x + v)

# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:
# [[ 5  6  7]
#  [ 9 10 11]]
print((x.T + w).T)
# Another solution is to reshape w to be a column vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))

# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
# [[ 2  4  6]
#  [ 8 10 12]]
print(x * 2)
```

Broadcasting typically makes your code more concise and faster, so you
should strive to use it where possible.

### Numpy Documentation
This brief overview has touched on many of the important things that you need to
know about numpy, but is far from complete. Check out the
[numpy reference](http://docs.scipy.org/doc/numpy/reference/)
to find out much more about numpy.

<a name='scipy'></a>

## SciPy
Numpy provides a high-performance multidimensional array and basic tools to
compute with and manipulate these arrays.
[SciPy](http://docs.scipy.org/doc/scipy/reference/)
builds on this, and provides
a large number of functions that operate on numpy arrays and are useful for
different types of scientific and engineering applications.

The best way to get familiar with SciPy is to
[browse the documentation](http://docs.scipy.org/doc/scipy/reference/index.html).
We will highlight some parts of SciPy that you might find useful for this class.

<a name='scipy-image'></a>

### Image operations
SciPy provides some basic functions to work with images.
For example, it has functions to read images from disk into numpy arrays,
to write numpy arrays to disk as images, and to resize images.
Here is a simple example that showcases these functions:

```python
from scipy.misc import imread, imsave, imresize

# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print(img.dtype, img.shape)  # 输出 "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)
```

<div class='fig figcenter fighighlight'>
  <img src='/assets/cat.jpg'>
  <img src='/assets/cat_tinted.jpg'>
  <div class='figcaption'>
    Left: The original image.
    Right: The tinted and resized image.
  </div>
</div>

<a name='scipy-matlab'></a>

### MATLAB files
The functions `scipy.io.loadmat` and `scipy.io.savemat` allow you to read and
write MATLAB files. You can read about them
[in the documentation](http://docs.scipy.org/doc/scipy/reference/io.html).

<a name='scipy-dist'></a>

### Distance between points
SciPy defines some useful functions for computing distances between sets of points.

The function `scipy.spatial.distance.pdist` computes the distance between all pairs
of points in a given set:

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print(x)

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print(d)
```
You can read all the details about this function
[in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html).

A similar function (`scipy.spatial.distance.cdist`) computes the distance between all pairs
across two sets of points; you can read about it
[in the documentation](http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html).

<a name='matplotlib'></a>

## Matplotlib
[Matplotlib](http://matplotlib.org/) is a plotting library.
In this section give a brief introduction to the `matplotlib.pyplot` module,
which provides a plotting system similar to that of MATLAB.

<a name='matplotlib-plot'></a>

### Plotting
The most important function in matplotlib is `plot`,
which allows you to plot 2D data. Here is a simple example:

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.
```

Running this code produces the following plot:

<div class='fig figcenter fighighlight'>
  <img src='/assets/sine.png'>
</div>

With just a little bit of extra work we can easily plot multiple lines
at once, and add a title, legend, and axis labels:

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```
<div class='fig figcenter fighighlight'>
  <img src='/assets/sine_cosine.png'>
</div>

You can read much more about the `plot` function
[in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot).

<a name='matplotlib-subplots'></a>

### Subplots
You can plot different things in the same figure using the `subplot` function.
Here is an example:

```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

<div class='fig figcenter fighighlight'>
  <img src='/assets/sine_cosine_subplot.png'>
</div>

You can read much more about the `subplot` function
[in the documentation](http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot).

<a name='matplotlib-images'></a>

### Images
You can use the `imshow` function to show images. Here is an example:

```python
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(img)

# Show the tinted image
plt.subplot(1, 2, 2)

# A slight gotcha with imshow is that it might give strange results
# if presented with data that is not uint8. To work around this, we
# explicitly cast the image to uint8 before displaying it.
plt.imshow(np.uint8(img_tinted))
plt.show()
```

<div class='fig figcenter fighighlight'>
  <img src='/assets/cat_tinted_imshow.png'>
</div>
