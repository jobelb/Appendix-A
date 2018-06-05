#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:45:13 2018
Summarized Notes from Appendix A
@author: jobelb
"""
#to run just one line of command, hit "ctrl + return" rather hitting F5
print "hello world"


#The " " and ' ' are called strings
x = 2
pi = 3.141592658
label = "Addition:"
print label, x, '+' , pi, '=', (x +pi)


label = 'Division:'
print label, x, '/', pi, '=', (x +pi)


#Use ** for exponentiation
#use % to find remainder b/w two numbers
print 2 ** 3
print 7 % 3

#combination of arithmetic operations
x = 4
x += 2    #means x = x + 2
print x

#Container Objects#
#List - holds an ordered sequence, accessed by using zero-based item index [ ]
L = [1, 2, 3]
print L[0]

#can be combination
pi = 3.14
L = [5, 'dog', pi]
print L[-1]

#slicing - allows access to sublists
L = ['Larry', 'Goran', 'Curly', 'Peter', 'Paul', 'Mary']
print L[0:3]   #slice containingg first three items
print L[:3]    #same, zero is implied

print L[-2:]  #last two items
print L[-2]   #output is just the second to the last item with no brackets


print L[1:4]  # item 1 is inclusive through 4 non-inclusive
              #will list "2nd" item but really #1, till third. don't include 4th
              
print L[::2]   #every 2nd item
print L[::-1]  #all items in reverse order
print L[::3]   #will always start with #0


#Tuples - like list, but use ( ), once created, item in them can't be changed
# commonly used in fuctionswhich return multiple values
#Sets - can be used when no repetition of values is desired
S = set([1, 1, 3, 2, 1, 3])
print S


#Dictionaries - stores an unordered sequence of key value pairs
#use { } annd allows mixes of types
D = {'a':1, 'b':2.5, 'L':[1, 2, 3]}
print D['a']
print D['L']


#A.3.5 Functions
#oprations w/c will be repeatedly executed, use functions
#convenient to define a function w/c iplements desired operation
#a function can optionally accept one/several Parameters

def convert_to_radians(deg): #keyword - def, function name - convert_to_radians
        pi = 3.141592653         #arguments in ( ), colon : - marking beginning
        return deg * pi / 180.0                               #code block
print convert_to_radians(90)      #local variable - pi, optional return statement
                                  # ..returns a computed calue to the point of
                                  # ..the function call
# define a sequential list of integers - built-in range function
#x = range(10     #SAYS SYNTAX ERROR, GO BACK TO THIS
#print x 
                                  
#A.3.6  Logical Operators
#boolean values F/T
#nonboolean variables - can be coereced   intro boolean types
    #types - nonzero integer evaluates to True, zero integer = False
    
# x or y = True for either or both True, otehrwise False
# x and y = True if both x and y evaluate to True, otherwise False
# not x = True only if x evaluates to False, otherwise False

#Comparison expressions
    # == means equal
    # != means not equal
    # >= , <=
#non empty string evaluates to true
    
print bool(''), bool('hello')

x = 2
y = 4
print (x == 2) and (y >= 3)

#A.3.7 Control FLow - conditionals and loops - allows programmer to control
    #the order of code execution
# Commands: if, elif, else
def check_value(x):
     if x < 0:
         return 'negative'
     elif x == 0:
         return 'zero'
     else:
         return 'positive'
print check_value(0)
print check_value(123.4)

#two types of Loops: for and while loops
#syntax is...
#1. or loops - used often used with the built-in range function
for drink in ['coffee', 'slivovitz', 'water']:
    print drink
     
words = ['twinkle', 'twinkle', 'little', 'star']
for i in range(4):
    print i, words[i]
    
#2. while loops - operates similarly but in other languages
i = 0
while i < 10:
    i += 3 
    
print i

#continue and break
i = 0
while True:
    i += 3
    if i > 10:
        break      #break out of the loop        
print i

for i in range(6):
    if i % 2 == 0:
        continue      #continue from beginning of loop block
    print i, 'only odd numbers get here'
 

#null operation - useful as placeholder
word = 'python'
for char in word:
    if char == 'o':
        pass    #this does nothing
    else:
        print char,

#A.3.8 Exceptions and Exception Handling
# Exception - when something goes wrong in Python
'''        
L = [1, 2, 3]
print L[5]     #output says out of range
'''
#to fix.. useful to Catch the exceptions --> use try, except statement

def safe_getitem(L, i):
    try:
        return L[i]
    except IndexError:
        return 'index out of range'
    except TypeError:
        return 'index of wrong type'

L = [1, 2, 3]
print safe_getitem(L, 1)
print safe_getitem(L, 100)
print safe_getitem(L, 'cat')

# Raise an exception
def laugh(n):
    if n <= 0:
        raise ValueError('n must be positive')
    else:
        return n * 'ha! '
print laugh(6)
#print laugh(-2)  #output says n must be positive

'''
A.3.9 Modules and Packages
Module - a collection of functions and variables
Package - a collection of modules
  ..can be accessed throguh "import" statement
'''

import math  
print math.sqrt(2)

from math import pi,sin  #can also import other specific variables, functions, 
print sin(pi / 2)             #or classes from modules

'''
help(math) - built-in help function with the module as an argument
dir(math)  - lists all the attributes of a module/object - will list all
                operations available
'''
#print dir(math)
#print help(math)

#A.3.10 Objects in Python
'''
Objects - bldng blocks of python
every variable in python is an object
variable containing integer is an **object type "int"**
**list is object of type "list"**
Objects - may have Attributes, Methods, and/or Properties
'''
#Ex: of Attributes and Methods
c = complex(1.0, 2.0)
print c   #ouput is (1+2j) --??

print c.real   #attribute
print c.imag   #attribute  
print c.conjugate()   #method   #output is (1-2j) - computes the complex 
                                     # conjugate p.484 

#Append method of "list" objects can be used to extend the array:
L = [4, 5, 6]
L.append(8)
print L

# short method - used to sort a list in place:
numbers = [5,2,6,3]
numbers.sort()
print numbers

words = ['we', 'will', 'alphabetize', 'these', 'words']
words.sort()
print words

#A.3.11 Classes: User-Defined Objects - users can create own objects
class MyObject(object): # derive from the base-class 'object'
    def __init__(self, x):    #__init__() when subj. is initialized
        print 'initializing with x =' , x#first argument is "self", 2nd=upto u
        self .x = x

    def x_squared(self):
        return self.x ** 2
    
obj = MyObject(4)
print obj .x    #access attribute
print obj .x_squared()    # invoke method


#A.3.12 Documentation Strings
'''
docstrings = built-in documentation strings
accessed by calling help() function on the object
'''
def radians(deg):
    """
    Convert an angle to radians
    """
    pi = 3.141592653
    return deg * pi / 180.0

help(radians)  #opens a text reader showing docstring defined at top

'''
A.4 IPython: THe Basics of Interactive COmputing - use IPython -->
A.4.1 Command History - to see history, type  "print In[#]
   - for output, type "print Out[#]"
A.4.2 Command Completion  - type first few letters,use tab or up arrow key
A.4.3 Help and Documentation - type "range?" on IPython console
                - can also do double "??"
                   exampe:  def myfunc(x):
                                return 2 * x
                                myfunc?? -NO SPACE
                                
A.4.4 Magic Functions - shortcuts to useful functions, marked by % p488
         dp % then tab to see other commands
'''

'''
A..5 Intro to NumPy - scientific computing


#A.5.1 Array Creation - use "import numpy as np"
#IN- np.aray([5, 2, 6, 7])  - creates an array from a python list
#OUT- array ([5,2,6,7])
       #in -  np.arange(4)
       #out - array ([0, 1,2,3])
# np.linspace(1,2,5)  - means 5 evenly-spaced steps from 1-2
# np.zeros(5)  - means array of zeros
# np.ones(6) - means array of ones
       # multidimentionsal - np.zeros((2,44)) - means 2x4 array of zeros
#Random array creation
       #np.random.random(size=4) means uniform b/w 0 and 1
       #np.random.normal(size=4) means standard norm distribution
  
A.5.2 Element Access
x = np.arange(10)
IN- x[0]
OUT- 0

IN- x[::2]
OUT- array([0, 2, 4, 6, 8])

Multidimensional arrays - element access uses multiple indices P. 490**

A.5.3 Array Data Type
dtype parameter - controls the types array
~~
x = np.zeros(1, dtype=int)
x[0] = 3.14   - converted to an integer
x[0]
out: 3   

A.5.4 Universal Functions and Broadcasting
ufunc = universal function 
 x = np.arange(3) #[0, 1, 2]
in: np.sin(x)  #take the sine of each element
out: array([ 0.        ,  0.84147098,  0.90929743])

binary ufuncs - ufuncs with two parameters (arithmetic operations)
in: x * x   #multiply each element of x  by itself
out: array([0, 1, 4])

can also operate between array and scalars ~ applying the operations + scalar
IN: x + 5  # add 5 to each element of x
OUT: array([5, 6, 7])
'''

#Broadcasting - an argument of a ufunc has a shape upgraded to the shape 
         #of the other argument; more complicated is adding a vector to it
#IN: x = np.ones((3, 3))   # 3x3 array of ones
#IN: y = np.arange(3)   #[0, 1, 2]
#IN: x + y  #add y to each row oof x
#OUT: array([[ 1.,  2.,  3.],
      # [ 1.,  2.,  3.],
      # [ 1.,  2.,  3.]])
#Sometimes, both arrays are broacast at the same time
    #  x = np.arange(3)
    #  y = x.reshape((3, 1))  # create a 3 x 1 column array
    #  x + y
      
    
   # next p.492

'''
Update June 6, 2018
'''
#A.5.5 Structured Arrays
#Numpy contains structured ararys - store compound dara types
import numpy as np
print np.arange(3) + 5
print np.ones((3,3)) + np.arange(3)
print np.arange(3).reshape((3,1)) + np.arange(3)

#store all in one array - each object has integer ID + 5-char string name
dtype = [('ID', int), ('name', (str, 5)), ('value', float)]
data = np.zeros(3, dtype=dtype)
print data

#Fileds of array can be thought of as "Columns"
data['ID'] = [154, 207, 669]
data['name'] = ['obj_1', 'obj_2', 'obj_3']
data ['value'] = [0.1 , 0.2 , 0.3]
print data[0]
print data[2]
print data['value']

#A.6 Visualitzation with Matplotlib - basic plotting examples "%pylab"

#%pylab
import numpy as np
import matplotlib.pyplot as plt

#%matplotlab
#Simple Sinusoid Plot
x = np.linspace(0, 2 * np.pi, 1000)   #1000 valuess from 0 to 2pi
y = np.sin(x)
ax = plt.axes()
ax.plot(x, y)
ax.set_xlim(0, 2 * np.pi)   #set x limits
ax.set_ylim(-1.3, 1.3)   #set y limits
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Simple Sinusoid Plot')

#Simple Sinusoid Plot + Error bar plot type - use plt.errorbar
x_obs = 2 * np.pi * np.random.random(50)
y_obs = np.sin(x_obs)
y_obs += np.random.normal(0, 0.1, 50)   #add some error
ax.errorbar(x_obs, y_obs, 0.1, fmt='o' , color='black')

#Histogram Plotting - executed through ax.hist()
fig = plt.figure()  #create a new fig window
ax = plt.axes()   #create new axes
x = np.random.normal(size=1000)
ax.hist(x, bins=50)
ax.set_xlabel('x')
ax.set_ylabel('N(x)')
'''
use "plt.hist?" on iPython to control appearance of histogram
Other plots: 
    plt.scatter = create scatter plot
    plt.imshow = display images
    plt.contour and plt.contourf =  disp contour plots
    plt.hexbin = craete hexagonall tessellations
    plt.fill and plt.fill_between = draw filled regions
    plt.subplot and plt.axes = create  multiple subplots
go to astroml.org for more source codes
'''
#Overview of Useful NumPy/SciPy Modules - collection of routines & tools
#A.7.1 Reading and Writing Data
#use np.savetxt( ) to save an array to an ASCII file
#use np.loadtxt( ) to load texr files

import numpy as np
x = np.random.random((10, 3))
np.savetxt('x.txt', x)
y = np.loadtxt('x.txt')
print np.all(x == y)

# np.genfromtxt - customziable text loader
# np.save and np.load - written to binary files for single arrays
# np.savez - to store multiple arrays in a single zipped file

#A.7.2 Pseudorandom Number Generation
np.random.seed(0)
print np.random.random(4)   #uniform between 0 and 1
print np.random.normal(loc=0, scale=1, size=4)  #standard norm
print np.random.randint(low=0, high=10, size=4)  #random integers

#np.random to see more
#scipy.stats - generates random variables using the distributions submodule in it
from scipy.stats import distributions
print distributions.poisson.rvs(10, size=4)
# scipy.distributions? for more info

#A.7.3 Linear Algebra
#basic dot (i.e., matrix-vector) product is implemented in NumPy
M = np.arange(9).reshape((3,3))
print M

x = np.arange(1, 4)  #[1, 2, 3]
print np.dot(M, x)

#numpy.linalg and scipy.linalg = more common lin alg operations available
# numpy.linalg? or scipy.linalg? in IPython for more info

'''
#A.7.4 Fast Fourier Transforms -an imp. algorithm in many aspects of data 
  analysis. Routines are available in NumpY (numpy.fft submodule) and
SciPy (scipy.fftpack submodule) <-SciPy has more options to control the results
'''
x = np.ones(4)
print np.fft.fft(x)   #forward transform "fft()"
print np.fft.ifft([ 4.+0.j,  0.+0.j,  0.+0.j,  0.+0.j]) #inverse "ifft()"
#scipy.fftpack? or numpy.fft? for more info

#A.7.5 Numerical Integration
#QUADPACK - an optimized numerical integration package written in Fortran, to
#integrate function sinx from 0 to (pi symbol) 
from scipy import integrate
result, error = integrate.quad(np.sin, 0, np.pi)
print result, error
# integrate? in IPython for more info

#A.7.6 Optimization
#use scipy.optimize
#ex: find the min of a simple function usinnng "fmin"
from scipy import optimize
def simple_quadratic(x):
    return x ** 2 + x
print optimize.fmin(simple_quadratic, x0=100)

'''
#A.7.7 Interpolation
# use scipy.interpolate to implement, from simple linear & poly interpolation
# to more sophisticated spline-based techniques
Spline fit has interpolated the sampled pts to create a smooth curve.
Additional Options: adjustment of smoothing factors, number of knots, 
weighting of points, spline degre, etc.

use interpolate.interp1d for Linear and Polu interpolation
type interpolate? in IPython for more info

'''
#%pylab
from scipy import interpolate
fig = plt.figure()
x = np.linspace(0, 16, 30)    #coarse grid: 30 pts
y = np.sin(x)
x2 = np.linspace(0, 16, 1000)  #fine grid: 1000 pts
spl = interpolate.UnivariateSpline(x, y, s=0)
ax = plt.axes()
ax.plot(x, y, 'o')  # 'o' means draw pts as circles
ax.plot(x2, spl(x2), '-')   # '-' means draw a line

'''
#A.7.8 Other Submodels - can be explored ussing online doc. and IPython's help
scipy.spatial: distance and spatial functions, nearest neighbor, Delaunay
tessellation

scipy.sparse: sparse matrix storage, sparse linear algebra, sparse solvers,
sparse graph traversal

scipy.stats: common statistical functions and distributions

scipy.special: special functions (e.g., Airy functions, Bessel functions,
orthogonal polynomials, gamma functions, and much more)

scipy.constants: numerical and dimensional constants
'''
#A.8 Efficient Coding with Python and NumPy
#below are few common examples made using Python and NumPy, and solutions to
# improve computation time
# use %timeit magic command, provides quick benchmark for Python execution
'''
#A.8.1 Data Structures
Python list object - good for small sequences for faster exec.
'''
#Example
L = range(10000000)  # a large list
#%timeit sum(L)   #1 loop, best of 3: 513 ms per loop = Slow
                    #only works when executed on IPython console?

import numpy as np     #better exec time when using NumPy rather than the
x = np.array(L, dtype=float)           # built-in Python list
#%timeit np.sum(x)   #100 loops, best of 3: 4.63 ms per loop-faster for NumPy

'''
GUIDELINE 1: Store data in NumPy arrays, not Python lists when there is a 
    sequence/list of data larger than a few dozen items. Store and manipulate
    as a NumPy array.


#A.8.2 Loops
If an algorithm seemms to require loops, it's better in Python to implement it
using VECTORIZED oprations in NumPy (ufuncs in A.5.4)
'''
# Example
import numpy as np
x = np.random.random(10000000)
def loop_add_one(x):
    for i in range (len(x)):
        x[i] += 1
        
#%timeit loop_add_one(x)   #1 loop, best of 3: 3.15 s per loop - longer

import numpy as np
x = np.random.random(10000000)       
def vectorized_add_one(x):
    x += 1
#%timeit vectorized_add_one(x)  #100 loops,best of 3: 7.83 ms per loop - faster

'''
Using VECTORIZED oprations (enabled by NumPy's ufuncs) for repeated operations 
leads to much faster code.

GUIDELINE 2: Avoid large loops in favor of vectorized operations. 
Vectorized methods within NumPy will be a better choice.
'''

#A.8.3 Slicing, Masks, and Fancy Indexing
#Slicing (done above) generally very fast, preferable to looping through 
#the arrays. Below is an ex. of a slicing operation:
x[:len(x) / 2] = x[len(x) / 2:]
#example - we have an array and we want every value >0.5 to be changed to
#999. Someone MIGHT DO THIS. (This is longer..read on for a faster method)
x[:len(x) / 2] = x[len(x) / 2:]
def check_vals(x):
    for i in range (len(x)):
        if x[i] > 0.5:
            x[i] = 999
#%timeit check_vals(x)  #1 loop, best of 3: 1.76 s per loop
            
#same operation can be performed FASTER using a boolean mask:
x[:len(x) / 2] = x[len(x) / 2:]
#%timeit x[(x > 0.5)] = 999   # vectorized version 
              #10 loops, best of 3: 69.3 ms per loop
              
#Masks can be combined using the bitwise operators & for AND, | for OR, ~ for
#NOT, and ^ for XOR. Example someone might write:
#x[(x < 0.1) | (x > 0.5)] = 999  be careful on using parenthesis
             # around boolean methods

#Fancy Indexing = indexing with lists
#example - the longer way
import numpy as np
X = np.random.random((10000000, 3))
def get_random_rows(X):
    X_new = np.empty(X.shape)
    for i in range(X_new.shape[0]):
        X_new[i] = X[np.random.randint(\
                X.shape[0])]
    return X_new

#Fancy Indexing can SPEED things up by generating the array of indices
   # all at once and vectoriziing the operation
import numpy as np
X = np.random.random((10000000, 3))       #NOTE: Fancy Indexing is much slower 
def get_random_rows_fast(X):               #than slicing for eq operations
    ind = np.random.randint(0, X.shape[0], #Remember ufuncs in A.5.4
                            X.shape[0])
    return X[ind]
'''
#NOTE: Fancy Indexing is much slower than slicing for eq operations
#Remember ufuncs in A.5.4
Most manipulation tasks can be accomplished w/o writing a loop.

GUIDELINE 3: Use array Slicing, Masks, Fancy Indexing, and 
    Broadcasting to Eliminate Looops

NOTE: Python loops are slow, and NumPy array tricks can be used to
sidestep this problem. Also, there are some algorithms for which loop
elimination through vectorization is difficult or impossible.

A.9 Some algorithms can be wrapped with Fortran, C, or C++ code for use
within Python. Packages like NumPy, SciPy and Skit-leaarn use seveeral of these
tools both to implement efficient algorithms and make se of lib packages
written in Fortran, C,, and C++
'''