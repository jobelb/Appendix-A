#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 16:45:13 2018
Summarized Notes from Appendix A
@author: jobelb
"""

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
      

