"""
EITQ , TP 2
"""

delt = lambda x : x(x)

# ♪lmao
#(lambda y : 3)(delt(delt))

def power(f,n):
  def aux(f,n,x=0):
    if n == 0:
      return x
    else:
      return aux(f,n-1,f(x))
  return lambda x : aux(f,n,x)

f = lambda x: x *2
print(power(f,5)(3))
print(3*2**5)

#from sympy import isprime
#print(*[i for i in range(1,100) if isprime(i)], sep=", ")

def syracuse(n):
  if n%2:
    return 3*n+1
  else:
    return n//2

syracuse(10)

def syracusesteps(x):
    def aux(x,y=0):
        if x == 1:
            return y
        else:
            return aux(syracuse(x),y+1)
    return aux(x)

print(syracusesteps(35))
##
l= [32,13,21,321,321,321,321,321,32,12,12,15,45,46,21,56,4,15,1,35,1,564,61,65,1]
from timeit import default_timer
from random import *

def genlist(n):
    return sample(range(0, n), n)

def linearsearch(x,l):
    return x in l

st = default_timer()
print("Start: ",st)
st = default_timer()
print(linearsearch(60,l))
en = default_timer()
print("End: ",en)
print("Duration",en-st)

def timeforsize(n):
    l = genlist(n)
    x = randint(0,n)
    start = default_timer()
    linearsearch(x,l)
    end = default_timer()
    return end-start

timeforsize(1000)
def average(f,n,k):
    tries=[f(n) for i in range(0,k)]
    return sum(tries)/len(tries)
import matplotlib.pyplot as plt
#3%matplotlib inline
import sys
sys.setrecursionlimit(10**6)

def plotef(n):
    x=list(range(0,n,n//200))
    y=[average(timeforsize,i,100) for i in x]
    plt.plot(x, y)
    plt.xlabel('size')
    plt.ylabel('time')
    plt.title('Efficiency of linear search')
    plt.show()


def listmap(f,l):
    """
    behaves just like `map`, albeit directly returning a list instead.
    """
    return [f(x) for x in l]

def listfilter(f,l):
    """
    behaves just like `filter`, albeit directly returning a list instead.
    """
    return [x for x in l if f(x)]
##
from functools import reduce

def lenreduce(L):
    return reduce(lambda x,y : x+1, L, 0)

lenreduce(list(range(1000)))


def listreduce(f,l):
    """which behaves just like `reduce`."""
    return reduce(f,l)

def listreduce(f,l,acc=None):
    """which behaves just like `reduce`."""
    z = acc
    for e in l:
        z = f(z,e)
    return z

def lenreduce2(L):
    return listreduce(lambda x,y : x+1, L, 0)
print(lenreduce2(list(range(1000))))

##
from numpy import sqrt

x = 0
def fibo(n):
    global x
    x=x+1
    if n > 1:
        return fibo(n-1)+fibo(n-2)
    else:
        return n

def plotfib(n):
    global x
    X = list(range(0,n))
    Y = []
    x=0
    for i in range(n):
        fibo(i)
        Y.append(x)
        x=0

    Z = [ ((1+sqrt(5))/2)**x  for x in range(0,n)]
    plt.plot(X, Z)

    plt.plot(X, Y)
    plt.xlabel('size')
    plt.ylabel('time')
    plt.title('Efficiency of naive fibo')
    plt.show()

fibodico = {1:1,0:0}
def memofibo(n):
    global x
    x=x+1
    if n > 1:
        try:
            return fibocheck(n)
        except:
            return fibocheck(n-1)+fibocheck(n-2)
    else:
        return n

def fibocheck(n):
    global x
    global fibodico
    x=x+1
    try:
        return fibodico[n]
    except:
        fibodico[n] = fibocheck(n-1) + fibocheck(n-2)
        return fibodico[n]

def plotmemofib(n):
    global x
    global fibodico
    X = list(range(0,n))
    Y = []
    W = []
    x=0
    #for i in range(n):
    #    fibo(i)
    #    Y.append(x)
    #    x=0
    #plt.plot(X, Y)

    #Z = [ ((1+sqrt(5))/2)**x  for x in range(0,n)]
    #plt.plot(X, Z)


    x=0
    for i in range(n):
        memofibo(i)
        W.append(x)
        x=0
        fibodico = {1:1,0:0}
    plt.plot(X, W)

    plt.xlabel('size')
    plt.ylabel('time')
    plt.title('Efficiency of memoized fibo')
    plt.show()

##

def memoize(f):
    """ without new functions"""
    global dico
    dico = {}
    def wrapper(f):
        def aux(*args,**kwargs):
            try:
                return dico[args]
            except:
                z = f(*args,**kwargs)
                dico[args] = z
                return z
        return aux
    @wrapper
    def f2(*args,**kwargs):
        return f(*args,**kwargs)

    return wrapper(f)

def exponential(n):
    if n==0:
        return 1
    else :
        return 2*exponential(n-1)

def newexponential(g,n):
    if n==0:
        return 1
    else :
        return 2*g(n-1)

def newfibo(f,n):
    global x
    x=x+1
    if n > 1:
        return f(n-1)+f(n-2)
    else:
        return n

def memint(newf):
    global dico
    dico = {}
    def aux(n):
        global dico
        global x
        x += 1
        try :
            return dico[n]
        except:
            z = newf(aux,n)
            dico[n] = z
            return z
    return aux

toto = memint(newfibo)

def plotmemnewfib(n):
    global x
    global fibodico
    X = list(range(0,n))
    Y = []
    W = []
    x=0
    #for i in range(n):
    #    fibo(i)
    #    Y.append(x)
    #    x=0
    #plt.plot(X, Y)

    #Z = [ ((1+sqrt(5))/2)**x  for x in range(0,n)]
    #plt.plot(X, Z)


    x=0
    for i in range(n):
        toto(i)
        W.append(x)
        x=0
        fibodico = {1:1,0:0}
    plt.plot(X, W)

    plt.xlabel('size')
    plt.ylabel('time')
    plt.title('Efficiency of new fibo')
    plt.show()

##
def power(f,n):
  if n:
    return lambda x: f(power(f,n-1)(x))
  else:
    return lambda x : x

def returntime(f):
  def aux(f,n,pow):
    if pow:
      return aux(f,n+1,f(pow))
    else:
      return n
  return lambda x : aux(f,0,x)

add5 = lambda x : x +5
power(add5,7)(2)
remove3 = lambda x : x//2
returntime(remove3)(32)