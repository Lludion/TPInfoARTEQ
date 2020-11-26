"""
Memoization of functions without decorators,
Lludion 2020
"""

import types
import functools
import inspect
import sys
import ast

def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g

def memoize(f):
    """ Memoises a function """
    dico = {}
    def aux(x):
        if x not in dico:
            dico[x] = f(x)
        return dico[x]
    return aux

## With decorators, it is trivial to memoise a non-memoized function
@memoize
def goodfibo(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return goodfibo(n-1) + goodfibo(n-2)

def fibotrash(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibotrash(n-1) + fibotrash(n-2)

## It is more difficult to memoize a function that is already defined...
def all_functions():
	return [(name,obj) for (name,obj) in inspect.getmembers(sys.modules[__name__]) if (inspect.isfunction(obj))]

def newmemo(f):
    L = all_functions()
    g = copy_func([fu for (n,fu) in L if n == f.__name__][0])
    print("This is an internal variable equal to the external function :",g)
    g = memoize(g)
    print(g(32))# this does not work !
    # This will only work with a global variable !
    return g

## The instruction is :
# f= memoize(f) 
#(with the same f before and after memoize,
# and f is global (or has the same scope as where it was defined)

def test_working_global():
	##This works :
	global fibotrash
	oldg = copy_func(fibotrash)
	fibotrash = memoize(fibotrash)#memoize(fibotrash)


	print(fibotrash(400))
	## However, the old fibotrash is lost.
	print(oldg(400))


## What can be done to retrieve the old version is this :
funname = 'fibotrash'
def resurrect(funname):
	""" recreates an old function """
	with open("memoization.py", "r") as source:
		tree = ast.parse(source.read())
	# This type of considerations is useless, as we will compile stuff and extract the fun name from the namespace
	#mod  = [x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == funname][0]
	#tree = ast.parse(mod, mode='exec')
	code = compile(tree, filename='toto', mode='exec')
	namespace = {}
	exec(code, namespace)
	return namespace[funname]

def test_failing():
	rottingfibo = resurrect('fibotrash')
	print("This will take forever, as we use the old fibo :")
	rottingfibo(400)

## However, with the same idea, one can bulid a function
# That does exactly what is expected :
def memo(f):
	""" recreates an old function and memoizes it"""
	with open("memoization.py", "r") as source:
		tree = ast.parse(source.read())
		code = compile(tree, filename='toto', mode='exec')
		namespace = {}
		exec(code, namespace)
	""" namespace[funname] is defined here, so we can use the same 
	recursive definition trick that we previously used on global variables """
	namespace[funname] = memoize(namespace[funname])
	print("Any function application here works :")
	print(namespace[funname](400))
	return namespace[funname]

def test_amazing(stop=False):
	newfibo = memo(fibotrash)
	print("This is working perfectly:",newfibo(340))
	if stop:
		print("But the global function is not modified : (comment this line to see further tests)")
		print(fibotrash(300))
		print("And externally, fibotrash is not modified as well;")

if __name__ == '__main__':
	print("1. This proves that memo works even when fibotrash is not modified globally:")
	test_amazing()
	
	print("2.The next test will work. (uses global variables, simpler version, not scalable)")
	test_working_global()
	#It carries the new value of the function globally :
	print(fibotrash(200))
	print("3. This proves that memo works even when fibotrash is modified globally:")
	test_amazing(True)
	print("4.The next test will fail.")
	test_failing()
	


"""


## Further examples :
#print(goodfibo(400))
#print("Defining fibo..")
#fibo = newmemo(fibotrash)
#print("Correct definition")
#print(fibo)
#print(all_functions())
#print(fibo(330))
"""
