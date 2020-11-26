import ast

with open("tp2.py", "r") as source:
    tree = ast.parse(source.read())
    functions = [x.name for x in ast.walk(tree) if isinstance(x, ast.FunctionDef)]



class Analyzer(ast.NodeVisitor):
    def __init__(self):
        self.stats = {"import": [], "from": []}

    def visit_Import(self, node):
        for alias in node.names:
            self.stats["import"].append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.stats["from"].append(alias.name)
        self.generic_visit(node)

    def get_fun(self,tree):
        self.tree = tree
        self.funcs = [x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef)]
        self.funnames = [x.name for x in self.funcs]

    def memoize(self,funname):
        assert funname in self.funnames
        self.fun = [x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == funname][0]
        print(self.fun)
        print(self.fun.__class__.__name__)
        object_methods = [method_name for method_name in dir(self.fun)]
        print(object_methods)
        print([getattr(self.fun, method_name) for method_name in object_methods])
        print("",end="\n\n\n\n\n\n\n")
        print(self.fun)
        print("",end="\n\n\n\n\n\n\n")
        print(ast.dump(ast.parse(self.fun)))
        ast.parse(self.fun).name

    def report(self):
        print(self.stats)

if False:

    analyzer = Analyzer()
    analyzer.visit(tree)
    analyzer.report()
    analyzer.get_fun(tree)
    analyzer.memoize("fibo")

##


def memoize(f):
    dico = {}
    def aux(x):
        if x not in dico:
            dico[x] = f(x)
        return dico[x]
    return aux

@memoize
def fibo(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibo(n-1) + fibo(n-2)

import functools
def newmemoize(f):

    def meemoize(fu):
        dir(__main__)
        dico = {}
        def aux(x):
            if x not in dico:
                dico[x] = fu(x)
            return dico[x]
        aux.__name__ = f.__name__

        return aux
    global f.__name__
    z = meemoize(f.__name__)

    return

def fibotrash(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibotrash(n-1) + fibotrash(n-2)


@functools.lru_cache(maxsize=None)
def figarbo(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return figarbo(n-1) + figarbo(n-2)

