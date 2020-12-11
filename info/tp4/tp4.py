""" TP 4 """
## TREES
from treelib import Node, Tree
import json
tree = Tree()
tree.create_node("Harry", "harry")  # root node
tree.create_node("Jane", "jane", parent="harry")
tree.create_node("Bill", "bill", parent="harry")
tree.create_node("Diane", "diane", parent="jane")
tree.create_node("Mary", "mary", parent="diane")
tree.create_node("Mark", "mark", parent="jane")
tree.show()

tree.depth()

sub_t = tree.subtree('diane')
sub_t.show()

tree.move_node('mary', 'harry')
tree.remove_node('bill')
tree.show()

print(json.dumps(tree.to_json(), indent=4))

class MyTree:

    def __init__(self,n="",ch=None):
        if ch is None:
            ch = []
        self.name = n
        self.children = ch

    def add_child(self,c):
        if c not in self.children:
            self.children.append(c)

    def __repr__(self,ind=0):
        s = ""
        for _ in range(ind):
            s += " "
        s += self.name
        if self.children:
            s += ":"
            for c in self.children:
                s += "\n"
                for _ in range(ind):
                    s += " "
                s += c.__repr__(ind+1)
        return s


toto = MyTree("toto")
toto.add_child(MyTree("tutu"))
toto.add_child(MyTree("tata"))
bobo = MyTree("bobo",[toto,MyTree("kenn")])

print(bobo)

## REG EXP
import re

text = "This is a string with term1, but it does not have the other term. Just term1, really."

print(re.findall("term2",text))
print(re.findall("term1",text))

match=re.search("term1",text)
print(match)
print(match.start())
print(match.end())

"""
idiot algorithm :
text of size n
sample of size m
complexity (n x m ) :

for each letter in text
read the m next letter
compare it to sample"""

email="joe@mama.com"#input("Please type your email address :")
domain=(re.split("@",email))[1]
print("I see, so your provider is",domain,"...")

def mfindall(patterns,text):
    '''
    Takes in a list of regex patterns
    Prints a list of all matches
    '''
    for pattern in patterns:
        print("Searching for :",pattern)
        print(re.findall(pattern,text))
        print()

text = "sdsd..sssddd...sdddsddd...dsds...dsssss...sdddd"

patterns =     [ 'sd*',         # s followed by zero or more d's
                'sd+',          # s followed by one or more d's
                'sd?',          # s followed by zero or one d's
                'sd{3}',        # s followed by three d's
                'sd{2,3}',      # s followed by two to three d's -> WILL RETURN THE MAXIMUM IT COULD !
                ]

mfindall(patterns,text)

text = 'sdsd..sssddd...sdddsddd...dsds...dsssss...sdddd'

patterns = [ '[sd]',    # either s or d
            's[sd]+']   # s followed by one or more s or d


mfindall(patterns,text)

l=re.findall('[^!.? ]+',text)


from functools import reduce
print("L",l)
z = reduce(lambda x,y: x+" "+y, l, "")
print("Z",z)

text = 'This is an example sentence. Lets see if we can find some letters.'

patterns=     [ '[a-z]+',      # sequences of lower case letters
                '[A-Z]+',      # sequences of upper case letters
                '[a-zA-Z]+',   # sequences of lower or upper case letters
                '[A-Z][a-z]+'] # one upper case letter followed by lower case letters

mfindall(patterns,text)


text = 'This is a string with some numbers 1233 and a symbol #hashtag'

patterns =      [ r'\d+', # sequence of digits
                r'\D+', # sequence of non-digits
                r'\s+', # sequence of whitespace
                r'\S+', # sequence of non-whitespace
                r'\w+', # alphanumeric characters
                r'\W+', # non-alphanumeric
                ]

mfindall(patterns,text)


text = "To email Guido, try guido@python.org or the older address guido@google.com."
email = re.compile(r'\w+@\w+\.[a-z]{2,3}')
print(email.findall(text))
email.sub("xxx@xxx.xxx", text)

# email 2 regexp

email2 = re.compile(r'[\w.]+@\w+\.[a-z]{2,3}')
print(email2.findall('barack.obama@whitehouse.gov'))

ipv4pat = r"\d{0,3}\.\d{0,3}\.\d{0,3}\.\d{0,3}"
#ipv6pat = r"([0-9a-fA-F]{0,4}\:){8}"


ipv6pat = r"([0-9a-fA-F]{0,4}\:){7,7}[0-9a-fA-F]{0,4}"

#carac = r"[(" + ipv4pat + ")|(" + ipv6pat + ")]"
ippat = re.compile(r"(" + ipv4pat + r")|(" + ipv6pat + r")")
#ippat = re.compile()
print(ippat.findall("sdlkf jslkdf jslkdfj lskdfj lksdfj lksdfj oe.j j.sl.kdfj .lskdfj lksjoe sdlfk jsldkfj lskdf joje sdklf jsldkfj lskdfj oje sd flksjdfl kjjoe sdj192.167.0.1 lklsdfj lsdkf 123.123.456.123 sdmlfj lskdfj lksdjf lskdjf lskdjflskdfj lsdkfj lsdkfjs.ldf 192.167.000.001 sdlfkjsdlfkj sdlfk jsdlfkj sdlf jsdlk.\x00f fe80:0000:0000:0000:0000:0000:0000:0001: kjsdhf kjsdhf kjsdhf kjsdhf ksjdhfskjfhskdjf "))
#raise MyTree
#regex.search(text).group()
#

# cherche ab,  et le premier truc matché
# renvoie donc abab
text="nnabnnnnnnnnnnnnnnnnnnnababababababnnnnnnnnnaaaaabbbnnnnnnnnnnnnn"
myreg=re.compile(r'(ab)\1')#repete une fois de plus qu'écrit
#myreg3=re.compile(r'(ab)\3')#-> ne fonctionne pas car fait ref au groupe 3
print(myreg.search(text))
print(myreg.search("abab"))
print(myreg.search("ababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababababjoeabababababababababababababababababababababababababababababababababababababababababab"))

## Parsing

import tatsu
from tatsu.ast import AST
from pprint import pprint
import json

grammar="""
@@grammar::CALC


start
    =
    expression $
    ;


expression
    =
    | expression '+' term
    | expression '-' term
    | term
    ;


term
    =
    | term '*' factor
    | term '/' factor
    | factor
    ;


factor
    =
    | '(' expression ')'
    | number
    ;


number
    =
    /\d+/
    ;
"""

parser = tatsu.compile(grammar)
ast = parser.parse("5 * (1-2) + 50 * ( 10 - 20 )")
print(type(ast))
pprint(ast, width=20, indent=4)

# more explicite grammar

rammar="""
@@grammar::CALC


start
    =
    expression $
    ;


expression
    =
    | left:expression op:'+' right:term
    | left:expression op:'-' right:term
    | term
    ;


term
    =
    | left:term op:'*' right:factor
    | left:term '/' right:factor
    | factor
    ;


factor
    =
    | '(' @:expression ')'
    | number
    ;


number
    =
    /\d+/
    ;

"""


parser = tatsu.compile(rammar)
ast = parser.parse("5 * (1-2) + 50 * ( 10 - 20 )")
print(type(ast))
pprint(ast, width=20, indent=4)

print("op",ast.op)
print("LL",ast.left.left)
print("Lop",ast.left.op)

# navig10

print("RRL ->  ",ast.right.right.left)

#semantics
class CalcBasicSemantics(object):
    def number(self, ast):
        print("number",ast)
        return int(ast)

    def term(self, ast):
        print("term",ast)
        if not isinstance(ast, AST):
            return ast
        if ast.op == '*':
            return ast.left * ast.right
        elif ast.op == '/':
            return ast.left / ast.right
        else:
            raise Exception('Unknown operator', ast.op)

    def expression(self, ast):
        print("expression",ast)
        if not isinstance(ast, AST):
            return ast
        if ast.op == '+':
            return ast.left + ast.right
        elif ast.op == '-':
            return ast.left - ast.right
        else:
            raise Exception('Unknown operator', ast.op)


ast=parser.parse("3 + 5 * ( 10 - 20 )")
pprint(ast, width=20, indent=4)
result = parser.parse("3 +5  * ( 10 - 20 )",semantics=CalcBasicSemantics())
print("-------------")
print(result)

##
#EXERCISE
#1.
#Pre-order NLR (d'abord à gauche puis milieu puis droite )
# c'est à dire l'ordre préfixe
# philo -> regarder feuilles avant agir dessus
#2.
#sinon le cas de base est impossible car
#les entiers left et right sont appelés parfois avec term !
#les number sont des term !
#

ammar ="""
@@grammar::CALC

start
    =
    expression $
    ;


expression
    =
    | addition
    | subtraction
    | term
    ;


addition
    =
    left:expression op:'+' ~ right:term
    ;

subtraction
    =
    left:expression op:'-' ~ right:term
    ;


term
    =
    | multiplication
    | division
    | factor
    ;


multiplication
    =
    left:term op:'*' ~ right:factor
    ;


division
    =
    left:term '/' ~ right:factor
    ;


factor
    =
    | '(' ~ @:expression ')'
    | number
    ;


number
    =
    /\d+/
    ;
"""

class CalcSemantics(object):
    def number(self, ast):
        print("number",ast)
        return int(ast)

    def addition(self, ast):
        print("addition",ast)
        return ast.left + ast.right

    def subtraction(self, ast):
        print("subtraction",ast)
        return ast.left - ast.right

    def multiplication(self, ast):
        print("multiplication",ast)
        return ast.left * ast.right

    def division(self, ast):
        print("division",ast)
        return ast.left / ast.right

parser = tatsu.compile(ammar)
ast=parser.parse("3 + 5 * ( 10 - 20 )")
pprint(ast, width=20, indent=4)
result= parser.parse("3 + 5 * ( 10 - 20 )",semantics=CalcSemantics())
print("-------------")
print(result)

#Ex
#on voit que le parcours de l'arbre est plus clair
#la notion de term a disparu de l'ast final (les term sont tous transformés en mul/div
# il n'y a plus de if car aucun entier n'est un terme
# la détermination de si on est mult div ou factor se fait lors du parsing et non plus de la sémantique

type(ast)
type(ast.right)

grammar = """
@@grammar::Calc


start
    =
    expression $
    ;


expression
    =
    | addition
    | subtraction
    | term
    ;


addition::Add
    =
    left:term op:'+' ~ right:expression
    ;


subtraction::Subtract
    =
    left:term op:'-' ~ right:expression
    ;


term
    =
    | multiplication
    | division
    | factor
    ;


multiplication::Multiply
    =
    left:factor op:'*' ~ right:term
    ;


division::Divide
    =
    left:factor '/' ~ right:term
    ;


factor
    =
    | subexpression
    | number
    ;


subexpression
    =
    '(' ~ @:expression ')'
    ;


number::int
    =
    /\d+/
    ;
"""

parser = tatsu.compile(grammar, asmodel=True)
ast = parser.parse("3 + 5 * ( 10 - 20 )")
print(type(ast).__name__)
print(type(ast))
print(json.dumps(ast.asjson(), indent=4))
from tatsu.walkers import NodeWalker
class CalcWalker(NodeWalker):
    def walk_object(self, node):
        print("Object",node)
        return node

    def walk__add(self, node):
        print("Add")
        return self.walk(node.left) + self.walk(node.right)

    def walk__subtract(self, node):
        print("Subtract")
        return self.walk(node.left) - self.walk(node.right)

    def walk__multiply(self, node):
        print("Multiply")
        return self.walk(node.left) * self.walk(node.right)

    def walk__divide(self, node):
        print("Divide", node)
        return self.walk(node.left) / self.walk(node.right)

parser = tatsu.compile(grammar, asmodel=True)
ast = parser.parse("3 + 5 * ( 10 - 20 )")
print(json.dumps(ast.asjson(), indent=4))
print("---Now-Walking---")
result = CalcWalker().walk(ast)
print("---Result---")
print(result)

# Ici l'ordre infixe est utilisé
# on voit l'operateur avant le terme de gauche puis de droite
#il obtient un résultat car pas de problème de mémoire que les expressions modifieraient avant le résultat de l'autre côté (accès aux même cases mémoire par ex)

### TRANSLATOR
from tatsu.codegen import ModelRenderer
from tatsu.codegen import CodeGenerator
import sys

THIS_MODULE =  sys.modules[__name__]


class PostfixCodeGenerator(CodeGenerator):
    def __init__(self):
        super().__init__(modules=[THIS_MODULE])


class Number(ModelRenderer):
    template = '''\
    PUSH {value}'''


class Add(ModelRenderer):
    template = '''\
    {left}
    {right}
    ADD'''


class Subtract(ModelRenderer):
    template = '''\
    {left}
    {right}
    SUB'''


class Multiply(ModelRenderer):
    template = '''\
    {left}
    {right}
    MUL'''


class Divide(ModelRenderer):
    template = '''\
    {left}
    {right}
    DIV'''

parser = tatsu.compile(grammar, asmodel=True)
ast = parser.parse("3 + 5 * ( 10 - 20 )")
print(json.dumps(ast.asjson(), indent=4))
postfix = PostfixCodeGenerator().render(ast)
print(postfix)
