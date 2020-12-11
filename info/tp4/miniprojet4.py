""" Implémentation en Python de Mini-ML """

import tatsu
from tatsu.ast import AST
from pprint import pprint
import json
from tatsu.walkers import NodeWalker
grammar = """
@@grammar::Calc


start
    =
    expression $
    ;


expression
    =
    | apporop

    | fst
    | snd

    | addition
    | subtraction

    | liaison


    | term

    | subexpression
    | pair
    ;

apporop
    =
    app
    ;

addition::Add
    =
    left:term op:'+' ~ right:expression
    ;


subtraction::Substract
    =
    left:term op:'-' ~ right:expression
    ;


term
    =
    | multiplication
    | division
    | factor
    | var
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
    |'(' ~ @:expression ')'
    |@:funparenth
    ;


number::int
    =
    /\d+/
    ;

pair::Pair
    =
    '[' ~ pl:expression ',' ~ pr:expression ']';

fun::Function
    =
    'lambda' ~ xfun:var ~ ':' ~ corps:expression ;

funparenth
    =
    |'(' ~ @:fun ')'
    |@:fun
    ;

var::Var
    =
    name:/[a-z]\w*/;

app::App
    =
    afun:expression  aexp:expression;

liaison::Liaison
    =
    lvar:var '=' lexp:expression ";" ~ lcorps:expression;

fst::Fst
    =
    'Fst' fspa:pairorvar;

snd::Snd
    =
    'Snd' snpa:pairorvar;

pairorvar
    =
    | pair
    | var;
"""
parser = tatsu.compile(grammar, asmodel=True)

#not totally useful
class CalcSemantics(object):
    def number(self, ast, *args,**kwargs):
        print("number",ast)
        return int(ast)

    def addition(self, ast, *args,**kwargs):
        print("addition",ast)
        return ast.left + ast.right

    def subtraction(self, ast, *args,**kwargs):
        print("subtraction",ast)
        return ast.left - ast.right

    def multiplication(self, ast, *args,**kwargs):
        print("multiplication",ast)
        return ast.left * ast.right

    def division(self, ast, *args,**kwargs):
        print("division",ast)
        return ast.left / ast.right

    def fst(self, ast, *args,**kwargs):
        print("fst",ast)
        return ast.fspa[0]

    def snd(self, ast, *args,**kwargs):
        print("snd",ast)
        return ast.snpa[1]

    def pair(self, ast, *args,**kwargs):
        print("pair",ast)
        return [ast.pl,ast.pr]

    def liaison(self, ast, *args,**kwargs):
        print("liaison",ast)
        return ast.lcorps

    def fun(self, ast, *args,**kwargs):
        print("fun",ast)
        #ceci est nimporte quoi
        return lambda x : ast.corps

    def app(self, ast, *args,**kwargs):
        print("app",ast)
        return (self.afun)(self.aexp)

class CalcWalker(NodeWalker):
    def walk_object(self, node, env={}, rep=[]):
        print("Object",node)
        return node

    def walk__add(self, node, env={}, rep=[]):
        print("Add")
        wl = self.walk(node.left, env=env, rep=rep)
        wr = self.walk(node.right, env=env, rep=rep)
        print("ADDL",wl)
        print("ADDR",wr)
        if isinstance(wr,(int,float)) and isinstance(wl,(int,float)):
            return wl + wr
        else:
            return node

    def walk__subtract(self, node, env={}, rep=[]):
        print("Subtract")
        wl = self.walk(node.left, env=env, rep=rep)
        wr = self.walk(node.right, env=env, rep=rep)
        if isinstance(wr,(int,float)) and isinstance(wl,(int,float)):
            return wl - wr
        else:
            return node

    def walk__multiply(self, node, env={}, rep=[]):
        print("Multiply")
        wl = self.walk(node.left, env=env, rep=rep)
        wr = self.walk(node.right, env=env, rep=rep)
        if isinstance(wr,(int,float)) and isinstance(wl,(int,float)):
            return wl * wr
        else:
            return node

    def walk__divide(self, node, env={}, rep=[]):
        print("Divide", node)
        wl = self.walk(node.left, env=env, rep=rep)
        wr = self.walk(node.right, env=env, rep=rep)
        if isinstance(wr,(int,float)) and isinstance(wl,(int,float)):
            return wl / wr
        else:
            return node

    def walk__fst(self, node, env={}, rep=[]):
        print("Fst", node)
        return self.walk(node.fspa, env=env, rep=rep)[0]

    def walk__snd(self, node, env={}, rep=[]):
        print("Snd", node)
        return self.walk(node.snpa, env=env, rep=rep)[1]

    def walk__pair(self, node, env={}, rep=[]):
        print("Snd", node)
        return (self.walk(node.pl, env=env, rep=rep),self.walk(node.pr, env=env, rep=rep))

    def walk__var(self, node, env={}, rep=[]):
        print("Var", node.name)
        #print("env = ",env)
        #print(rep,node.name)
        if node.name in rep:
            try:
                newval = env[node.name]
                print("--------REPLACING--",node.name,"---------")
                print("oldvalue :",node)
                print("newvalue :",newval)
                return self.walk(newval, env=env, rep=rep)
            except:
                return node.name
        else:
            try:
                zval = env[node.name]
                #print(node.name,"OK")
            except:
                #print(node.name,"NON OK")
                return node.name
            #print("ZVAL",zval,env,)
            return get_zval(zval,env,rep,self)

    def walk__liaison(self, node, env={}, rep=[]):
        print("Liaison", node)
        newenv = dict(env, **{ self.walk(node.lvar) : self.walk(node.lexp,env=env, rep=rep)})
        #print("NEW ENV : ",newenv)
        return self.walk(node.lcorps, env=newenv, rep=rep)

    def walk__function(self, node, env={}, rep=[]):
        print("Function", node)# surtout pas de rep = rep a gauche!
        return self.walk(node.xfun,env),self.walk(node.corps,env,rep=rep)

    def walk__app(self, node, env={}, rep=[]):
        print("App", node)
        f = self.walk(node.afun, env=env, rep=rep)
        xf,_ = f#on extrait le ptit nom de la variable
        newenv2 = dict(env, **{ xf : self.walk(node.aexp,env=env, rep=rep)})
        #print("NEWenv definition ----------------",newenv2)
        print("We will try to replace all",xf)
        print("We will try to replace by",newenv2[xf])
        f = self.walk(node.afun, env=newenv2, rep=rep+[xf])
        #print("ENDnew env definition ------------",xf)
        _,cf = f#variable, corps
        return self.walk(cf, env=newenv2, rep=rep)

def get_zval(zval,env,rep,texasranger):
    variables = []
    while True:
        try:
            #print("Unpacking" ,zval)
            #print("0,1",zval[0],zval[1])
            variables.append(zval[0])
            zval =  zval[1]
            #print("--------------------------------------------------")
        except:
            #print("Did not unpack",zval)
            break
    newzval = texasranger.walk(zval,env,rep)
    zval = newzval
    for var in variables:
        zval = (var, zval)
    return zval


"""
#useless code and useful examples
try :
    f[0] = f[0]#utilise la non mutabilité des tuples pour encoder le type
    xf,cf = f#variable, corps
    newenv = dict(env, **{ self.walk(node.lvar) : (xf,cf), xf : cf})
except:
    newenv = dict(env, **{ self.walk(node.lvar) : lexpval})

    def walk__app(self, node, env={}, rep=[]):
        print("App", node)
        f = self.walk(node.afun, env=env)
        try :
            f[0] = f[0]#utilise la non mutabilité des tuples pour encoder le type
            #si mutable, c'est une fonction
            xf,cf = f#variable, corps
            #newenv = dict(env, **{ self.walk(node.lvar) : (xf,cf), xf : cf})
            newenv2 = dict(env, **{ xf : self.walk(node.aexp,env=env)})
        except:
            xf = self.walk(node.afun, env=env)
            #newenv = dict(env, **{ self.walk(node.lvar) : lexpval})
            newenv2 = dict(env, **{ xf : self.walk(node.aexp,env=env)})
            return self.walk(node.afun.corps, env=newenv2)
        #print("THIS IS F:",f)
        #xf,cf = f#variable, corps


        return self.walk(node.afun.corps, env=newenv2)

pprint(ast, width=20, indent=4)
print(json.dumps(ast.asjson(), indent=4))

#exemples

ast = parser.parse("fst [10/10,12]",semantics=CalcSemantics())

print(ast)
ast = parser.parse(" 3 * 2 + 4 ")
ast = parser.parse("joe = 5 ; fst [10/10,12]",semantics=CalcSemantics())

print(ast)
ast = parser.parse("z = 15; y = ( lambda x : x + z ) ; y 5 + y 12 ")
ast = parser.parse("distpair = (lambda f : (lambda p : [f (Fst p), f (Snd p)] )) ; distpair (lambda z : z + 1) [2,3]")
ast = parser.parse("x = 1; y = 2 ; ( ( lambda z : z) x ) + y")
ast = parser.parse("x = 1; y = 2 ; z = (lambda a : a) 2 ; x + y")
ast = parser.parse("xaafuun = Fst [ 2 , 1 ] ;\n toto = [3 + 5 * ( ras = [45,1]; 10 - 20 ),[10+1,(lambda x : x + 5) 5]]; xafuun = 5; (Fst toto) + xaafuun")
"""
print("---AST---")
# il calcule un peu n'importe quoi mais ne bugue pas si on parenthèse bien
# l'ordre de liaison est un peu contre intuitif aussi, me semble t'il
ast = parser.parse(" plus = (lambda a: (lambda b : (a + b))); (plus 145) ((lambda m : ((plus 140) m) + 13) 2)")
print(ast)
print("---Now-Walking---")
result = CalcWalker().walk(ast)
print("---Result---")
print(result)

