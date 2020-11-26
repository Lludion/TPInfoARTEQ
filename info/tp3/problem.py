## PROBLEM TP3
from regexp import *
STATENAMEGENERATOR = 1
def statenamegenerator():
    global STATENAMEGENERATOR
    STATENAMEGENERATOR += 1
    return STATENAMEGENERATOR

class State:

    def __init__(self,d=None):
        if isinstance(d,(lambda x:x).__class__):
            self.d = d
        self.name = str(statenamegenerator())

    def d(self,q=None):
        """ To be replaced """
        return []

    def __str__(self):
        return str(self.name)
    def __repr__(self):
        return str(self.name)

class Ndfa:

    def __init__(self,Q=None,S=None,d=None,s=None,F=None):
        """
        Q (set of states) should be an iterable
        S is the alphabet. If not provided, it is all Python-accepted letters
        d is the transition function. Takes a State, returns its transition function
        s is the initial states (a set)
        F is the final states (a set)
        """
        self.Q = Q
        self.S = S
        self.d = d
        self.s = s
        self.F = F
        self.c = s # current state
        if isinstance(Q,Regexp):
            self._from_regexp(Q)
        else:
            for x in self.Q:
                x.d = self.d(x)
        self.c = self.s

    def accept(self,w):
        # safety save. Should be useless
        Q = self.Q
        S = self.S
        d = self.d
        s = self.s
        F = self.F
        c = self.c
        z = self.__rec_accept(w)
        self.refresh(Q,S,d,s,F)
        self.c = c
        return z



    def __rec_accept(self,w):
        # safety save. Should be useless
        Q = self.Q
        S = self.S
        d = self.d
        s = self.s
        F = self.F
        if w:
            #print("Current Automaton:",self)
            possibil = self.c.d(w[-1])
            possepsi = self.c.d("")
            #print("Current possibillities:",possibil)
            #print("Current epsilon possis:",possepsi)
            for c in possibil:
                print(self.c , "->", c)
                self.c = c
                z = self.__rec_accept(w[:-1])
                self.refresh(Q,S,d,s,F)
                if z :return z
            for c in possepsi:
                self.c = c
                z = self.__rec_accept(w) # c has changed
                self.refresh(Q,S,d,s,F)
                if z :return z
            return False
        else:
            if self.c in self.F:
                return True
            possepsi = self.c.d("")
            print("Current Automaton:",self)
            print("Current epsilon possis:",possepsi)
            for c in possepsi:
                self.c = c
                z= self.__rec_accept(w) # c has changed
                self.refresh(Q,S,d,s,F)
                if z :return z
            return False

    def refresh(self,Q,S,d,s,F):
        self.Q = Q
        self.S = S
        self.d = d
        self.s = s
        self.F = F

    def _from_regexp(self,e):
        assert isinstance(e,Regexp)
        if isinstance(e,Or):
            a1 = Ndfa(e.a) #automaton 1
            a2 = Ndfa(e.b) #automaton 2
            self.Q = a1.Q + a2.Q + [State()]
            s = self.Q[-1]
            def espifun(x):
                if x == "":
                    return [a1.s,a2.s]
                return []
            s.d = espifun
            self.s = s
            self.F = a1.F.union(a2.F)
        elif isinstance(e,Conc):
            a1 = Ndfa(e.a) #automaton 1
            a2 = Ndfa(e.b) #automaton 2
            self.Q = a1.Q + a2.Q
            s = a1.s
            def finfun(d):
                def fun(x):
                    if x == "":
                        return [a2.s] + d("")
                    return d(x)
                return fun
            for i in a1.F:
                i.d = finfun(i.d)
            self.s = s
            self.F = a2.F
        elif isinstance(e,Star):
            a1 = Ndfa(e.a) #automaton 1
            self.Q = a1.Q
            s = a1.s
            def finfun(d):
                def fun(x):
                    if x == "":
                        return [a1.s] + d("")
                    return d(x)
                return fun
            for i in a1.F:
                i.d = finfun(i.d)
            self.s = s
            self.F = a1.F.union({self.s})
        elif isinstance(e,Epsi):
            self.Q = [State()]
            self.s = Q[0]
            self.F = {Q[0]}
            self.s.d = lambda x : (y:=[],)[0]
        elif isinstance(e,Void):
            self.Q = [State()]
            self.s = self.Q[0]
            self.F = {}
            self.s.d = lambda x : (y:=[],)[0]
        elif isinstance(e,Letter):
            self.Q = Q = [State(),State()]
            self.s = Q[0]
            self.F = {Q[1]}
            def letter(x):
                if x == e.n:
                    return [Q[1]]
                return []
            self.s.d = letter
        else:
            raise NotImplemetedError("Bad Regexp")
        self.c = self.s
        print("Current state : ",self.c)

    def determinise(self):
        raise NotImplementedError("Determinization unavailable")

    def __repr__(self):
        Q = self.Q
        S = self.S
        d = self.d
        s = self.s
        F = self.F
        c = self.c
        return "States:"+str(Q) +"\nAlphabet:"+ str(S) + str(d)+"\nInitial:"+str(s)+"\nFinal:"+str(F)+"\nCurrent:"+str(c)

class FiniteAutomata(Ndfa):
    def __init__(self,Q=None,S=None,d=None,s=None,F=None):
        super().__init__(Q=Q,S=S,d=d,s=s,F=F)

    def __rec_accept(self,w):
        """ Only change is the addition of the assert statements """
        # safety save. Should be useless
        Q = self.Q
        S = self.S
        d = self.d
        s = self.s
        F = self.F
        if w:
            possibil = self.c.d(w[-1])
            possepsi = self.c.d("")
            assert len(possibil) < 2
            assert len(possepsi) < 2
            for c in possibil:
                print(self.c , "->", c)
                self.c = c
                z = self.__rec_accept(w[:-1])
                self.refresh(Q,S,d,s,F)
                if z :return z
            for c in possepsi:
                self.c = c
                z = self.__rec_accept(w) # c has changed
                self.refresh(Q,S,d,s,F)
                if z :return z
            return False
        else:
            if self.c in self.F:
                return True
            possepsi = self.c.d("")
            assert len(possepsi) < 2
            for c in possepsi:
                self.c = c
                z = self.__rec_accept(w) # c has changed
                self.refresh(Q,S,d,s,F)
                if z :return z
            return False

NonDetFiniteAutomata = Ndfa


a_b_c = Or(Or(Letter("A"),Letter("B")),Letter("C"))
nabc = Ndfa(a_b_c)

ad = Conc(Letter("A"),Letter("D"))
nadd = Ndfa(ad)

joe = Or(Conc(Letter("E"),Conc(Letter("O"),Letter("J"))),Conc(Letter("A"),Conc(Letter("M"),Conc(Letter("A"),Letter("M")))))
nj = Ndfa(joe)

joem = Star(Or(Conc(Letter("E"),Conc(Letter("O"),Letter("J"))),Conc(Letter("A"),Conc(Letter("M"),Conc(Letter("A"),Letter("M"))))))
njm = Ndfa(joem)

void = Conc(Letter("A"),Void())
nv = Ndfa(void)

toto = FiniteAutomata(joem)
print(toto.accept("JOEJOEMAMAJOEMAMA"))