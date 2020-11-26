
class Regexp:

    def __init__(self,a,b,n=" "):
        self.a = a
        self.b = b
        self.n = n

class Or(Regexp):

    def __init__(self,a,b):
        self.n = "0"
        super().__init__(a,b,self.n)

    def generate(self):
        return self.a.generate(w)

class Conc(Regexp):

    def __init__(self,a,b):
        self.n = "6"
        super().__init__(a,b,self.n)

    def check(self,w):
        return self.a.check(w) or self.b.check(w)

class Star(Regexp):

    def __init__(self,a,b=None):
        assert b is None
        self.n = "5"
        super().__init__(a,b,self.n)

class Epsi(Regexp):

    def __init__(self,a=None,b=None):
        """ Epsilon class """
        assert b is None
        assert a is None
        self.n = "3"
        super().__init__(a,b,self.n)
class Letter(Regexp):

    def __init__(self,letter="A",a=None,b=None):
        """ Letter class """

        assert b is None
        assert a is None
        self.n = letter
        super().__init__(a,b,self.n)

class Void(Regexp):

    def __init__(self,a=None,b=None):
        """ Void class """
        assert b is None
        assert a is None
        self.n = "7"
        super().__init__(a,b,self.n)
