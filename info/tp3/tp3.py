
class WrapperList:

    def __init__(self):
        self.L = []

    def append(self,a):
        return self.L.append(a)

    def pop(self):
        return self.pop(self.L)

class MyList:

    def __init__(self,a=None,L=None):
        self.L = L
        self.a = a

    def append(self,a):
        self.L = MyList(self.a,self.L)
        self.a = a

    def pop(self):
        assert self.a is not None
        a = self.a
        if self.L is not None:
            self.a = self.L.a
            self.L = self.L.L
        else:
            self.L = self.a = None
        return a

    def __repr__(self):
        if self.L is not None:
            return str(self.L) + ',' + str(self.a)
        else:
            return str(self.a)

toto = MyList(3,MyList(4,MyList(5)))
print(toto)
toto.append(120)
print(toto)
print(toto.pop())
print(toto)
print(toto.pop())
print(toto)

class Rational:
    """This class implements exact rational numbers as tuples of integers."""

    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def __repr__(self):
        if self.denominator == 1:
            return str(self.numerator)
        elif self.numerator == 0:
            return str(self.numerator)
        else:
            return str(self.numerator)+"/"+str(self.denominator)

    def __mul__(self, number):
        if isinstance(number, int):
            return Rational(self.numerator * number, self.denominator)
        elif isinstance(number, Rational):
            return Rational(self.numerator * number.numerator, self.denominator * number.denominator)
        else:
            raise TypeError("Expected number to be int or Rational. Got "+str(type(number)))

    def __add__(self,number):
        if isinstance(number, int):
            return Rational(self.numerator + number * self.denominator, self.denominator)
        elif isinstance(number, Rational):
            return Rational(self.numerator * number.denominator + self.denominator * number.numerator, self.denominator * number.denominator)
        else:
            raise TypeError("Expected number to be int or Rational. Got "+str(type(number)))

    def _gcd(self):
        smaller = min(self.numerator, self.denominator)
        small_divisors = {i for i in range(1, smaller + 1) if smaller % i == 0}
        larger = max(self.numerator, self.denominator)
        common_divisors = {i for i in small_divisors if larger % i == 0}
        return max(common_divisors)

    def reduce(self):
        """This will simplify the fraction"""
        gcd = self._gcd()
        self.numerator = self.numerator // gcd
        self.denominator = self.denominator // gcd
        return self

    def __eq__(self,ot):
        self.reduce()
        ot = ot.reduce()
        return ot.numerator == self.numerator and self.denominator == ot.denominator

    def __lt__(self, ot):
        self.reduce()
        ot = ot.reduce()
        return ot.numerator*self.denominator > self.numerator*ot.denominator

class Rectangle():
    def __init__(self, height, length):
        self.height = height
        self.length = length

    def area(self):
        return self.height * self.length

    def perimeter(self):
        return 2 * (self.height + self.length)

class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)

    def side(self):
        return self.height

    def fathers(self):
        return self.__class__.__bases__


class HomogeneousList(MyList):
    def append(self,a):
        assert self.a is None or isinstance(a,self.a.__class__)
        self.L = HomogeneousList(self.a,self.L)
        self.a = a

tau = HomogeneousList(5,HomogeneousList(6,HomogeneousList(45)))