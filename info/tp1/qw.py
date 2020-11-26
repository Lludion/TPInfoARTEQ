
from matplotlib.pyplot import *
from numpy import *


def vectordot(a,b):
    return kron(a,b)
vd = vectordot

class QW:
    def __init__(s,N=100,d=1):
        s.d = d
        s.warning_use()
        s.N = N
        s.P = 2 * N + 1
        P = s.P
        s.coin0 = coin0 = array([1,0],dtype='complex')
        s.coin1 = coin1 = array([0,1],dtype='complex')

        s.C00 = outer(coin0, coin0) # |0><0| 
        s.C01 = outer(coin0, coin1) # |0><1| 
        s.C10 = outer(coin1, coin0) # |1><0| 
        s.C11 = outer(coin1, coin1) # |1><1|

        #default operators
        #Id
        s.Ip = eye(P)
        #evolution
        s.H = (s.C00 + s.C01 + s.C10 - s.C11)/sqrt(2.)
        #scattering
        s.ShiftPlus = roll(eye(P,dtype='complex'), 1, axis=0) # 1 après la diagonale
        s.ShiftMinus = roll(eye(P,dtype='complex'), -1, axis=0) # 1 avant
        s.S_hat = kron(s.ShiftPlus, s.C00) + kron(s.ShiftMinus, s.C11)

        s.C_hat = s.H
        s.probabuilt = False

        # Unused unless Dirac Walk :
        s.theta = 0
        s.c = cos(s.theta)
        s.s = sin(s.theta)

        #Initialisation of positions is not done upon creation

    def warning_use(self):
        if self.d > 1:
            print("The dimension may be too large for this specific class to handle.\n Try using a more powerful class instead.")

    def dirac(self,theta=0.1):
        
        # new values
        self.theta = theta
        self.c = c = cos(theta)
        self.s = s = sin(theta)

        self.C_hat = c * self.C00 + s * self.C01 - s * self.C10 + c * self.C11

    def _renewU(self):
        self.U = self.S_hat.dot(kron(eye(self.P), self.C_hat))

    def _initpos(self,d=1):
        P = self.P
        N = self.N
        coin0 = self.coin0
        coin1 = self.coin1
        
        ## Initial position
        self.posn0 = zeros(P)
        self.posn0[N] = 1 # array indexing starts from 0, so index N is the central p
        self.base_state =  (coin0+coin1*1j)/sqrt(2.)
        self.psi0 = vd(self.posn0,self.base_state)

        # au tps N, etat est de :
        self.psiN = linalg.matrix_power(self.U, N).dot(self.psi0)

    def prob(self):
        """ builds the array of probabilities if necessary """
        if not self.probabuilt:
            self.prob = empty(self.P)
            for k in range(self.P):
                posn = zeros(self.P)
                posn[k] = 1
                M_hat_k = kron( outer(posn,posn), eye(2))
                proj = M_hat_k.dot(self.psiN)
                self.prob[k] = proj.dot(proj.conjugate()).real
            self.probabuilt = True

    def _P10(self,p=None):
        if p is None:
            p = self.P
        return int(p / 10)

    def plotprob(self,o=None,size=None):
        """ plots the array of probabilities """
        size = self.P
        N = self.N
        fig = figure()
        ax = fig.add_subplot(111)
        plot(arange(size), self.prob)
        if o is not None:
            plot(arange(size), prob, o)
        loc = range (0, size, self._P10(size)) #Location of ticks

        # gives the right label
        xticks(loc)
        xlim(0, size)
        ax.set_xticklabels(range (-N, N+1, self._P10(size)))

        show()

    def pp(self,o=None,size=None):
        return self.plotprob(o=o,size=size)

    def exe(self,t="dirac",plot=True):
        """ computes the entire walk and draws it """
        self._renewU()
        self._initpos()
        if "irac" in t:
            self.dirac()
        self.prob()
        if plot:
            self.plotprob()

if __name__ == '__main__':
	qw = QW()
	qw.exe()

