import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import *
from numpy import *
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter#Used for 3d plotting

def vectordot(a,b):
    return tensordot(a,b,0)
vd = vectordot


def sp(U,v):
    """ shaped product of a array of dimensions
    U : 3 x 3 x 3 x 3 x 2 x 2
    with an array of dimensions :
    v :   3   x   3   x   2   

    """
    return np.einsum('ijklmo,ikm->ikm', U, v)

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
        self.psi0 = vp(self.posn0,self.base_state)

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
        if size is None:
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

#qw = QW()
#qw.exe()
## QW2
class QW2D(QW):
    def __init__(self,N=100,M=None):
        """ Two dimensional QW
        To do as it was shown in the slides and in the 
        basic QW, we use kron (kroenecker product) instead
        of tensordot to implement tensorial products
        """
        super().__init__(N)
        if M is None:
            M = N
        self.M = M
        P = self.P
        self.Q = Q  = 2 * M + 1
        self.dimensions = (self.P,self.Q)
        self.Un = None

        self.X = array([[0,1],[1,0]],dtype='complex')
        
        #scattering
        self.ShiftPlus = vectordot(roll(eye(P,dtype='complex'), 1, axis=0),eye(Q,dtype='complex')) # 1 après la diagonale
        self.ShiftMinus = vectordot(roll(eye(P,dtype='complex'), -1, axis=0),eye(Q,dtype='complex')) # 1 avant
        self.S_hat = vd(self.ShiftPlus, self.C00) + vd(self.ShiftMinus, self.C11)

        self.ShiftLeft = vectordot(eye(P,dtype='complex'),roll(eye(Q), -1, axis=0))
        self.ShiftRight = vectordot(eye(P,dtype='complex'),roll(eye(Q), 1, axis=0)) 
        self.T_hat = vd(self.ShiftLeft, self.C00) + vd(self.ShiftRight, self.C11)

    def init(self):        
        self._renewU()
        self._initpos()

    def warning_use(self):
        if self.d > 2:
            print("The dimension may be too large for this specific class to handle.\n Try using a more powerful class instead.")

    def _renewU(self):
        # CεHT1,εHT2,ε
        mx = kron(eye(self.P*self.Q), self.C_hat)
        #print(mx)
        self.H_hat = vd(eye(self.P),vd(eye(self.Q), self.H))
        self.U = self.S_hat #@ self.H_hat# simpler version
        print(self.maxU(),self.U.shape)
        #self.T_hat @ self.H_hat @ self.S_hat @ self.H_hat @ mx

    def _initpos(self,d=2):
        P = self.P
        N = self.N
        Q = self.Q
        M = self.M
        
        coin0 = self.coin0
        coin1 = self.coin1
        
        ## Initial position
        self.posn0 = vd(zeros(P,dtype='complex'),zeros(Q,dtype='complex'))
        self.posn0[N][M] = 1 # array indexing starts from 0, so index N is the central p
        self.base_state =  coin0#(coin0+coin1*1j)/sqrt(2.)
        self.psi0 = vd(self.posn0,self.base_state)
        #print(self.psi0)
        #print(self.U)

        # au tps N, etat est de :
        self.psiN = sp(self.UN() , (self.psi0))
        #print(self.psiN)
        
    def prob(self):
        """ builds the array of probabilities if necessary """
        if not self.probabuilt:
            self.prob = np.abs(self.psiN)**2
            Z = np.einsum('ikm->ik',self.prob)
            self.probpos = Z
            self.probabuilt = True

    def plotprob(self,o=None,size=None):
        """ plots the array of probabilities """
        if size is None:
            size = self.P
        N = self.N

        #fig = figure()
        #ax = fig.gca(projection='3d')
        #X = arange(-self.N,self.N+1,1)
        #Y = arange(-self.M,self.M+1,1)
        Z = self.probpos
        plt.imshow(Z, interpolation='bilinear')
        #print(X.shape,Y.shape,Z.shape)
        #
        #my_col = cm.jet(Z/np.amax(Z))
        #surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, linewidth=0, cmap=cm.jet,antialiased=True)
        show()

    # test and debugging functions
    def UN(self):
        if self.Un is None:
            self.Un = linalg.matrix_power(self.U, self.N)
        return self.Un

    def shsh(self,sh=True):
        """ return the shifts or the shift shape """
        if sh:
            return self.ShiftLeft.shape
        else:
            return self.ShiftPlus,self.ShiftMinus,self.ShiftLeft,self.ShiftRight

    def maxU(self,z=None):
        if z is None:
            z = self.U
        i = 0
        for l in z:#P
            for l2 in l:#P
                for l3 in l2:#Q
                    for l4 in l3:#Q
                        for l5 in l4:#2
                            for l6 in l5:#2
                                if l6 != 0:
                                    i += 1
        print(i)
## tests

qw = QW2D(1)
qw.init()
qw.exe(plot=False)


"""M = array([[1,2,3],[7,8,9],[11,12,13]])
print(roll(M,(1,0),(1,0)))
print(M)
print("")

print(roll(M,-1))
print(M)
print("")

print(roll(M,1,1))
print(M)
print("")

posn = zeros(self.dimensions)
self.prob = empty(self.dimensions)
for k in range(self.P):
    for q in range(self.Q):
        posn[k,q] = 1
        M_hat_k = kron( outer(posn,posn), eye(2))
        proj = M_hat_k.dot(self.psiN)
        self.prob[k] = proj.dot(proj.conjugate()).real
        posn[k,q] = 0
"""

