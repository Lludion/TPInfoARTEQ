import matplotlib.pyplot as plt
import numpy as np
from numpy import array,roll,eye,kron,zeros,linalg,dot,arange
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter#Used for 3d plotting
from qw import QW,vd,vectordot

def sp(U,v):
	"""
	alias for dot ?
	"""
	return dot(U,v)

def XYrecons(Z,Q,prob=True):
	"""
	Reconstructs X and Y from a Z.
	Z : the initial array
	Q : dimension of the reconstructed array
	prob : whether the two coordinates in each position have been merged
	"""
	if not prob:
		raise NotImplementedError("Only proba implemented")
	X = []
	for x in range(Q):
		Y = []
		for y in range(Q):
			Y.append(Z[x*Q+y])
		X.append(Y)
	return array(X)

## QW2
class QW2D(QW):
	def __init__(self,N=5,M=None):
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
		print(self.posn0)
		self.posn0[N*P+M] = 1 # array indexing starts from 0, so index N is the central p
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
			z = []
			loc = 0
			for i in range(len(self.prob)):
				if i%2:
					z.append(loc+self.prob[i])
				else:
					loc = self.prob[i]
			Z = np.array(z)
			self.probpos = Z
			self.probabuilt = True

	def plotprob(self,o=None,size=None):
		""" plots the array of probabilities """
		if size is None:
			size = self.P
		N = self.N

		fig = plt.figure()
		ax = fig.gca(projection='3d',aspect='auto')
		X = arange(-self.N,self.N+1,1)
		Y = arange(-self.M,self.M+1,1)
		Z = self.probpos

		print("Initial Z :",Z)
		#my_col = cm.jet(Z/np.amax(Z))
		Z = XYrecons(Z,self.Q)
		plt.imshow(Z,interpolation='bilinear',aspect='auto')
		print('Interpolation successful !')
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.5, linewidth=0, cmap=cm.jet,antialiased=True)
		plt.show()

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
		for m in z:#P
			for n in m:#Q
				i += int(bool(n))
		return i

if __name__ == '__main__':
	## tests
	qw = QW2D(10)
	qw.init()
	qw.exe(plot=True)



