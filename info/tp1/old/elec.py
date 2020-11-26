import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import *
from numpy import *

N = 100
P = 2 * N + 1

coin0 = array([1,0])
coin1 = array([0,1])

C00 = outer(coin0, coin0) # |0><0| 
C01 = outer(coin0, coin1) # |0><1| 
C10 = outer(coin1, coin0) # |1><0| 
C11 = outer(coin1, coin1) # |1><1|

#default operators
#Id
Ip = eye(P)
#evolution
H = (C00 + C01 + C10 - C11)/sqrt(2.)
#scattering
ShiftPlus = roll(eye(P), 1, axis=0) # 1 après la diagonale
ShiftMinus = roll(eye(P), -1, axis=0) # 1 avant
S_hat = kron(ShiftPlus, C00) + kron(ShiftMinus, C11)

# U = S_hat.dot(kron(eye(P), C_hat))

# new values
theta = 0.1
c = cos(theta)
s = sin(theta)

C_hat = c * C00 + s * C01 - s * C10 + c * C11

U = S_hat.dot(kron(eye(P), C_hat))

## Initial position
posn0 = zeros(P)
posn0[N] = 1 # array indexing starts from 0, so index N is the central p
psi0 = kron(posn0,(coin0+coin1*1j)/sqrt(2.))
# au tps N, etat est de :
psiN = linalg.matrix_power(U, N).dot(psi0)

# build the array of probabilities
prob = empty(P)
for k in range(P):
    posn = zeros(P)
    posn[k] = 1
    M_hat_k = kron( outer(posn,posn), eye(2))
    proj = M_hat_k.dot(psiN)
    prob[k] = proj.dot(proj.conjugate()).real

# plot proba

fig = figure()
ax = fig.add_subplot(111)
plot(arange(P), prob)
#plot(arange(P), prob, 'o')
loc = range (0, P, int(P / 10)) #Location of ticks
xticks(loc)
xlim(0, P)
ax.set_xticklabels(range (-N, N+1, int(P / 10)))
show()
