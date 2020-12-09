import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
import qiskit
from qiskit.providers.aer import *
from numpy import pi,conj,array,arange
from numpy.random import rand
var = 2*pi*rand(36) - pi*array([1 for i in range(36)])

def vabs(l):
    """ vectorized abs """
    return [abs(x) for x in l]

def prob(d,i,j,bi,bj):
    """ proba d'après le dict d d'avoir les bits bi bj en i et j respectivement """
    r = 0
    for k,v in d.items():
        if k[i] == bi and k[j] == bj:
            r += v # v= d[k]
    return r

def ZZ(d,i,j):
    return prob(d,i,j,'0','0') - prob(d,i,j,'0','1') - prob(d,i,j,'1','0') + prob(d,i,j,'1','1')

def somH(sv):
    """ calcul de la fonction à minimiser, connaissant le psi

    il s'agit du cost """
    sum = 0
    for i in [0,1,2,3]:# pour
        for j in [0,1,2,3]: # chaque
            if (i,j) != (1,2) and i != j and (i,j) != (1,2):# arête
                sum += ZZ(sv,i,j)
    return sum

def circ(var):
    """ crée le circuit voulu """
    qreg_q = QuantumRegister(4, 'q')
    creg_c = ClassicalRegister(4, 'c')
    circuit = QuantumCircuit(qreg_q, creg_c)

    circuit.h(qreg_q[0])
    circuit.h(qreg_q[1])
    circuit.h(qreg_q[2])
    circuit.h(qreg_q[3])
    circuit.u(var[0], var[1], var[2], qreg_q[0])
    circuit.u(var[3], var[4], var[5], qreg_q[1])
    circuit.u(var[6], var[7], var[8], qreg_q[2])
    circuit.u(var[9], var[10], var[11], qreg_q[3])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cx(qreg_q[3], qreg_q[0])
    circuit.u(var[12], var[13], var[14], qreg_q[0])
    circuit.u(var[15], var[16], var[17], qreg_q[1])
    circuit.u(var[18], var[19], var[20], qreg_q[2])
    circuit.u(var[21], var[22], var[23], qreg_q[3])
    circuit.cx(qreg_q[0], qreg_q[1])
    circuit.cx(qreg_q[1], qreg_q[2])
    circuit.cx(qreg_q[2], qreg_q[3])
    circuit.cx(qreg_q[3], qreg_q[0])
    circuit.u(var[24], var[25], var[26], qreg_q[0])
    circuit.u(var[27], var[28], var[29], qreg_q[1])
    circuit.u(var[30], var[31], var[32], qreg_q[2])
    circuit.u(var[33], var[34], var[35], qreg_q[3])
    return circuit

def int_to_bin(n):
    """ écrit l'entier en binaire """
    return '{0:04b}'.format(n)

def f(var,returnpsi =False):
    """
    Ceci est la fonction à minimiser, qui va calculer :

    - |psi> H <psi|
     """
    circuit = circ(var)

    sim = qiskit.providers.aer.Aer.get_backend('statevector_simulator')
    job = execute(circuit, sim, shots=4096)
    sv = job.result().get_statevector(circuit)

    if returnpsi:
        return sv
    # transforme state vector en un dictionnaire de probas
    dictsv = {}
    for i in range(len(sv)):
        dictsv[int_to_bin(i)] = abs(sv[i]) * abs(sv[i])
    return somH(dictsv)

arr = scipy.optimize.minimize(f,var,method="Powell")#None)#,bounds=[[-pi,pi] for _ in range(36)])

## SHOW

res = f(arr.x,True)
print("finale:",res)

plt.plot(arange(16),vabs(res))
plt.show()