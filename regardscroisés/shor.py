import numpy as np
from math import pi
from time import time
from qiskit import *
from qiskit.circuit import *
from qiskit.extensions import *
from qiskit.circuit.library import *
from qiskit.extensions.simulator.snapshot import snapshot
from qiskit.quantum_info.operators import Operator
from qiskit.extensions.simulator.snapshot import snapshot
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
from scipy import optimize
from matplotlib.pyplot import plot,show
STARTTIME = time()

size_eig = 4
size_phi = 2

eig = QuantumRegister(size_eig)
phi = QuantumRegister(size_phi)
ceig = ClassicalRegister(size_eig)
cphi = ClassicalRegister(size_phi)
qc = QuantumCircuit(eig,phi,ceig,cphi)

for i in range(size_phi):
    qc.x(phi[i])

for i in range(size_eig):
    qc.h(eig[i])

U = UnitaryGate(
    Operator([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,np.exp(pi*2j*(5/16))]]), label="U")

contU = U.control()

for j in range(size_eig):
    qc.append(contU.power(2**(j)),[list(eig)[size_eig-1-j]]+list(phi))

qc.append(QFT(size_eig).inverse(),list(eig))

for i in range(size_eig):
    qc.measure(eig[i], ceig[i])

for i in range(size_phi):
    qc.measure(phi[i], cphi[i])

qc.draw()
print(qc)
show()
input("joe")
