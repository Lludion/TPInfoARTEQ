#!/usr/bin/env python
# coding: utf-8

# # 0 - Des librairies à charger au démarrage

# In[2]:


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
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # Makes the images look nice")


# ## Quelques fonctions utiles
# 
# - `nat2bl` : d'un entier naturel à une liste de `0` et de `1` (bit de poid faible à droite)
# - `nat2bs` : d'un entier naturel à une chaine de caractères telle que `"010010011"`
# - `bl2bs` : d'une liste à une chaine
# - ... et les fonctions dans l'autre sens.

# In[3]:


def nat2bl(pad,n):
    if n == 0 :
        return [0 for i in range(pad)]
    elif n % 2 == 1:
        r = nat2bl(pad-1,(n-1)//2)
        r.append(1)
        return r
    else:
        r = nat2bl(pad-1,n//2)
        r.append(0)
        return r

for i in range(16):
    print(nat2bl(5,i))


# In[76]:


def bl2nat(s):
    if len(s) == 0:
        return 0
    else:
        a = s.pop()
        return (a + 2*bl2nat(s))

def bl2bs(l):
    if len(l) == 0:
        return ""
    else:
        a = l.pop()
        return (bl2bs(l) + str(a))

def nat2bs(pad,i):
    return bl2bs(nat2bl(pad,i))

def bs2bl(s):
    l = []
    for i in range(len(s)):
        l.append(int(s[i]))
    return l

def bs2nat(s):
    return bl2nat(bs2bl(s))


print(nat2bs(10,17))
print(bs2nat("0011010"))
##

# # 1 - Introduction
# 
# Dans ce TP nous allons coder le coeur de l'algorithme de Shor : le circuit permettant de trouver la période de la fonction 
# 
# $$
# x \mapsto a^x mod N.
# $$
# 
# Il s'agit donc de d'abord réaliser QPE, puis l'oracle calculant la multiplication modulo, puis de combiner tout ça.
# 
# C'est parti.
# 

# # 2 - QPE
# 
# Nous avons vu l'algorithme de QPE en cours sur 3 qubits, 
#et vous avez eu à vérifier qu'il faisait ce qu'il fallait
#. Ici nous allons l'implémenter ici pour l'opérateur $U$ suivant.

# In[61]:


U = UnitaryGate(
    Operator([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,np.exp(pi*2j*(5/16))]]), label="U")


# ## Questions 
# 
# ###  Q 2.1 Préliminaire
# 
# * Que réalise cet opérateur ? (`2j` est en python le nombre complexe $2\cdot i$)
# 
# C'est une Cphase gate (controlled phase )
# 
# * Sur combien de qubits agit-il ?
# 
# sur 2 qubits 
# 
# 
# * Quels sont ses valeurs/vecteurs propres ?
# 
# les vecteurs de la base, puisque diagonale
# 
# * Pour chaque valeur propre, que doit retourner QPE avec 3 bits de précision, tel que vu en cours ?
# 
# 000 (-> exp(0) = 1)
# 000 (-> exp(0) = 1)
# 000 (-> exp(0) = 1)
# 110 (-> exp(2pi i 6/8))
# 
# 

# ### Q 2.2 Implémentation de QPE
# 
# Ci-dessous un template à remplir pour
# - réaliser QPE avec 3 bits de précisions.
# - sur le vecteur propre de valeur propre non-trivial
# 
# On a initialisé un circuit quantique avec 3 registres: 
#  - `eig` qui stockera les valeurs propres finales
#  - `phi` pour stocker le vecteur propre sur lequel travailler
#  - `ceig` pour stocker le résultat de la mesure des valeurs propres.
#  
# Notez qu'on a uniquement besoin de mesurer les valeurs propres !
# 
# Ce dont vous aurez besoin:
#  - `QFT(size)` génère pour vous une QFT sur `size` qubits.
#  - `U.control()` réalise la porte `U` controlé avec un qubit. Le qubit controlé doit être placé en début de liste.
#  - `U.inverse()` réalise l'inversion de la porte `U`.
#  - `U.power(n)` va mettre `p` fois `U` sur le circuit.
#  - `qc.append(U, liste_de_qubits)` applique la porte `U` sur la liste de qubits en entrée.
#  - Attention : `phi` (par exemple) n'est pas une liste mais un registre. Donc si vous voulez la concaténer avec autre chose, il faut d'abord fabriquer une liste avec `list(phi)`.

# In[69]:


size_eig = 4
size_phi = 2

eig = QuantumRegister(size_eig)
phi = QuantumRegister(size_phi)
ceig = ClassicalRegister(size_eig)
cphi = ClassicalRegister(size_phi)
qc = QuantumCircuit(eig,phi,ceig,cphi)
"""
for i in range(size_phi):#quelconque phi
    qc.h(phi[i])
"""
#phi = |0001>
qc.x(phi[1])
qc.x(phi[0])
#qc.h(phi[0]) superposition


for i in range(size_eig):
    qc.h(eig[i])

#for j in size_phi

contU = U.control()

for j in range(size_eig):
    qc.append(contU.power(2**(j)),[list(eig)[size_eig-1-j]]+list(phi))
#qc.append(contU.power(4),[list(eig)[0]]+list(phi))
#qc.append(contU.power(2),[list(eig)[1]]+list(phi))
qc.append(QFT(size_eig).inverse(),list(eig))

for i in range(size_eig):
    qc.measure(eig[i], ceig[i])

for i in range(size_phi):
    qc.measure(phi[i], cphi[i])


qc.draw()

# D'abord, assurez-vous que le dessin est correct


# In[70]:


# Puis lancez l'exécution.

backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=10000)
job.result().get_counts(qc)


# ### Q 2.3 Analyse
# 
# - Est-ce le résultat attendu ?
# Oui !!
# 
# - Faitez varier le $\frac68$ de la phase de $U$: en $\frac18$, puis en $\frac28$... Est-ce que QPE vous retourne la bonne réponse
# 
# oui!!
# 
# - Augmentez la précision : passez le nombre de bits de `eig` à $4$, et changez la fraction de phase de $U$ en $\frac{10}{16}$ : est-ce que QPE vous retourne bien $10$ en binaire ?
# - Passez à $5$ bits de précision: est-ce que cela marche toujours ?
# 
# ### Q 2.4 Pour aller plus loin
# 
# Nous avions vu que le circuit de la QPE travaille sans problème avec une superposition de vecteurs propres. Essayez donc de changer l'initialisation de `phi` avec 
# $$
# \frac1{\sqrt2}(|\phi_1\rangle + |\phi_2\rangle),
# $$
# deux vecteurs propres de $U$ (l'un de valeur propre triviale, l'autre non).
# 
# on obtient un melange statistique des deux phases !
# 
# Mesurez aussi le registre `phi` à la fin du circuit, et analysez le résultat : que constatez-vous ?
# 
# Comme prévu, psi n'a pas été modifié (on obtient aussi la phase associée à chaque état) 

# # 3 - Synthèse d'oracle
# 
# La fonction $f : x\mapsto (a^p\cdot x)~mod~N$ est une bijection de $\{0...N-1\}$ dans $\{0...N-1\}$ si $a$ et $N$ sont premiers entre eux. 
# 
# Dans ce cas, on peut regarder $f$ comme un operateur unitaire agissant sur un espace de Hilbert de dimension $N$.
# On peut donc regarder $f$ comme un unitaire agissant sur un registre de qubits, pour autant que $N$ soit une puissance de $2$.
# 
# C'est un peu limité : on souhaite pouvoir considerer des nombres $N$ quelconques. On va donc plutôt considérer la fonction suivante:
# $$
# Mult_{a^p~mod~N} : x\mapsto 
# \left\{
# \begin{array}{ll}
# (a^p\cdot x)~mod~N & \text{si }x < N
# \\
# x & \text{si} N \leq x < 2^n
# \end{array}\right.
# $$
# pour autant que $N<2^n$.
# 
# Cette nouvelle fonction $Mult_{a^p~mod~N}$ est bien une bijection de $\{0..2^n\}$: c'est celle que nous allons implémenter. Ici, nous allons faire quelque chose de simple, basé sur la synthèse d'opérateurs de QisKit. Ce n'est pas le plus rentable en terme de circuit, mais c'est le plus simple.
# 
# 

# ## Questions
# 
# ### Q 3.1 Opérateur de la multiplication modulo.
# 
# - Complétez la fonction ci-joint pour que `M` soit une matrice implémentant la bijection $Mult_{a^p~mod~N}$.
# - Testez avec le code en dessous, avec $a = 1, 3, 6, 11, 15$.

# In[80]:


# a, p, N et n sont comme au dessus : 
# on calcule x |--> a^p * x mod N, vu comme un operateur sur C^{2^n}

def gateMult(a,p,N,n):
    nn = 2 ** n
    M = [[0 for x in range(nn)] for i in range(nn)]

    for x in range(nn):
        if x >= N:
            M[x][x] = 1
        else:
            apmodn = (x * (a ** p) ) % N
            M[apmodn][x] = 1

    U = Operator(M)
    return(UnitaryGate(U))

"""
# In[82]:


# Un code pour tester le circuit calculant (x * a ^ p) mod N, sur n qubits

# VALEURS À CHANGER LE CAS ÉCHÉANT.
# ATTENTION à préserver x, N < 2^n
x = 5
a = 2
p = 4
N = 11
n = 4

# TEST À NE PAS MODIFIER

phi = QuantumRegister(n)
cphi = ClassicalRegister(n)

qc = QuantumCircuit(eig,phi,cphi)
vl = nat2bl(n,x)
print(f"Input : {str(vl)} (= {x} en décimal)")

vl.reverse() # Changement de la place du bit de poid faible
for i in range(len(vl)):
    if vl[i] == 1:
        qc.x(phi[i])

qc.append(gateMult(a,p,N,n),list(phi))
qc.measure(phi,cphi)

backend = BasicAer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
d = job.result().get_counts(qc)

assert(len(d) == 1)
s =  list(d.keys())[0]

if x < N:
    print(f"La vraie réponse devrait être {x} * {a}^{p} mod {N} = {(x*(a ** p)) % N}")
else:
    print(f"On est au delà de {N} donc on devrait être l'identité")
print(f"Le circuit répond : {s} (= {bs2nat(s)} en décimal)")
"""
# Si vous exécutez, cela devrait afficher un log de ce qui est attendu et de ce qui est retourné.


# ### Q 3.2 Taille du circuit généré
# 
# Qiskit package l'opérateur dans une jolie boite. Mais quelle est la taille réelle du circuit qui a été produit ? Nous allons essayé pour voir.
# 
# - À faire : modifier le code ci-dessous pour calculer la taille des circuits générés par QisKit pour les matrices:
# (vous pouvez écrire à coté de chaque bullet point la réponse que vous trouvez).
#   * `gateMult(3,3,2 ** 2,2)`5 
#   * `gateMult(3,3,2 ** 3,3)`131
#   * `gateMult(3,3,2 ** 4,4)`712 
#   * `gateMult(3,3,2 ** 5,5)`2592 
#   * `gateMult(3,3,2 ** 6,6)`10667 
#   * `gateMult(3,3,2 ** 7,7)`43691 -> exponentielle de raison ~ 4 !

# In[105]:
"""

# VALEURS À CHANGER LE CAS ÉCHÉANT.
# ATTENTION à préserver x, N < 2^n
x = 2
a = 2
p = 4
N = 3
n = 2
#qc.h(q[0])
#qc.x(q[1])
#qc.ccx(q[0],q[1],q[2]) # une porte Toffoli, histoire de dire.
#for i in range(2,8):
#    q = QuantumRegister(i)
#    qc = QuantumCircuit(q)
#    qc.append(gateMult(3,3,2**i,i))


phi = QuantumRegister(n)
cphi = ClassicalRegister(n)

qc = QuantumCircuit(eig,phi,cphi)
vl = nat2bl(n,x)
print(f"Input : {str(vl)} (= {x} en décimal)")

vl.reverse() # Changement de la place du bit de poid faible
for i in range(len(vl)):
    if vl[i] == 1:
        qc.x(phi[i])

qc.append(gateMult(a,p,N,n),list(phi))
qc.measure(phi,cphi)
# Notre circuit n'est pas nécessairement en portes unitaires élémentaires.
# Décomposons-le au cas où avant de compter.

pass_ = Unroller(['u1', 'u2', 'u3', 'cx'])
pm = PassManager(pass_)
new_circ = pm.run(qc)

count = new_circ.count_ops()
count


# In[106]:


r = 0
for k in count:
    r += count[k]
print(f"Pour {n} qubits il y a {r} portes en tout")
"""
print("PARTIE 4 ")
# ### Q 3.3 Analyse
# 
# - Quelle est la complexité de la taille du circuit en fonction du nombre de qubits ?
# elle est exponentielle !
# 
# - Si c'est jouable pour de petites tailles, est-ce une méthode réaliste pour quelque chose de non-trivial ?
# 
# NON §!!
# 

# # 4 - Réunissons les morceaux : Shor
# 
# Nous sommes maintenant prêt à réaliser l'algorithme de Shor 
# 
# ## Questions
# 
# ### Q 4.1 Le circuit
# 
# Réalisez le circuit:
# - Reprenez le code de QPE : on va prendre 4 bits de précisions pour `eig`
# - Introduisez notre unitaire "maison" : on va choisir 5 bits pour `phi`
# - Mettez `phi` dans l'état correspondant à l'entier $1$ en décimal (attention à la place du bit de poid faible)
# 

# In[125]:


a = 27
#p = 3
#x = 3

N = 31

def findorder(a,N,prec=5,Nbmax=5):
    """ Nmax= 2**Nbmax """
    size_eig = prec
    size_phi = Nbmax
    
    eig = QuantumRegister(size_eig)
    phi = QuantumRegister(size_phi)
    ceig = ClassicalRegister(size_eig)
    #cphi = ClassicalRegister(size_phi)
    qc = QuantumCircuit(eig,phi,ceig)#,cphi)
    
    qc.x(phi[0]) # dépend peu de la valeur de l'indice ici !
    #qc.h(phi[0]) superposition
    
    
    for i in range(size_eig):
        qc.h(eig[i])
     
    contU = gateMult(a,1,N,size_phi).control()
    
    for j in range(size_eig):
        qc.append(contU.power(2**(j)),[list(eig)[size_eig-1-j]]+list(phi))
    qc.append(QFT(size_eig).inverse(),list(eig))
    
    
    for i in range(size_eig):
        qc.measure(eig[i], ceig[i])
    
    #for i in range(size_phi):
    #    qc.measure(phi[i], cphi[i])
    
    # À la fin, on dessine le circuit pour vérifier visuellement que c'est en ordre
    
    qc.draw()
    
    
    # In[128]:
    
    
    backend = BasicAer.get_backend('qasm_simulator')
    job = execute(qc, backend, shots=1000)
    d = job.result().get_counts(qc)
    print(d)
    return d

#d = findorder(a,N)
#print(d)


# ### Q 4.2 Les résultats
# 
# On récupère une distribution de résultats sous la forme d'un dictionnaire.
#  - chaque clé correspond moralement à la fraction $s/r$, avec $r$ l'ordre de $a$ mod $N$
#  - évidemment, c'est écrit avec une certaine précision (qui dépend du nombre de bits), et c'est écrit en binaire
#  - chaque valeur est le nombre de fois où on a trouvé cette clé.
#  
# À faire: un graphique avec en abscisse la clé, en décimal, et en ordonnée le nombre de fois où on l'a trouvé.
# 
# Par exemple, ci-dessous ce qui est attendu.
##{{}}
# In[9]:
"""d0 = {'00 1101': 110, '00 1010': 112, '00 0001': 182, '00 0110': 38, '00 0000': 190, '00 0101': 32, '00 1011': 121, '00 1100': 104, '00 0111': 29, '00 1110': 7, '00 0100': 29, '00 0011': 14, '00 1111': 6, '00 1000': 6, '00 0010': 15, '00 1001': 5}
d1 = {'00 0000': 179, '00 0001': 176, '00 1100': 114, '00 1101': 102, '00 0110': 31, '00 0010': 21, '00 0111': 34, '00 1010': 110, '00 0100': 36, '00 0011': 16, '00 1111': 8, '00 1011': 119, '00 0101': 37, '00 1001': 6, '00 1110': 6, '00 1000': 5}
#plot(d)
#show()
# Supposons que d est comme suit
d2 = {"100": 12, "101": 20, "011" : 45, "111" : 28}
d3 = {"00 100": 12, "00 101": 20, "00 011" : 45, "00 111" : 28}"""
# on veut réaliser:
#plot([3,4,5,7], [45,12,20,28])
#show()

# In[ ]:
##
import operator
def showdic(d):
    """ show dict """
    x = []
    y = []
    #for key in sorted(d.items(), key=lambda x : bs2nat(operator.itemgetter(0)(x)[3:][::-1])) :
    for key in sorted(d.items(), key=lambda x : bs2nat(operator.itemgetter(0)(x)[::-1])) :
        key,val = key
        key=key#[3:]
        n = bs2nat(key[::-1])
        x.append(n)
        y.append(val)
    plot(x,y)
    show()
#showdic(d)

#11_25 -> 5
#2 21 -> 6
#9 26 -> 3
#7 15 -> exact ! -> b est divisible par 8 (donc par 4) (c'est "plat" si on n'augmente pas la précision)
#8 31 -> 5
#27 31 -> 10 (en theorie -> il faut passer à 5 bits de précision (on ne voit pas tous les pics avec seulement 4 bits ! (il faudrait afficher 10 hauts et 10 bas : 20 bits minimum 2**4=16<20<2**5=32))

# ### Q 4.3 Analyse
# 
# - Le dessin n'est pas très précis. Comment l'améliorer ? (AUGMENTER LE NOMBRE DE BITS) Essayez.
# - Pouvez-vous inférer des résultats la valeur de $r$ ?(OUI) Où la voyez-vous sur le graphique ? (> nombre de pics)
# - Est-ce que cela marche toujours si vous modifiez la valeur de `a` et/ou de `N` ? Attention à ne pas prendre de trop grandes valeurs pour `N`...
# (oui. (dans la mesure où r n'a pas une représentation en binaire finie de taille < nb bits)

# # 5 - Pour aller plus loin
# 
# Vous aurez noté que le circuit est très lourd, malgré le fait qu'on a que quelques qubits. Si on essayait de prendre le modulo d'un nombre pluss grand, il faudrait augmenter `size_phi` ce qui rendrait les choses impraticables.
# 
# Comme ce dont nous avons discuté en cours, pour réaliser l'orable on peut utiliser le fait que la fonction n'est pas quelconque, mais possède une structure bien particulière. On peut donc la "programmer" avec des circuits. C'est ce qui est proposé par exemple ici : [https://arxiv.org/abs/quant-ph/0205095].

# In[ ]:


from math import sqrt,isclose,gcd
from itertools import count, islice

def is_prime(n):
    return n > 1 and all(n % i for i in islice(count(2), int(sqrt(n)-1)))

from random import randint 
# In[ ]:

def log2(n):
    m=0
    while 2**m < n:
        m += 1
    return int(m - 1)

def Shor(M):
    """ shor algorithm """
    
    prec=log2(M)+1
    Nbmax=log2(M)+1
    
    na = randint(2,M-1)
    fc = M/na
    class Rimpair(BaseException):
        pass
    try:
        if isclose(int(fc),fc):
            return na
        else:# na and N are prime
            print("Comptez bien les pics !")
            d = findorder(na,M,prec=prec,Nbmax=Nbmax)
            showdic(d)
            nbpic = input("Combien de pics ?")# (ça peut se faire automatiquement sans problème)
            print("Réponse:",nbpic,type(nbpic))
            r = int(nbpic)
            if r%2:# r impair
                print("r:",r,"est impair")
                raise Rimpair
            else:#r pair
                rprime = r//2
                return gcd(M,na**rprime-1)
    except Rimpair:
        return Shor(M)
    except qiskit.extensions.exceptions.ExtensionError:
        return Shor(M)



# In[ ]:
from sys import argv
try:
	num = int(argv[1])
except:
	num = 28
print("Launching Shor with value",num)
print("SHOR's response : ",Shor(num))


