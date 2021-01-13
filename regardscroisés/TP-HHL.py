#!/usr/bin/env python
# coding: utf-8

# # L'algorithme HHL de résolution de systèmes linéaires

# ## Préliminaires
# 
# ### Des librairies

# In[34]:


import numpy as np
from scipy import linalg
from math import pi
from qiskit import *
from qiskit.circuit import *
from qiskit.extensions import *
from qiskit.circuit.library import *
from qiskit.providers.aer.extensions import *
from qiskit.extensions.simulator.snapshot import snapshot
from qiskit.quantum_info.operators import Operator
from qiskit.extensions.simulator.snapshot import snapshot
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller
from scipy import optimize
from matplotlib.pyplot import plot,show
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg' # Makes the images look nice")


# ### Des fonctions de conversion que nous avons déja vu

# In[200]:


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

# ## Proposition pour débugguer un circuit
# 
# Avec le backend `statevector`, on peut faire des *snapshots* de l'état de la mémoire en cours de calcul. Après l'évaluation du circuit, on peut récupérer ces résultats intermédiaires pour vérifier que tout va bien.
# 
# Par exemple, on peut fabriquer le circuit suivant.

# In[184]:


q1 = QuantumRegister(3, name="q1")
q2 = QuantumRegister(1, name="q2")
qc = QuantumCircuit(q1,q2)

qc.x(q1[0])
qc.x(q1[1])

# Un premier snapshot, avec un nom
qc.snapshot_statevector('afterX')

qc.h(q2[0])

# Un deuxième snapshot, avec un autre nom
qc.snapshot_statevector('afterH')

qc.cx(q2[0],q1[2])

# Affichage du circuit, pour faire bonne mesure.
# Notez comment les snapshot sont indiqués sur le circuit...
qc.draw()


# In[185]:


# Maintenant, hop, on calcule !

# Le backend statevector
simulator = Aer.get_backend('statevector_simulator')

# Execution !
result = execute(qc, simulator).result().data(0)

# Il s'agit d'un dictionnaire, avec les snapshots, et le statevector final.
result


# Évidemment, techniquement on a l'info. Mais dire que c'est lisible est difficile... Pour les besoins du TP, je vous ai fait une petite librairie qui affiche ça de façon pas trop illisible.
# 
# #### Des fonctions d'impression de vecteur d'état

# In[186]:


def processStates(state): # List of states
    def aux(st): # Longueur = puissance de 2
        s = list(st)
        if len(s) == 2:
            return {'0' : s[0], '1' : s[1]}
        else:
            a0 = aux(s[:len(s)//2])
            a1 = aux(s[len(s)//2:])
            r = {}
            for k in a0:
                r['0' + k] = a0[k]
            for k in a1:
                r['1' + k] = a1[k]
            return r
    r = []
    for i in range(len(state)):
        r.append(aux(state[i]))
    return r

def printOneState(d): # get a dict as per processStates output
    for k in d:
        im = d[k].imag
        re = d[k].real
        if abs(im) >= 0.001 or abs(re) >= 0.001:
            print("% .3f + % .3fj |%s>" % (re,im,k))

def getDictSnapshot(result,name):
    return processStates(result['snapshots']['statevector'][name])[0]

def getDictFinalRes(result):
    return processStates([result['statevector']])[0]

def printSnapshot(result,name):
    printOneState(processStates(result['snapshots']['statevector'][name])[0])

def printFinalRes(result):
    printOneState(processStates([result['statevector']])[0])


# #### Example d'utilisation
# 
# On peut s'en servir pour afficher les infos de `result`, comme suit.

# In[187]:


getDictSnapshot(result,'afterX')


# In[188]:


printSnapshot(result,'afterX')


# In[189]:


printSnapshot(result,'afterH')


# In[190]:


getDictFinalRes(result)


# In[191]:


printFinalRes(result)


# Notez comment `q2` qu'on a mis à la fin dans `QuantumRegister` apparait au début du vecteur de kets. Par rapport au dessin du circuit, il faut "tourner la tête à gauche". J'ai essayé d'utiliser les même conventions que pour les résultats de mesure.
# 
# #### Question
# 
# Est-ce que ce qui précède vous semble compréhensible ? OUI
# 
# Si vous faites les calculs à la main, est-ce que vous retombez bien sur les mêmes valeurs pour ce que disent les snapshots ? OUI
# 
# Si non, signalez-le ! ....

# 
# 
# ## L'algorithme HHL
# 
# Une présentation très lisible se trouve sur [le site de QisKit](https://qiskit.org/textbook/ch-applications/hhl_tutorial.html). Nous allons reprendre brièvement la présentation qui y est faite.
# 
# L'algorithme HHL est nommé d'après les auteurs éponymes du papier
# 
# * A. W. Harrow, A. Hassidim, and S. Lloyd, “Quantum algorithm for linear systems of equations,” Phys. Rev. Lett. 103.15 (2009), p. 150502. [arXiv:0811.3171](https://arxiv.org/abs/0811.3171)
# 
# Cet algorithme utilise une mémoire quantique pour résoudre un système d'équations linéaire de manière plus efficace que ce que l'on peut faire en calcul classique. Donc de pouvoir donner un vecteur $\vec{x}$ tel que
# 
# $$A\cdot \vec{x} = \vec{b}$$
# 
# pour une matrice $A$ et un vecteur $\vec{b}$. L'idée est de coder les vecteurs dans les coefficients de l'état des registres. Donc si $\vec{b} = (b_0,b_1,b_2,b_3)$, on va stocker le vecteur dans $|b\rangle = b_0\cdot|00\rangle+b_1\cdot|01\rangle+b_2\cdot|10\rangle+b_3\cdot|11\rangle$, modulo une renormalisation.
# 
# Évidemment, il y a des petites lignes dans le contrat. Mais en résumé, si
# * $N$ est la taille du système,
# * $s$ est le nombre d'éléments non-nuls par ligne
# * $κ$ est le conditionnement de la matrice (le ratio entre la plus grande valeur propre et la plus petite)
# * $ϵ$ est l'erreur autorisée
# 
# alors on peut donner les complexités suivantes:
# 
# * $\mathcal{O}(Nsκ\log(1/ϵ))$ dans le cas classique
# 
# * $\mathcal{O}(\log(N)s^2κ^2/ϵ)$ pour HHL
# 
# Donc, bref, on a un gain exponentiel par rapport à la taille de la matrice, en théorie.
# 
# #### À faire
# 
# Je vous invite à aller voir les sections 1 et 2 du [tutoriel de QisKit](https://qiskit.org/textbook/ch-applications/hhl_tutorial.html) qui expliquent très bien l'algorithme.
# 

# ### Le cas d'étude
# 
# Nous allons reprendre le cas proposé par le [tutoriel de QisKit](https://qiskit.org/textbook/ch-applications/hhl_tutorial.html): le système linéaire
# 
# $$\begin{align}
# x_1-\frac{x_2}3 &= 1\\
# \frac{-x_1}3 + x_2 &= 0
# \end{align}
# $$
# 
# #### Question
# 
# Quelle est la solution de ce système ? x1 = 9/8 x2 = 3/8  
# 
# Quel est le ratio $\frac{x_1}{x_2}$ ? 3
# 
# Quels sont les valeurs propres de la matrice $A$ correspondante ? 2/3 et 4/3
# 
# Ses vecteurs propres ? 2/3 -> (1  1) 4/3 -> (1, -1)
# 
# ### Le codage dans la mémoire
# 
# Nous aurons besoin d'un seul qubit pour stocker le vecteur $\vec{b}$ et le vecteur $\vec{x}$:
# 
# $$\begin{align}
# \vec{b} &\quad\equiv\quad 1\cdot|0\rangle + 0\cdot|1\rangle\\
# \vec{x} &\quad\equiv\quad (D\,x_1)\cdot|0\rangle + (D\,x_2)\cdot|1\rangle
# \end{align}$$
# 
# où $D$ est un facteur de renormalisation. Notez comment le ratio entre le coefficient de $|0\rangle$ et celui de $|1\rangle$ reste le même pour $\vec{x}$
# 
# 

# ### Les besoins de l'algorithme
# 
# Vous avec lu les sections 1 et 2 du [tutoriel de QisKit](https://qiskit.org/textbook/ch-applications/hhl_tutorial.html) ? Nous allons mettre en oeuvre cet algorithme, en jouant avec les différents paramètres pour tacher de comprendre de manière concrète comment cela fonctionne.
# 
# Pour cet algorithme, nous avons besoin de différents morceaux:
# 
# 1. L'initialisation du vecteur d'entrée $\vec{b}$ : pour l'exemple considéré, c'est trivial (pourquoi ?) un vecteur de base
# 
# 2. QPE : nous avons réalisé ce circuit pour Shor -- vous êtes donc parés.
# 
# 3. L'unitaire à utiliser pour QPE : ce sera donc $U = e^{iAt}$, pour un $t$ "correct" (dépendant de $\kappa$ -- nous jouerons avec pour voir).
# 
# 4. Un opérateur d'inversion
# 
# La difficulté majeure repose donc sur les deux derniers points. Nous allons donc nous y attarder un tout petit peu.
# 
# 

# ## Codons HHL
# 
# Pour nous assurer que tout fonctionne bien, nous allons utiliser les valeurs et données proposées par la section 3 du [tutoriel de QisKit](https://qiskit.org/textbook/ch-applications/hhl_tutorial.html). De la sorte, les valeurs tombent juste, et on peut calculer les choses à la main et vérifier que le circuit se comporte bien (par exemple en utilisant des snapshots).
# 
# Le circuit à réaliser travaille donc sur 4 qubits, et réalise QPE avec 2 bits de précisions. On souhaite arriver au circuit
# ![a.svg](attachment:a.svg)
# 
# * Les deux premiers `unitaire` sont $U = e^{iAt}$
# * Les deux derniers `unitaire` sont $U^{-1}$
# * `Iqft` est la QFT inverse
# * `inv` est le circuit qui réalise l'inverse
# * `Qft` est la QFT
# 

# ### La matrice A et l'opérateur U
# 
# Pour coder notre algorithme, nous avons besoin de pouvoir implémenter l'exponentielle $U = e^{iAt}$ d'une matrice. En Python, cela peut de faire simplement à l'aide de la fonction `expm` de la bibliothèque `linalg`.
# 
# On peut donc simplement faire comme suit:

# In[329]:


# La matrice A avec laquelle on va travailler

A = np.array([[1,-1/3],[-1/3,1]])
A


# In[330]:


# Le coefficient qui va faire tomber rond les valeurs

t = 2*pi*10/8

# L'exposant de notre matrice 

EA = linalg.expm((1j)*t*A)
EA


# In[331]:


# Et finalement on peut réaliser un unitaire sur 1 qubit avec

U = UnitaryGate(Operator(EA))
U


# ### L'opérateur d'inversion
# 
# On veut inverser une valeur $\lambda\gt 0$ codée sur 2 bits, par exemple comme suit:
# Posons $\lambda = \frac{e_0}{2} + \frac{e_1}{4}$ (avec $e_0$ et $e_1$ des booléens).
# 
# Notez que $\frac13\leq \frac1{4\lambda} \leq 1$
# 
# On veut un opérateur qui génère: 
# $|e_1e_0\rangle\otimes|0\rangle\mapsto |e_1e_0\rangle\otimes\left(\sqrt{1 - \frac1{16\lambda^2}}|0\rangle + \frac1{4\lambda}|{1}\rangle\right)$. Ici, la valeur $\frac14$ est le coefficient $C$ dans le tutoriel QisKit.
# 
# #### Question
# 
# Réalisez cette opération à l'aide de portes $R_y(\theta)$ multi-controllées
# * La fonction `np.arcsin` va être votre amie
# * Vous pouvez fabriquer une porte Ry avec `RYGate(angle)`
# * une porte peut être controllée avec `porte.control(num_ctrl_qubits=3, ctrl_state='101')` si vous voulez trois controls, 1 négatif, 1 positif, 1 négatif.
# * On insère une porte avec `moncircuit.append(maporte, list-of-wires)`.

# In[337]:


# For the value to invert
n = 5
N = round(2 ** n)

def create_inv(n,draw=False):
	N = round(2 ** n)
	q = QuantumRegister(n, name="q")

	# To store the invert
	r = QuantumRegister(1, name="r")

	qcinv = QuantumCircuit(q,r)

	# On fixe C à 1/4 pour 2 qubits
	C = 1 / (N * 2)

	for i in range(1,N):
		th = 2*np.arcsin(C*N/i)
		if draw : print(th)
		R = RYGate(th)
		if draw: print(R)
		rc = R.control(num_ctrl_qubits=n,ctrl_state=nat2bs(n,i))
		if draw: print(rc)
		qcinv.append(rc,list(q)+list(r))
	if draw:
		qcinv.draw()
	return qcinv


# On va maintenant stocker ce circuit comme une (grosse) porte unitaire `invCirc` agissant sur 3 fils.

# In[338]:


invCirc = create_inv(n).to_gate(label="inv")


# #### Questions
# 
# Pour comprendre comment utiliser cette porte (et s'assurer qu'elle fonctionne), voici un petit code. Essayez-le en modifiant la valeur de `v` : 
# 
# * La porte `invCirc` s'utilise avec par exemple `qc.append(invCirc, list-of-wires)`
# * Où lit-on la valeur $1/v$ ? On la lit dans la valeur du ket |nu>
# * Au niveau des ket, où se trouvent chacun des deux registres ? |???> 
# * Est-ce que vous obtenez le comportement attendu ? oui... ?
# 
# 

# In[339]:


# The value to invert : test 1, 2 and 3 and make sure to understand what's going on
def test_inv(v,n=n,draw=True):
	v = v

	# For the value to invert
	q = QuantumRegister(n, name="q")

	# To store the value
	r = QuantumRegister(1, name="r")

	qctest = QuantumCircuit(q,r)
	bi = list(reversed(nat2bl(n,v)))       # Pour l'instant, n = 2
	for i in range(len(bi)):
		if bi[i] == 1:
		    #print(bi[i],bi,i,qctest)
		    #print(q[i])
		    qctest.x(q[i])

	qctest.append(invCirc, list(q) + list(r))# adds the former circuit, as a gate, to qctest
		    
	simulator = Aer.get_backend('statevector_simulator')

	# Execute and get result
	result = execute(qctest, simulator).result().data(0)
	if draw:printFinalRes(result)
	return result


# ### Nous sommes maintenant prêt à coder l'algorithme au complet
# 
# Ci-dessous, vous trouverez un code à trous pour réaliser HHL avec 2 bits de précision, en utilisant les morceaux du dessus (sans compter QPE que vous avez déjà réalisé une fois).
# 
# #### À faire
# 
# L'objectif est de réaliser le circuit comme ci-dessous. Placez aux endroits indiqués en rouge deux demande de snapshots.
# ![a2.svg](attachment:a2.svg)
# 
# 

# In[340]:



def create_U(A,t=None):
    if t is None: t = 1/8
    t = 2*pi*t
    EA = linalg.expm((1j)*t*A)
    return UnitaryGate(Operator(EA))
    
def hhl(A,t=None,n=n):
    U = create_U(A,t=t)
    # For |b>
    nb = 1
    qb = phi = QuantumRegister(nb, name="b")

    # For the eigenvalue : 2 bits of précision
    ne = size_eig = n
    qe = eig = QuantumRegister(ne, name="eig")

    # For the angle : only one qubit
    qi = inv = QuantumRegister(1, name="inv")

    # Creating the circuit
    qc = QuantumCircuit(qi,qe,qb)

    ## QPE on b and eig

    for i in range(size_eig):
        qc.h(eig[i])

    contU = U.control()

    for j in range(size_eig-1,-1,-1):
        qc.append(contU.power(2**(j)),[list(eig)[size_eig-1-j]]+list(phi))

    qc.append(QFT(size_eig).inverse(),list(eig))

    ## The first snapshot
    qc.snapshot_statevector('qpe')

    ## invCirc on eig and inv

    qc.append(create_inv(n).to_gate(label="inv"), list(eig)[::-1] + list(inv))

    ## The second snapshot
    qc.snapshot_statevector('inv')

    ## QPE, inverted

    qc.append(QFT(size_eig),list(eig))
    for j in range(size_eig):
        qc.append(contU.power(-2**(j)),[list(eig)[size_eig-1-j]]+list(phi))


    for i in range(size_eig-1,-1,-1):
        qc.h(eig[i])


    # Finally, one draw the nice circuit
    qc.draw()
    # On lance un simulateur
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(qc, simulator).result().data(0)
    r = getDictFinalRes(result)
    r0,r1 = getres(r)

    c = abs(r0)/abs(r1)
    print(r0,r1,c)
    return c

def getres(r,n):
	s = "1"
	for _ in range(n):
		s = "0" + s 
	s1 = "1" + s
	s0 = "0" + s
    r0 = r[s0]#proba ket inv = 0# coefficient corresponding to x0 ????
    r1 = r[s1]#proba ket inv = 1 coefficient corresponding to x1 ???? 
    return r0,r1

# #### À faire
# 
# Maintenant, il faut tester votre circuit ! Ci-dessous un code qui exécute le circuit puis qui imprime les résultats intermédiaires et finaux. Vérifiez que les differents résultats sont ceux attendus
# * Au fait, quelles sont sensées être les résultats intermédiaires ?
# * On a pris soin d'avoir des résultats exacts ! On ne doit donc pas avoir un résultat approché
#  * Ceci dit, $0.9999999999$ est bien la même chose que $1$...

# In[341]:


# On lance un simulateur
def simulate(n):
	simulator = Aer.get_backend('statevector_simulator')
	result = execute(hhl(A,3/8,n=n), simulator).result().data(0)
	printSnapshot(result,'qpe')
	printSnapshot(result,'inv')
	printFinalRes(result)
	r = getDictFinalRes(result)

	#r0 = r["0001"]#proba ket inv = 0# coefficient corresponding to x0 ???? pour n = 2
	#r1 = r['1001']#proba ket inv = 1 coefficient corresponding to x1 ???? 
	r0 = r["0000001"]#proba ket inv = 0# coefficient corresponding to x0 ????
	r1 = r['1000001']#proba ket inv = 1 coefficient corresponding to x1 ???? 

	r0, r1, abs(r0)/abs(r1)


# #### Questions
# 
# * Où lit-ont $x_1$ et $x_2$ (modulo renormalisation) dans l'état final ?
# * Récupérez les deux coefficients
#  * `getDictSnapshot` et `getDictFinalRes` définis au dessus sont vos amis
# * Est-ce que le ratio est bien le bon ?
#  * Sinon, discutons-en !



# ## Jouons avec les valeurs
# 
# Pour l'instant, on a simplement 2 bits de précision: c'est un peu faible pour imaginer pouvoir "résoudre" un autre système. Dans cette partie, nous allons "jouer" avec les différents paramètres, et augmenter la précision du calcul.
# 
# ### Influence de C et t
# 
# Deux paramètres ont été choisis de manière apparemment arbitraire: C et t. Sans changer la précision, on peut ceci dit déjà regarder comment ces deux valeurs influent sur le calcul
# 
# #### Questions
# 
# * Sur le papier, qu'est-ce que cela changerait de faire passer la constante `C` de 1/4 à 1/8 ?  (rien)
#  * Indice : quel est le rapport avec les coefficients finaux ?
# * Essayez dans le code, et vérifiez ce qui se passe pour les coefficients stockant $x_0$ et $x_1$. (il leur arrive des trucs)
# * Remettez la valeur de `C` à 1/4. Non.
# * Maintenant, divisez `t` (dans la définition de $U$) par 2.
# * Que doit-on observer ? (???) Qu'observez-vous ? (ça marche, mais moins bien)
#  * Indice : Pensez *précision*
# * Remettez la valeur de `t` à sa valeur d'origine.
# * Changez le $1/3$ dans la matrice $A$ par $1/5$, et essayez de résoudre. Est-ce probant ? Pourquoi ? non.  Sûrement à cause de la précision. le QPE will output an nl-bit (2-bit in this case) binary approximation to λjt/2π. Ainsi, si t = 2pi * 1/lamb, ok ! malheureusement, lambda t / 2 pi * t n'est pas nesé avoir une decomp finie en base 2 donc ce n'est pas précis
# 
#  * Indice : Pensez *précision*
# * Remettez à $1/3$.
# * Changez le $3$ dans la définition de $t$ en $5$, et essayez de résoudre. Est-ce probant ? Pourquoi ? oui.car t n'influe pas plus que ça .
#  * Indice : Pensez *précision*
# * Remettez comme au départ
# 
# 
# ### Plus de précision !
# 
# Dans cette deuxième partie, je vous propose de rajouter des bits de précisions pour le registre `eig` (c-a-d `qe`): on veut passer de 2 à un nombre arbitraire. Cela va permettre
# * de pouvoir traiter des systèmes avec une autre fraction que $1/3$
# * de constater l'invariance du résultat suite à des modifications de t
# 
# #### À faire.
# 
# * Rendez votre code paramétrique en `ne`, la taille du registre `eig` (c-a-d `qe`).
#  * Note : cela nécessite de faire un code paramétrique. Si cela vous semble insurmontable, vous pouvez évidemment faire une version statique. Dans ce cas, choisissez un nombre assez grand, comme `ne = 5` par exemple. Plus le nombre choisi est grand, plus votre code sera résiliant aux erreurs.
#  * Attention : il faut aussi modifier le circuit qui réalise l'inversion
#    * Il vous faudra du coup changer $C$... car on veut que $C/\lambda$ soit toujours entre 0 et 1
#  * Attention : dans la QPE, les exposant de $U$ sont des puissances de 2. Donc : 1, 2, 4, 8, 16, 32... et non pas 1, 2, 3, 4, 5, ...
#  * Vous pouvez trouver utiles mes petites fonctions nat2bl, etc, définies au début de ce notebook
# * Sans changer les valeurs de t, C, etc, vérifiez que vous trouvez toujours la même réponse, avec la même exactitude. C'est un bon test que les choses sont faites correctement.
# * Maintenant, choisissez un t différent: par exemple, $2\cdot\pi\cdot5/16$. Est-ce que cela change quelque chose à la qualité ?
# * Changez $C$ en quelque chose de plus petit : vérifiez que les coefficient de la solution sont bien changés, mais que le ratio reste le même
# * Changez maintenant le $1/3$ de la matrice $A$ : est-ce que vous pouvez bien résoudre le système avec d'autre valeurs: $1/5$? $1/2$? $1/4$? $1/10$? OUI OUI OUI OUI OUI
# 
# 
# ## Discussion
# 
# Dans cette implémentation, nous avons techniquement un circuit de taille exponentielle sur la précision (la taille du registre `eig`). En effet, notre circuit de calcul d'inverse est idiot, et nous utilisons pour générer le circuit $U$ la librairie standard de QisKit, qui ne capitalise pas sur la structure de notre matrice.
# 
# Il y aurait évidemment moyen de faire mieux:
# 
# 1. Pour le circuit d'inversion, il s'agit d'un oracle ! On a donc des méthodes à priori moins couteuses (avec éventuellement ceci dit l'utilisation de fils auxiliaires)
# 
# 2. Pour la réalisation des circuits pour $(e^{iAt})^{2^n}$, comme $A$ est un Hermitien, il y a une technique standard: la trotterisation. 
#  * Voir par exemple les sections 2.1 et 2.2 de [arXiv:1904.01336](https://arxiv.org/abs/1904.01336v1) qui résume très bien la méthode
#  * Il faut ensuite simplement réaliser que la matrice $A$ peut s'écrire comme une combinaison linéaire à coefficient réels de matrices de Pauli sous la forme $C_P = Z\otimes X\otimes X\otimes\cdots$. Et on peut donner un circuit simple pour les unitaires sous la forme $e^{itC_P}$.

M = np.array([[1,-1],[-1,1]])
z = hhl(A,3/8)
print(z)
np.linalg.eigvals(A)

TAU = np.array([[1,-1/4],[-1/4,1]])
z = hhl(TAU,2/16)
print(z)
np.linalg.eigvals(TAU)


