
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

