
## STRINGS

s = "hello"
print(s.capitalize())  # Capitalize a string
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces
print(s.center(7))     # Center a string, padding with spaces
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another
print('  world '.strip())  # Strip leading and trailing whitespace

# Dictionnary Comprehensions 

nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)

## Sets

animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))

for z in enumerate(even_num_to_square): 
    print(z) # print tuples key,val
    
## powerful for loops

print("reversed")
for ch in reversed("abc"):
    print(ch)
print("\nenumerated:\t")
for i,ch in enumerate("abc"):
    print(i,"=",ch,end="; ")
print("\nzip'ed: ")
for a,x in zip("abc","xyz"):
    print(a,":",x)

## else statements loops
count = 0
while count < 10:
    count += 1
    if count % 2 == 0: # even number
        count += 2
        continue
    elif 5 < count < 9:
        break # abnormal exit if we get here!
    print("count =",count)
else: # while-else
    print("Normal exit with",count)

## args = arguments in iterable
## kwargs = arguments in a dict (keyword arguments)

## bool idx
import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = ((a > 2))  # Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is % 2.

print(bool_idx)

## BROADCASTING

import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)

"""
Broadcasting two arrays together follows these rules:

   1 If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
   2 The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
   3 The arrays can be broadcast together if they are compatible in all dimensions.
   4 After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
   5 In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension
"""

## pyplotons !
import matplotlib.pyplot as plt

#  incantation : %matplotlib inline
x = np.arange(-5,5,0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.plot(x, y_cos)
# Set up a subplot grid that has height 2 and width 1,
# and set the second such subplot as active.
plt.subplot(2, 1, 2)
plt.plot(x,-y_sin/2)
plt.plot(x,-y_cos/2)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'],loc='best')
#plt.show()

## ANIMAÇ~AO

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, rc
from IPython.display import HTML

# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()
plt.close()


ax.set_xlim(( 0, 2))
ax.set_ylim((-2, 2))

line, = ax.plot([], [], lw=2)

# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return (line,)

# animation function. This is called sequentially  
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return (line,)
  

anim = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=100, interval=100, blit=True)

# Note: below is the part which makes it work on Colab
rc('animation', html='jshtml')
anim
# does not work on emacs ??
## GRAPH

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_node(1)
G.add_nodes_from([3,6])
G.add_edge(1,3)
G.add_edge(1,6)
G.add_edge(3,6)
nx.draw(G)
