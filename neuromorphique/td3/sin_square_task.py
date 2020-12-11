import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from Reservoir import reservoir
import time

STARTTIME = time.time()

#def plotacc():


res_sizes=np.arange(0.1,5,0.002)
list_accuracies = []
for j,dens in enumerate(res_sizes):
    # PREPARATION DU DATASET
    dens /= 100
    # Prepare a synthetic dataset
    nb_exemples=20
    data=np.random.randint(2, size=nb_exemples)
    square=[1,1,1,1,-1,-1,-1,-1]
    sin=[0.7,1,0.7,0,-0.7,-1,-0.7,0]

    dataset=[]
    target=[]
    for i in data:
        if i==0:
            dataset=np.concatenate((dataset, square), axis=0)
            target=np.concatenate((target,np.zeros(8)))
        if i==1:
            dataset=np.concatenate((dataset, sin), axis=0)
            target = np.concatenate((target, np.ones(8)))

    # Visualise dataset

    #on a ploté un sin ou un carré
    #plt.plot(dataset)
    #la target indique si on a sin ou carré
    #plt.plot(target)
    #plt.show()

    #on transforme l'array en tenseur
    dataset=torch.from_numpy(dataset)
    dataset=dataset.to(torch.float32)
    target=torch.from_numpy(target)
    target=target.to(torch.float32)



    # Split training set and test set
    # on prend la moité pour le training
    training_x = dataset[0:np.int(8*nb_exemples/2)].view(-1,1)
    training_y = target[0:np.int(8*nb_exemples/2)]
    test_x = dataset[np.int(8*nb_exemples/2):-1].view(-1,1)
    test_y = target[np.int(8*nb_exemples/2):-1]

    #  # # CREATION DU MODELE

    model = reservoir(input_size=1, reservoir_size = 35, contractivity_coeff=0.8, density=dens)

    # Collect states for training set

    X = model(training_x)
    Y = training_y

    # Train the model by Moore-Penrose pseudoinversion.
    try:
        W = X.pinverse() @ Y

        # Evaluate the model on the test set
        # We pass the latest training state in order to avoid the need for another washout
        X_test = model(test_x)
        predicted_test = X_test @ W

        # Compute and print the accuracy

        #♦we set  the threshold
        correct = ((predicted_test>0.5) == test_y).float().sum()
        accuracy = correct / test_y.shape[0]
        print('i = ',j,'accuracy =', accuracy)

        list_accuracies.append((j,accuracy))
    except:
        # matrix inversion did not work
        pass

##
plt.figure()
for i,y in list_accuracies:
    if y < .4:
        y = .4
    if y > 1:
        y = 1
    try:
        plt.scatter(i/500, y, s=6, c=[((0.4 + y)/2,y/2,(1.3-y)/2)])
    except:
        pass

plt.xlabel('density (in %)')
plt.ylabel('accuracy')
#plt.legend()
plt.show()

"""
plt.figure()
plt.plot(test_x, label='data')
plt.plot(torch.round(predicted_test),label='prediction')
plt.plot(test_y, label='target')
plt.legend()
plt.plot()
"""

ENDTIME = time.time()
TIME = ENDTIME - STARTTIME
print("It took",TIME,"seconds.")

##
