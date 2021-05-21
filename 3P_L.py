#!/usr/bin/env python
# coding: utf-8

# # Model

import numpy as np
import pandas as pd
import collections
import random
import math
import matplotlib.pyplot as plt
from time import time
from matplotlib import colors
from matplotlib.pyplot import figure



# ### Temperature and volatility

beta = float(0.6)
Phi = float(0.6)
print('beta: ', beta, ' - phi = ', Phi)

# ### Energy parameters

C = int(1)
J = np.zeros((3,3))
J[1][1] = float(0)
J[2][1] = float(1)
J[2][2] = float(0)
J[0][2] = float(6)
J[0][0] = float(0)
J[0][1] = float(1)
J[1][2] = J[2][1]
J[2][0] = J[0][2]
J[1][0] = J[0][1]

# ### Initial percentage

s0 = float(40)
s0 = s0/100
spos = float(30)
spos = spos/100
sneg = float(30)
sneg = sneg/100
if s0+spos+sneg != 1 :
    print('!!! Wrong data !!!')
Pstar = spos/(1-s0)


# ### Initial lattice

print('Building the lattice...')
L1 = int(128)
L2 = int(128)
L = [L1, L2]
Lambda = int(4)
if L1%Lambda != 0 or L2%Lambda  != 0 :
    print(' ! ! !  E R R O R  ! ! !')
    print('(L1 or L2) % Lambda != 0')
# Generation if the initial configuration.
l1 = L1//Lambda
l2 = L2//Lambda
l = [l1, l2]
start = time()
# Generation if the initial configuration.
A = np.random.choice((0, 1, 2), size=(L1,L2), replace=True, p=(spos,s0,sneg))
end = time()
print('Time to build the model: ', (end-start)/60, ' minutes.')

# Print of the visual map.
col = colors.ListedColormap(['yellow', 'red', 'blue'])
lin1 = np.linspace(-0.5, (L1-1)+0.5, l1+1)
lin2 = np.linspace(-0.5, (L2-1)+0.5, l2+1)
fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
ax.imshow(A, cmap=col)
ax.set_xticks(lin1, minor=False)
ax.xaxis.grid(True, which='major')
ax.set_yticks(lin2, minor=False)
ax.yaxis.grid(True, which='major')
ax.grid(True, color='k', lw = 0.9)
plt.savefig('Lambda_' + str(Lambda) + '_C' + str(C) + '_b' + str(beta) + '_phi' + str(Phi) + '_L' + str(L1) + '_J' + str(J[0][2]) + '_ratio' + str(s0p) + str(sposp) + str(snegp) + '_100p.png')

print(' Initial number of y: ', np.count_nonzero(A == 0), '(', (np.count_nonzero(A == 0)*100)/(L1*L2), '%)')
print(' Initial number of r: ', np.count_nonzero(A == 1), '(', (np.count_nonzero(A == 1)*100)/(L1*L2), '%)')
print(' Initial number of b: ', np.count_nonzero(A == 2), '(', (np.count_nonzero(A == 2)*100)/(L1*L2), '%)')


# ### Main code

def energy(S) :
    H = 0
    Sx = S[int(x[0]*Lambda):int(x[0]*Lambda+Lambda), int(x[1]*Lambda):int(x[1]*Lambda+Lambda)]
    Sy = S[int(y[0]*Lambda):int(y[0]*Lambda+Lambda), int(y[1]*Lambda):int(y[1]*Lambda+Lambda)]
    for i in np.arange(Lambda) :
        for j in np.arange(Lambda) :
            H = H + J[Sx[a[0], a[1]], Sx[i, j]] + J[Sx[a[0], a[1]], Sx[i, j]]
            H = H + J[Sy[b[0], b[1]], Sy[i, j]] + J[Sy[b[0], b[1]], Sy[i, j]]
    H = H - J[Sx[a[0], a[1]], Sx[a[0], a[1]]] - J[Sy[b[0], b[1]], Sy[b[0], b[1]]]
    H = C*H
    
    x_num = [np.count_nonzero(Sx == 0), np.count_nonzero(Sx == 1), np.count_nonzero(Sx == 2)]
    y_num = [np.count_nonzero(Sy == 0), np.count_nonzero(Sy == 1), np.count_nonzero(Sy == 2)]

    n_x1 = [0,0,0]
    n_y1 = [0,0,0]
    n_x2 = [0,0,0]
    n_y2 = [0,0,0]
    for i in [-1,1] :
        neighb_x1 = [(x[0]+i)%l1, x[1]]
        neighb_x2 = [x[0], (x[1]+i)%l2]
        neighb_y1 = [(y[0]+i)%l1, y[1]]
        neighb_y2 = [y[0], (y[1]+i)%l2]
        
        if y != neighb_x1 or y != neighb_x2 :

            S_nx = S[(neighb_x1[0]*Lambda):(neighb_x1[0]*Lambda+Lambda), (neighb_x1[1]*Lambda):(neighb_x1[1]*Lambda+Lambda)]
            n_x1 = [np.count_nonzero(S_nx == 0), np.count_nonzero(S_nx == 1), np.count_nonzero(S_nx == 2)]

            S_nx = S[(neighb_x2[0]*Lambda):(neighb_x2[0]*Lambda+Lambda), (neighb_x2[1]*Lambda):(neighb_x2[1]*Lambda+Lambda)]
            n_x2 = [np.count_nonzero(S_nx == 0), np.count_nonzero(S_nx == 1), np.count_nonzero(S_nx == 2)]

        if x != neighb_y1 or x != neighb_y2 :

            S_ny = S[(neighb_y1[0]*Lambda):(neighb_y1[0]*Lambda+Lambda), (neighb_y1[1]*Lambda):(neighb_y1[1]*Lambda+Lambda)]
            n_y1 = [np.count_nonzero(S_ny == 0), np.count_nonzero(S_ny == 1), np.count_nonzero(S_ny == 2)]

            S_ny = S[(neighb_y2[0]*Lambda):(neighb_y2[0]*Lambda+Lambda), (neighb_y2[1]*Lambda):(neighb_y2[1]*Lambda+Lambda)]
            n_y2 = [np.count_nonzero(S_ny == 0), np.count_nonzero(S_ny == 1), np.count_nonzero(S_ny == 2)]

        for o in range(3) :
            H = H + x_num[0]*n_x1[o]*J[0,o] + x_num[1]*n_x1[o]*J[1,o] + x_num[2]*n_x1[o]*J[2,o]
            H = H + x_num[0]*n_x2[o]*J[0,o] + x_num[1]*n_x2[o]*J[1,o] + x_num[2]*n_x2[o]*J[2,o]
            H = H + y_num[0]*n_y1[o]*J[0,o] + y_num[1]*n_y1[o]*J[1,o] + y_num[2]*n_y1[o]*J[2,o]
            H = H + y_num[0]*n_y2[o]*J[0,o] + y_num[1]*n_y2[o]*J[1,o] + y_num[2]*n_y2[o]*J[2,o]
    
    return(H)
            


S = A.copy()


t = 0
y = [0,0]
NL = L1 * L2

# Stop parameter

count = np.count_nonzero(S == 1)
stop1 = count*75//100
stop2 = count*50//100
stop3 = count*25//100
stop4 = count*10//100


print('The red is evaporating...')
start = time()

# Beginning of the code with the stop condition.
while count > stop1 :
    
    
    # Select the first CELL!!!
    x = [np.random.randint(l1), np.random.randint(l2)]
    # Select a direction (x1 or x2).
    [c] = random.choices((0,1), weights=(0.5,0.5), k=1)
    # Update the second CELL!!! that will form the bond.
    y[c] = (x[c]+1) % l[c]
    y[(c+1)%2] = x[(c+1)%2]
    
    
    # Select a site in each CELL!!!
    a = [np.random.randint(Lambda), np.random.randint(Lambda)]
    b = [np.random.randint(Lambda), np.random.randint(Lambda)]
    
    X = [int(x[0]*Lambda + a[0]), int(x[1]*Lambda + a[1])]
    Y = [int(y[0]*Lambda + b[0]), int(y[1]*Lambda + b[1])]


    
    
    
    # Initial check on conditions for the evaporation process.
    if (y[0] == 0 and c == 0 and S[Y[0], Y[1]] == 1) :
        
        # The new probability Pstar is computed by updating
        #     the number of blue and red particles.
        Pstar = np.count_nonzero(S == 2)/(NL-np.count_nonzero(S == 1))

        # The red particle in the selected bond
        #     is replaced by the blue or the yellow one
        #     according to the probability Pstar.
        [S[Y[0], Y[1]]] = random.choices((2,0), weights=(Pstar,1-Pstar), k=1)

        
    elif (c == 0 and  S[Y[0], Y[1]] == 1) :
        
        # A copy of the lattice S is created.
        S1 = np.copy(S)
        # Swapping of the two particles in the copy.
        S1[X[0],X[1]] , S1[Y[0],Y[1]]  = S1[Y[0],Y[1]] , S1[X[0],X[1]]
        
        # Choice of the configuration (S or S1) with probability Phi.
        [S] = random.choices((S1,S), weights=(Phi,1-Phi), k=1)
        
        if S[X[0], X[1]] == S1[Y[0], Y[1]] :
            H = energy(S)
            H1 = energy(S1)
            # Difference between energies (delta_H).
            delta_H = H1 - H

            # Is the energy decreasing after the swap?
            if delta_H < 0 :
                # Then accept the new configuration.
                S = np.copy(S1)
            # Is the energy increasing after the swap?     
            else :
                # Then accept the "uphill move" with probability
                enrg = math.exp(-beta*delta_H/(Lambda**2))
                [S] = random.choices((S1,S), weights=(enrg,1-enrg), k=1)
         
    else :
        # A copy of the lattice S is created.
        S1 = np.copy(S)
        # Swapping of the two particles in the copy.
        S1[X[0],X[1]] , S1[Y[0],Y[1]]  = S1[Y[0],Y[1]] , S1[X[0],X[1]]
        
        H = energy(S)
        H1 = energy(S1)
        # Difference between energies (delta_H).
        delta_H = H1 - H
        
        # Is the energy decreasing after the swap?
        if delta_H < 0 :
            # Then accept the new configuration.
            S = np.copy(S1)
        # Is the energy increasing after the swap?     
        else :
            # Then accept the "uphill move" with probability
            enrg = math.exp(-beta*delta_H/(Lambda**2))
            [S] = random.choices((S1,S), weights=(enrg,1-enrg), k=1)
            
    t+=1
    if t % 10000 == 0 : 
        count = np.count_nonzero(S == 1)
    
end = time()
print('Running time: ', (end - start)/60 , 'minutes, with ', t, 'iterations')
col = colors.ListedColormap(['yellow', 'red', 'blue'])
lin1 = np.linspace(-0.5, (L1-1)+0.5, l1+1)
lin2 = np.linspace(-0.5, (L2-1)+0.5, l2+1)
fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
ax.imshow(S, cmap=col)
ax.set_xticks(lin1, minor=False)
ax.xaxis.grid(True, which='major')
ax.set_yticks(lin2, minor=False)
ax.yaxis.grid(True, which='major')
ax.grid(True, color='k', lw = 0.9)
plt.savefig('Lambda_' + str(Lambda) + '_C' + str(C) + '_b' + str(beta) + '_phi' + str(Phi) + '_L' + str(L1) + '_J' + str(J[0][2]) + '_ratio' + str(s0p) + str(sposp) + str(snegp) + '_75p.png')

print(' Actual number of y: ', np.count_nonzero(S == 0), '(', (np.count_nonzero(S == 0)*100)/(L1*L2), '%)')
print(' Actual number of r: ', np.count_nonzero(S == 1), '(', (np.count_nonzero(S == 1)*100)/(L1*L2), '%)')
print(' Actual number of b: ', np.count_nonzero(S == 2), '(', (np.count_nonzero(S == 2)*100)/(L1*L2), '%)')





t = 0

print('The red is evaporating again...')
start = time()

# Beginning of the code with the stop condition.
while count > stop2 :

    # Select the first CELL!!!
    x = [np.random.randint(l1), np.random.randint(l2)]
    # Select a direction (x1 or x2).
    [c] = random.choices((0,1), weights=(0.5,0.5), k=1)
    # Update the second CELL!!! that will form the bond.
    y[c] = (x[c]+1) % l[c]
    y[(c+1)%2] = x[(c+1)%2]

    # Select a site in each CELL!!!
    a = [np.random.randint(Lambda), np.random.randint(Lambda)]
    b = [np.random.randint(Lambda), np.random.randint(Lambda)]

    X = [int(x[0]*Lambda + a[0]), int(x[1]*Lambda + a[1])]
    Y = [int(y[0]*Lambda + b[0]), int(y[1]*Lambda + b[1])]


    # Initial check on conditions for the evaporation process.
    if (y[0] == 0 and c == 0 and S[Y[0], Y[1]] == 1) :

        # The new probability Pstar is computed by updating
        #     the number of blue and red particles.
        Pstar = np.count_nonzero(S == 2)/(NL-np.count_nonzero(S == 1))

        # The red particle in the selected bond
        #     is replaced by the blue or the yellow one
        #     according to the probability Pstar.
        [S[Y[0], Y[1]]] = random.choices((2,0), weights=(Pstar,1-Pstar), k=1)

    elif (c == 0 and  S[Y[0], Y[1]] == 1) :

        # A copy of the lattice S is created.
        S1 = np.copy(S)
        # Swapping of the two particles in the copy.
        S1[X[0],X[1]] , S1[Y[0],Y[1]]  = S1[Y[0],Y[1]] , S1[X[0],X[1]]

        # Choice of the configuration (S or S1) with probability Phi.
        [S] = random.choices((S1,S), weights=(Phi,1-Phi), k=1)

        if S[X[0], X[1]] == S1[Y[0], Y[1]] :
            H = energy(S)
            H1 = energy(S1)
            # Difference between energies (delta_H).
            delta_H = H1 - H

            # Is the energy decreasing after the swap?
            if delta_H < 0 :
                # Then accept the new configuration.
                S = np.copy(S1)
            # Is the energy increasing after the swap?
            else :
                # Then accept the "uphill move" with probability
                enrg = math.exp(-beta*delta_H/(Lambda**2))
                [S] = random.choices((S1,S), weights=(enrg,1-enrg), k=1)

    else :
        # A copy of the lattice S is created.
        S1 = np.copy(S)
        # Swapping of the two particles in the copy.
        S1[X[0],X[1]] , S1[Y[0],Y[1]]  = S1[Y[0],Y[1]] , S1[X[0],X[1]]

        H = energy(S)
        H1 = energy(S1)
        # Difference between energies (delta_H).
        delta_H = H1 - H

        # Is the energy decreasing after the swap?
        if delta_H < 0 :
            # Then accept the new configuration.
            S = np.copy(S1)
        # Is the energy increasing after the swap?
        else :
            # Then accept the "uphill move" with probability
            enrg = math.exp(-beta*delta_H/(Lambda**2))
            [S] = random.choices((S1,S), weights=(enrg,1-enrg), k=1)

    t+=1
    if t % 10000 == 0 :
        count = np.count_nonzero(S == 1)

end = time()
print('Running time: ', (end - start)/60 , 'minutes, with ', t, 'iterations')
col = colors.ListedColormap(['yellow', 'red', 'blue'])
lin1 = np.linspace(-0.5, (L1-1)+0.5, l1+1)
lin2 = np.linspace(-0.5, (L2-1)+0.5, l2+1)
fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
ax.imshow(S, cmap=col)
ax.set_xticks(lin1, minor=False)
ax.xaxis.grid(True, which='major')
ax.set_yticks(lin2, minor=False)
ax.yaxis.grid(True, which='major')
ax.grid(True, color='k', lw = 0.9)
plt.savefig('Lambda_' + str(Lambda) + '_C' + str(C) + '_b' + str(beta) + '_phi' + str(Phi) + '_L' + str(L1) + '_J' + str(J[0][2]) + '_ratio' + str(s0p) + str(sposp) + str(snegp) + '_50p.png')

print(' Actual number of y: ', np.count_nonzero(S == 0), '(', (np.count_nonzero(S == 0)*100)/(L1*L2), '%)')
print(' Actual number of r: ', np.count_nonzero(S == 1), '(', (np.count_nonzero(S == 1)*100)/(L1*L2), '%)')
print(' Actual number of b: ', np.count_nonzero(S == 2), '(', (np.count_nonzero(S == 2)*100)/(L1*L2), '%)')



t = 0

print('The red is evaporating again...')
start = time()

# Beginning of the code with the stop condition.
while count > stop3 :

    # Select the first CELL!!!
    x = [np.random.randint(l1), np.random.randint(l2)]
    # Select a direction (x1 or x2).
    [c] = random.choices((0,1), weights=(0.5,0.5), k=1)
    # Update the second CELL!!! that will form the bond.
    y[c] = (x[c]+1) % l[c]
    y[(c+1)%2] = x[(c+1)%2]

    # Select a site in each CELL!!!
    a = [np.random.randint(Lambda), np.random.randint(Lambda)]
    b = [np.random.randint(Lambda), np.random.randint(Lambda)]

    X = [int(x[0]*Lambda + a[0]), int(x[1]*Lambda + a[1])]
    Y = [int(y[0]*Lambda + b[0]), int(y[1]*Lambda + b[1])]


    # Initial check on conditions for the evaporation process.
    if (y[0] == 0 and c == 0 and S[Y[0], Y[1]] == 1) :

        # The new probability Pstar is computed by updating
        #     the number of blue and red particles.
        Pstar = np.count_nonzero(S == 2)/(NL-np.count_nonzero(S == 1))

        # The red particle in the selected bond
        #     is replaced by the blue or the yellow one
        #     according to the probability Pstar.
        [S[Y[0], Y[1]]] = random.choices((2,0), weights=(Pstar,1-Pstar), k=1)

    elif (c == 0 and  S[Y[0], Y[1]] == 1) :

        # A copy of the lattice S is created.
        S1 = np.copy(S)
        # Swapping of the two particles in the copy.
        S1[X[0],X[1]] , S1[Y[0],Y[1]]  = S1[Y[0],Y[1]] , S1[X[0],X[1]]

        # Choice of the configuration (S or S1) with probability Phi.
        [S] = random.choices((S1,S), weights=(Phi,1-Phi), k=1)

        if S[X[0], X[1]] == S1[Y[0], Y[1]] :
            H = energy(S)
            H1 = energy(S1)
            # Difference between energies (delta_H).
            delta_H = H1 - H

            # Is the energy decreasing after the swap?
            if delta_H < 0 :
                # Then accept the new configuration.
                S = np.copy(S1)
            # Is the energy increasing after the swap?
            else :
                # Then accept the "uphill move" with probability
                enrg = math.exp(-beta*delta_H/(Lambda**2))
                [S] = random.choices((S1,S), weights=(enrg,1-enrg), k=1)

    else :
        # A copy of the lattice S is created.
        S1 = np.copy(S)
        # Swapping of the two particles in the copy.
        S1[X[0],X[1]] , S1[Y[0],Y[1]]  = S1[Y[0],Y[1]] , S1[X[0],X[1]]

        H = energy(S)
        H1 = energy(S1)
        # Difference between energies (delta_H).
        delta_H = H1 - H

        # Is the energy decreasing after the swap?
        if delta_H < 0 :
            # Then accept the new configuration.
            S = np.copy(S1)
        # Is the energy increasing after the swap?
        else :
            # Then accept the "uphill move" with probability
            enrg = math.exp(-beta*delta_H/(Lambda**2))
            [S] = random.choices((S1,S), weights=(enrg,1-enrg), k=1)

    t+=1
    if t % 10000 == 0 :
        count = np.count_nonzero(S == 1)

end = time()
print('Running time: ', (end - start)/60 , 'minutes, with ', t, 'iterations')
col = colors.ListedColormap(['yellow', 'red', 'blue'])
lin1 = np.linspace(-0.5, (L1-1)+0.5, l1+1)
lin2 = np.linspace(-0.5, (L2-1)+0.5, l2+1)
fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
ax.imshow(S, cmap=col)
ax.set_xticks(lin1, minor=False)
ax.xaxis.grid(True, which='major')
ax.set_yticks(lin2, minor=False)
ax.yaxis.grid(True, which='major')
ax.grid(True, color='k', lw = 0.9)
plt.savefig('Lambda_' + str(Lambda) + '_C' + str(C) + '_b' + str(beta) + '_phi' + str(Phi) + '_L' + str(L1) + '_J' + str(J[0][2]) + '_ratio' + str(s0p) + str(sposp) + str(snegp) + '_25p.png')

print(' Actual number of y: ', np.count_nonzero(S == 0), '(', (np.count_nonzero(S == 0)*100)/(L1*L2), '%)')
print(' Actual number of r: ', np.count_nonzero(S == 1), '(', (np.count_nonzero(S == 1)*100)/(L1*L2), '%)')
print(' Actual number of b: ', np.count_nonzero(S == 2), '(', (np.count_nonzero(S == 2)*100)/(L1*L2), '%)')





t = 0

print('The red is evaporating again...')
start = time()

# Beginning of the code with the stop condition.
while count > stop4 :

    # Select the first CELL!!!
    x = [np.random.randint(l1), np.random.randint(l2)]
    # Select a direction (x1 or x2).
    [c] = random.choices((0,1), weights=(0.5,0.5), k=1)
    # Update the second CELL!!! that will form the bond.
    y[c] = (x[c]+1) % l[c]
    y[(c+1)%2] = x[(c+1)%2]

    # Select a site in each CELL!!!
    a = [np.random.randint(Lambda), np.random.randint(Lambda)]
    b = [np.random.randint(Lambda), np.random.randint(Lambda)]

    X = [int(x[0]*Lambda + a[0]), int(x[1]*Lambda + a[1])]
    Y = [int(y[0]*Lambda + b[0]), int(y[1]*Lambda + b[1])]


    # Initial check on conditions for the evaporation process.
    if (y[0] == 0 and c == 0 and S[Y[0], Y[1]] == 1) :

        # The new probability Pstar is computed by updating
        #     the number of blue and red particles.
        Pstar = np.count_nonzero(S == 2)/(NL-np.count_nonzero(S == 1))

        # The red particle in the selected bond
        #     is replaced by the blue or the yellow one
        #     according to the probability Pstar.
        [S[Y[0], Y[1]]] = random.choices((2,0), weights=(Pstar,1-Pstar), k=1)

    elif (c == 0 and  S[Y[0], Y[1]] == 1) :

        # A copy of the lattice S is created.
        S1 = np.copy(S)
        # Swapping of the two particles in the copy.
        S1[X[0],X[1]] , S1[Y[0],Y[1]]  = S1[Y[0],Y[1]] , S1[X[0],X[1]]

        # Choice of the configuration (S or S1) with probability Phi.
        [S] = random.choices((S1,S), weights=(Phi,1-Phi), k=1)

        if S[X[0], X[1]] == S1[Y[0], Y[1]] :
            H = energy(S)
            H1 = energy(S1)
            # Difference between energies (delta_H).
            delta_H = H1 - H

            # Is the energy decreasing after the swap?
            if delta_H < 0 :
                # Then accept the new configuration.
                S = np.copy(S1)
            # Is the energy increasing after the swap?
            else :
                # Then accept the "uphill move" with probability
                enrg = math.exp(-beta*delta_H/(Lambda**2))
                [S] = random.choices((S1,S), weights=(enrg,1-enrg), k=1)

    else :
        # A copy of the lattice S is created.
        S1 = np.copy(S)
        # Swapping of the two particles in the copy.
        S1[X[0],X[1]] , S1[Y[0],Y[1]]  = S1[Y[0],Y[1]] , S1[X[0],X[1]]

        H = energy(S)
        H1 = energy(S1)
        # Difference between energies (delta_H).
        delta_H = H1 - H

        # Is the energy decreasing after the swap?
        if delta_H < 0 :
            # Then accept the new configuration.
            S = np.copy(S1)
        # Is the energy increasing after the swap?
        else :
            # Then accept the "uphill move" with probability
            enrg = math.exp(-beta*delta_H/(Lambda**2))
            [S] = random.choices((S1,S), weights=(enrg,1-enrg), k=1)

    t+=1
    if t % 10000 == 0 :
        count = np.count_nonzero(S == 1)

end = time()
print('Running time: ', (end - start)/60 , 'minutes, with ', t, 'iterations')
col = colors.ListedColormap(['yellow', 'red', 'blue'])
lin1 = np.linspace(-0.5, (L1-1)+0.5, l1+1)
lin2 = np.linspace(-0.5, (L2-1)+0.5, l2+1)
fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
ax.imshow(S, cmap=col)
ax.set_xticks(lin1, minor=False)
ax.xaxis.grid(True, which='major')
ax.set_yticks(lin2, minor=False)
ax.yaxis.grid(True, which='major')
ax.grid(True, color='k', lw = 0.9)
plt.savefig('Lambda_' + str(Lambda) + '_C' + str(C) + '_b' + str(beta) + '_phi' + str(Phi) + '_L' + str(L1) + '_J' + str(J[0][2]) + '_ratio' + str(s0p) + str(sposp) + str(snegp) + '_10p.png')

print(' Actual number of y: ', np.count_nonzero(S == 0), '(', (np.count_nonzero(S == 0)*100)/(L1*L2), '%)')
print(' Actual number of r: ', np.count_nonzero(S == 1), '(', (np.count_nonzero(S == 1)*100)/(L1*L2), '%)')
print(' Actual number of b: ', np.count_nonzero(S == 2), '(', (np.count_nonzero(S == 2)*100)/(L1*L2), '%)')




