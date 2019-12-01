# sid199
Protein structure-function relationship

#from Bio import SeqIO
x= open('C:/sidrah/study/ACB/project/New folder/keratin.pdb')
y = open('C:/sidrah/study/ACB/project/New folder/hemoglobin.pdb')
import Bio.PDB
import numpy as np
path = 'C:/sidrah/study/ACB/project/New folder/keratin.pdb'
p = Bio.PDB.PDBParser()
structure = p.get_structure('myStructureName', path)
v = [a.get_vector() for a in structure.get_atoms()]
xkeratin=[]
ykeratin=[]
zkeratin=[]
i=0
with open("C:/sidrah/study/ACB/project/New folder/vector.txt","w") as result:
    for a in structure.get_atoms():
        v=a.get_vector()
        #print(v[0], v[1], v[2])
        xkeratin.append(v[0])
        ykeratin.append(v[1])
        zkeratin.append(v[2])

y = open('C:/sidrah/study/ACB/project/New folder/hemoglobin.pdb')
path = 'C:/sidrah/study/ACB/project/New folder/hemoglobin.pdb'
p = Bio.PDB.PDBParser()
structure = p.get_structure('myStructureName', path)
v = [a.get_vector() for a in structure.get_atoms()]
xheme=[]
yheme=[]
zheme=[]
i=0
for a in structure.get_atoms():
    v=a.get_vector()
    #print(v[0], v[1], v[2])
    xheme.append(v[0])
    yheme.append(v[1])
    zheme.append(v[2])
xkerdiff = [xkeratin[i+1] - xkeratin[i] for i in range(len(xkeratin)-1)]
ykerdiff = [ykeratin[i+1] - ykeratin[i] for i in range(len(ykeratin)-1)]
zkerdiff = [zkeratin[i+1] - zkeratin[i] for i in range(len(zkeratin)-1)]
xhemediff = [xheme[i+1] - xheme[i] for i in range(len(xheme)-1)]
yhemediff = [xheme[i+1] - yheme[i] for i in range(len(yheme)-1)]
zhemediff = [xheme[i+1] - zheme[i] for i in range(len(zheme)-1)]
from numpy import array
from numpy.linalg import norm
t=[]
kertvalue=[]
hemetvalue=[]
for i in range(len(xkerdiff)):
    kertvalue.append(norm([xkerdiff[i], ykerdiff[i], zkerdiff[i]]))
for i in range(len(xhemediff)):
    hemetvalue.append(norm([xhemediff[i], yhemediff[i], zhemediff[i]]))
    #print(kertvalue[i])
kvalue = kertvalue[len(kertvalue)-1]
hvalue = hemetvalue[len(hemetvalue)-1]

for i in range(len(kertvalue)-1):
    kertvalue[i]=kertvalue[i]/kvalue
for i in range(len(hemetvalue)-1):
    hemetvalue[i]=hemetvalue[i]/hvalue
    
    
    
import numpy as np
qxvaluekeratin =[]
qyvaluekeratin =[]
qzvaluekeratin =[]
qxvalueheme =[]
qyvalueheme =[]
qzvalueheme = []
from numpy import diff
from sympy import *

import math

for i in range(len(xkeratin)-1):
    x = xkeratin[i]/kertvalue[i]
    xn = norm(x)
    xden=math.sqrt(xn)
    final = x/xden
    qxvaluekeratin.append(final)
    
    
    y = ykeratin[i]/kertvalue[i]
    yn = norm(y)
    yden=math.sqrt(yn)
    final1 = y/yden
    qyvaluekeratin.append(final1)
    
    z = zkeratin[i]/kertvalue[i]
    zn = norm(z)
    zden=math.sqrt(zn)
    final2 = z/zden
    qzvaluekeratin.append(final2)
    
    
for i in range(len(xheme)-1):
    x = xheme[i]/hemetvalue[i]
    xn = norm(x)
    xden=math.sqrt(xn)
    final = x/xden
    qxvalueheme.append(final)
    
    y = yheme[i]/hemetvalue[i]
    yn = norm(y)
    yden=math.sqrt(yn)
    final1 = y/yden
    qyvalueheme.append(final1)
    
    
    z = zheme[i]/hemetvalue[i]
    zn = norm(z)
    zden=math.sqrt(zn)
    final2 = z/zden
    qzvalueheme.append(final2)

T = list(set(kertvalue+hemetvalue))    
qxker=[]
qyker=[]
qzker=[]
qxheme=[]
qyheme=[]
qzheme=[]

for i in range(len(xkeratin)-1):
    x = xkeratin[i]/T
    xn = norm(x)
    xden=math.sqrt(xn)
    final = x/xden
    qxker.append(final)
    
    
    y = ykeratin[i]/T
    yn = norm(y)
    yden=math.sqrt(yn)
    final1 = y/yden
    qyker.append(final1)
    
    z = zkeratin[i]/T
    zn = norm(z)
    zden=math.sqrt(zn)
    final2 = z/zden
    qzker.append(final2)

for i in range(len(xheme)-1):
    x = xheme[i]/T
    xn = norm(x)
    xden=math.sqrt(xn)
    final = x/xden
    qxheme.append(final)
    
    
    y = yheme[i]/T
    yn = norm(y)
    yden=math.sqrt(yn)
    final1 = y/yden
    qyheme.append(final1)
    
    z = zheme[i]/T
    zn = norm(z)
    zden=math.sqrt(zn)
    final2 = z/zden
    qzheme.append(final2)

qkeratin = np.vstack((qxker, qyker, qzker))
qheme = np.vstack((qxheme, qyheme, qzheme))

qhemet = np.transpose(qheme)
print(qhemet.shape)
print(qkeratin.shape)

RA=np.dot(qkeratin,qhemet)


from numpy import array
from scipy.linalg import svd
u,s,v= svd(RA)

k=len(s)
reconst_matrix = np.dot(u[:,:k],np.dot(np.diag(s[:k]),v[:k,:]))

Q2R= np.dot(reconst_matrix,qheme)

EW= []
for i in range(len(qkeratin)-1):
    EW.append(Q2R[i]-qkeratin[i])


inf = 1000000000
def floyd(x, EW):
    for k in range(x):
        for i in range(x):
            for j in range(x):
                EW[i][j] = min(EW[i][j], EW[i][k]+ EW[k][j])
    print("o/d", end=' ')
    for i in range(x):
        print("\t{:d}".format(i+1), end='')
    for i in range(x):
        print("{:d}".format(i+1), end='')
        for j in range(x):
            print("\t{:d}".format(EW), end='')
floyd(4526,EW)
        
    
qkeratint=np.transpose(qkeratin)
RAA = np.dot(qheme,qkeratint)

u1,s1,v1= svd(RAA)
k1= len(s1)
reconst_matrix1 = np.dot(u1[:,:k1],np.dot(np.diag(s1[:k1]),v1[:k1,:]))

Q2R2= np.dot(reconst_matrix1,qkeratin)

TR = np.transpose(T)

D = np.dot(Q2R,TR)
print(D.shape)

geo =[]
import math
for i in range(len(D)):
    x= math.cos((i)-1)
    geo.append(x)
    
print(geo)

import plotly.express as px
import pandas as pd
f = px.scatter(x,geo)
f.show()


