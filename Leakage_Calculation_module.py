import numpy as np
import scipy 
from scipy.integrate import nquad, dblquad,quad
from scipy.special import erfinv
from scipy import integrate,linalg
from scipy.special import erfinv, erf
#################################################################
#probabilities
def pk(k,alpha,p):
    if k==0:
        ak=-np.infty
        bk=((2/2**p)*(k+1)-1)*alpha
    elif k==2**p-1:
         ak=((2/2**p)*k-1)*alpha
         bk=np.infty
    else: 
        ak=((2/2**p)*k-1)*alpha
        bk=((2/2**p)*(k+1)-1)*alpha
    A=erf((ak)/np.sqrt(2))
    B=erf((bk)/np.sqrt(2))
    return (1/2)*(B-A)

def pk_x(k,x,rho,alpha,p):
    if k==0:
        ak=-np.infty
        bk=((2/2**p)*(k+1)-1)*alpha
    elif k==2**p-1:
         ak=((2/2**p)*k-1)*alpha
         bk=np.infty
    else: 
        ak=((2/2**p)*k-1)*alpha
        bk=((2/2**p)*(k+1)-1)*alpha
    A=erf((ak-x*rho)/np.sqrt(2*(1-rho**2)))
    B=erf((bk-x*rho)/np.sqrt(2*(1-rho**2)))
    return (1/2)*(B-A)

def px(x):
    return (1/np.sqrt((2*np.pi)))*np.exp(-x**2/2)


def pkx(k,x,rho,alpha,p):
    return px(x)*pk_x(k,x,rho,alpha,p)

#conditional entropy, conditioanl variance, & finite-size leakage
def cond_ent_fun(x,k,rho,alpha,p):
    A=pk_x(k,x,rho,alpha,p)
    if A==0:
        S=0
    else:
        S=pkx(k,x,rho,alpha,p)*(-np.log2(pk_x(k,x,rho,alpha,p)))
    return S

def cond_var_fun(x,k,rho,alpha,p):
    A=pk_x(k,x,rho,alpha,p)
    if A==0:
        S=0
    else:
        S=pkx(k,x,rho,alpha,p)*(-np.log2(pk_x(k,x,rho,alpha,p)))**2
    return S

def sum_ent(x,rho,alpha,p):
    S=0
    for k in range(2**p):
        S+=cond_ent_fun(x,k,rho,alpha,p)
    return S

def sum_var(x,rho,alpha,p):
    S=0
    for k in range(2**p):
        S+=cond_var_fun(x,k,rho,alpha,p)
    return S

def cond_ent(rho,alpha,p):
    return integrate.quad(sum_ent,-4,4,args=(rho,alpha,p))[0]

def cond_var(rho,alpha,p):
    return integrate.quad(sum_var,-4,4,args=(rho,alpha,p))[0]

def leak(n,p_ec,rho,alpha,p):
    H=cond_ent(rho,alpha,p)
    V=cond_var(rho,alpha,p)-H**2
    revPhi=erfinv(p_ec)
    RestTerms=(1/2)*np.log2(n)
    return n*H+np.sqrt(n*V)*revPhi


#descretization & beta_quant
def Hk(alpha,p):
    S=0
    for k in range(2**p):
        A=pk(k,alpha,p)
        if A==0 or A==1:
            S+=0
        else:
            S+=-A*np.log2(A)
    return S

def sum_Hkx(x,rho,alpha,p):
    S=0
    for k in range(2**p):
        A=pkx(k,x,rho,alpha,p)
        if A==0 or A==1:
            S+=0
        else:
            S+=-A*np.log2(A)
    return S

def Hkx(rho,alpha,p):
    return integrate.quad(sum_Hkx,-4,4,args=(rho,alpha,p))[0]
    
def IMxk(rho,alpha,p):
    Hx=(1/2)*np.log2(2*np.pi*np.exp(1))
    return Hx+Hk(alpha,p)-Hkx(rho,alpha,p)

def IMxy(rho):
    return -(1/2)*np.log2(1-rho**2)

def beta_quant(rho,alpha,p):
    return IMxk(rho,alpha,p)/IMxy(rho)
#===============================================================


# ###################################
# n=10**4
# p_ec=0.99
# rho=0.4
# alpha=8
# p=10
# x=0.3
# k=0
# start=time.time()
# #print(cond_ent_fun(x,k,rho,alpha,p))
# print(leak(n,p_ec,rho,alpha,p))
# #print(cond_ent_0(rho,alpha,p))
# print("Time",time.time()-start)
#-----------------------------------------
# start=time.time()
# print(cond_ent_1(rho,alpha,p))
# print("Time",time.time()-start)