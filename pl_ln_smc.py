import numpy as np 
import scipy.stats as stat
import matplotlib.pyplot as plt
import math as math

data=np.load("PTZ-WILDTYPE-02_2photon_sess-01-6dpf_BLN_run-01_0.590bin0.10nnbav.npy")
sizes=data[0,:]
M=len(sizes)
a=min(sizes)
b=max(sizes)

def powerlaw(n,lam):
    zeta=np.sum(1.0/np.arange(a,b+1)**lam)
    return(n**(-lam)/zeta)

def LogPrior(x):
    #return(np.repeat(-np.log(5-0.1),len(x)))
    return(stat.norm.logpdf(x,1,3))

def LogLikelihood(lam):
    zetamat=np.power.outer(1.0/np.arange(a,b+1),lam)
    zeta=np.sum(zetamat,0)
    nprod=-lam*np.sum(np.log(sizes))
    norm=-M*np.log(zeta)
    loglik=nprod+norm
    return(loglik) 

def move_part(v,h):
    npart=len(v)
    ll0=h*LogLikelihood(v) + LogPrior(v)
    fac=np.random.gamma(1000,0.001,npart)
    v_new = v*fac 
    ll1=h*LogLikelihood(v_new) + LogPrior(v_new)
    alpha = ll1-ll0 + stat.gamma.logpdf(1.0/fac,1000,scale=0.001) - stat.gamma.logpdf(fac,1000,scale=0.001)
    u=np.random.uniform(size=npart)
    v2=v
    mask=np.log(u)<alpha
    v2[mask]=v_new[mask]
    return(v2)

def SMC(npart,steps=11):
    protocol=np.linspace(0,1,steps)
    LogML=0

    lambda_sample=np.random.normal(1,3,npart)

    W = np.repeat(1.0/npart,npart)
    logW=-np.log(npart)

    k=0
    for h in protocol[1:]:
        
        ESS=1.0/np.sum(W**2)
        #if ESS<npart/2.0:
        if True:
            lambda_sample=np.random.choice(lambda_sample,npart,True,W)
            W=np.repeat(1.0/npart,npart)
            logW=-np.log(npart)

        delta=protocol[k+1]-protocol[k]
        log_w_inc = delta*LogLikelihood(lambda_sample)  
        log_w_un = log_w_inc + logW

        lwun_max=np.max(log_w_un)

        W = np.exp(log_w_un-lwun_max)
        W = W/np.sum(W)

        logML_inc = lwun_max+np.log(np.sum(np.exp(log_w_un-lwun_max)))
        logW=log_w_un-logML_inc

        LogML=LogML+logML_inc
        
        
        lambda_sample=move_part(lambda_sample,h)

        k=k+1

    mean_lambda = np.dot(lambda_sample,W)
    return([mean_lambda,ESS,lambda_sample,W,LogML])

def plot_samples(sample):
    lambda_sample=sample[2]
    weights=sample[3]

    plt.hist(lambda_sample,weights=weights,bins=np.linspace(2.5,2.8))
    plt.show()

def plotcomp(lam):
    x = np.linspace(a,b,40) 
    plt.hist(sizes,40,log=True,density=True)
    plt.plot(x,powerlaw(x,lam))
    plt.show()

