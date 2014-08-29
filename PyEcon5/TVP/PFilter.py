# -*- coding: utf-8 -*-
from TVP import *



# Particle filter using array states -----------------------------------

def PFilter_A(DD,X,ww,Dyn,*args,**kwords):
        """
        Particle filtering for array startes 
        
        X0:  2Darray(nstate, NN)
        ww:  Function returns residual via observation function
        Dyn: Dynamic transition function of state
        DD:  DataFrame contains endog and exog
        
        args: something to path into predefined ww(*args)
        
        kwords:
            start: start of time iteration              
            sigw:  1Darray(nstate), stdev of observation error
            merge: Bool, True= use merging particle filter
            
        Return:
            tuple(X,Xhat,LOGLIK)
        """
        
        # keyword installation
        start=0; merge=False;
        if kwords.has_key('start'): start=kwords['start']
        if kwords.has_key('sigw'): sigw=kwords['sigw']
        if kwords.has_key('merge'): merge=kwords['merge']        
        
        # Dimention setting
        ns = X.shape[0]
        NN = X.shape[1]
        TT = DD.shape[0]
        ll = len(sigw)
        
        # Time iter skelton
        Xhat=zeros((ns,TT))
        plpath=[]
     
        # Start of time iter       
        t=start
        while t<=TT-1:
            # Reset particle skelton
            Xf=zeros((ns,NN))
            pl=zeros(NN)
            
            
            for i in range(NN):
                Xf[:,i]= Dyn(X[:,i])
                w = ww(Xf[:,i],DD,t,*args)
                
                
                pl[i] = array([log(1./sigw[i]*((1+(1.*(abs(w[i])-0.))/sigw[i])**(-1/1.-1))) for i in range(ll)]).sum()
                                
            # Selection 
            plpath.append(pl.mean())
            dpl=exp((pl-max(pl)))
            rpl=dpl/sum(dpl)
            
            # Resampling
            if merge == True: # Merging Particle Filter
                Xmm=zeros((ns,3*NN))
                Afin=array([3./4.,(sqrt(13.)+1.)/8.,-(sqrt(13.)-1.)/8.])
                
                for j in range(3*NN):
                    xj=(j+1-0.5)/(3*NN)
                    cum_rpl=0
                    for h in range(NN):
                        cum_rpl+=rpl[h]
                        if cum_rpl>=xj:
                            Xmm[:,j]=Xf[:,h] 
                            break
                # Merging
                for k in range(ns):
                    Xmvec=Xmm[k,:]
                    random.shuffle(Xmvec)
                    Xmmat=Xmvec.reshape((3,NN))
                    X[k,:]=dot(Afin,Xmmat)
                
                
            
            else:  # Particle Filter
                for j in range(NN):
                    xj=(j+1-0.5)/(NN)
                    cum_rpl=0
                    for h in range(NN):
                        cum_rpl+=rpl[h]
                        if cum_rpl>=xj:
                            X[:,j]=Xf[:,h] 
                            break 
                        
        
            # Storing mean path of state variables
            Xhat[:,t]=mean(X,axis=1) #median(X[:nx,:],axis=1)       
        
            t+=1
        
        LOGLIK=sum(plpath)
               
        return X, Xhat, LOGLIK
        
# ----------------------------------------------------------------------
        
        
        
        
        
        

# Particle filter using dictionary states ------------------------------
def PFilter_D(DD,X,ww,Dyn,*args,**kwords):
    """
    Particle filtering for dictionary startes 
    
    X:   List, containing NN dictionaries that stores heterogeneouse shaped matrixes 
    ww:  Function returns residual via observation function
    Dyn: Dynamic transition function of state
    DD:  DataFrame contains endog and exog
    
    args: something to path into predefined ww(*args), and Dyn(*args)
    
    kwords:
        start: start of time iteration              
        sigw:  1Darray(nstate), stdev of observation error
        merge: Bool, True= use merging particle filter
        
    Return:
        tuple(X, Xhat, resid, othpath, LOGLIK)
    """
    # keyword installation
    
    start=0; merge=True;
    if kwords.has_key('start'): start=kwords['start']
    if kwords.has_key('sigw'): sigw=kwords['sigw']
    if kwords.has_key('merge'): merge=kwords['merge']        
    
    # Dimention setting 
    NN = len(X)
    TT = DD.shape[0]
    ll = len(sigw)
    
    # ## Start of Iteration ########################################
    
    # skeleton
    Xhat=[]
    Xstd=[]
    plpath=[]
    resid=[]
    othpath=[]
    
    
    # /// Iteration ///
    t=start
    while t<=TT-1:
        # reset skelton
        Xf=[]
        Xmm=[]
        Xnew=[]
        pl=[]
        oth=[]
    
    
        for x in X:
    
        # Evaluate X by loglik ######################################
        # /// forecasting via system ///
            xf=Dyn(x, *args)
            Xf.append(xf)
    
            w,ot=ww(xf,DD,t, *args)
            oth.append(ot)
    
    
            # evaluate particle likelihood
            pl_xi=1. ## Generalized Paleto 1./sigw[i]*((1+(pl_xi*(abs(w[i])-0.))/sigw[i])**(-1/pl_xi-1))
            ## Normal 1./(sqrt(2*pi*sigw[i]**2))*exp(-((w[i]-0)**2/(2*sigw[i]**2)))
    
            pl.append(array([log(1./sigw[i]*((1+(pl_xi*(abs(w[i])-0.))/sigw[i])**(-1/pl_xi-1))) for i in range(ll)]).sum())
    
        # /// Relative particle likelihood ///
        pl=array(pl)
        plpath.append(pl.mean()) 
        dpl=exp((pl-max(pl)))
        rpl=dpl/sum(dpl)
    
        # /// Latent variable saving ///
        othpath.append(average(oth,axis=0))
    
    
    
        # Merging Particle Filtering ###################################
        
        if merge==True:
            mm=3
            Afin=array([3./4.,(sqrt(13.)+1.)/8.,-(sqrt(13.)-1.)/8.])
            # /// Resampling ///
            for j in range(mm*NN):
                xj=(j+1-0.5)/(mm*NN)
                cum_rpl=0
                for h in range(NN):
                    cum_rpl+=rpl[h]
                    if cum_rpl>=xj:
                        Xmm.append(Xf[h]) ## here we get filtered particles
                        break
        
            # /// Merging ///
            random.shuffle(Xmm)
        
            for i in range(NN):
                Xtri=Xmm[(mm*i):(mm*(i+1))]
                xnew = dict_sumprod(Xtri,Afin)
                Xnew.append(xnew)
            X=Xnew
            
        else:
            # /// Resampling ///
            for j in range(NN):
                xj=(j+1-0.5)/(NN)
                cum_rpl=0
                for h in range(NN):
                    cum_rpl+=rpl[h]
                    if cum_rpl>=xj:
                        Xnew.append(Xf[h]) ## here we get filtered particles
                        break
            X=Xnew
    
    
        ## Storing mean path of state variables
        Xhat.append(dict_mean(X))
        Xstd.append(dict_std(X))
        resid.append(ww(dict_mean(X),DD,t)[0])
       
        
    
    
        t+=1
        print str(round(float(t-start)/float(TT-start)*100,2))+'%'
    
    
    
    # Total log-likelihood
    LOGLIK=sum(plpath)
    
    return X, Xhat, Xstd, resid, othpath, LOGLIK


        
        
        