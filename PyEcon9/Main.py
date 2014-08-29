# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

import scipy
from scipy import *
from scipy import linalg, stats



from tips import *
import Tseries



from sympy import Matrix,var,log,exp, lambdify



# ====================== Particle Filter Initialization =======================


# /// Number of State variable and jump variable ///
#x = [vega,R,beta,A,mp,Sg];
nx = 6
#y = [gdp,con,l,mc,x1,x2,wage,Pi,Pio,gov,bnd];
ny = 11


# /// Data Installation ///
url=r"GoHNKM.xlsx"
Data=pd.ExcelFile(url).parse("Go",index_col="TIME")


# /// Std of Obs Vars to be used to calc loglik
obslist=   ['gdp','con','wage','Pi']
#y = [gdp,con,l,mc,x1,x2,wage,Pi,Pio,gov,bnd];
obsYflag = array([True,True,False,False,False,False,True,True,False,False,False])


# HP detrending
DD=Data[obslist].to_period(freq='Q')
DD=pd.DataFrame(DD.values-Tseries.HPF_mlt(DD.values).T,index=DD.index,columns=DD.columns)



sigw=std(DD.values,axis=0)
ll=len(obslist)

TT=DD.shape[0]


# /// Number of particles ///
NN = 50

# /// Time iter Config ///
Lag = 1 # Maximum lag used in Obs or Exo function
start=Lag+1  #TT-1#
t=start
T=TT-Lag


# /// Merging PF setting ///
merge=True
mm=3
Afin=array([3./4.,(sqrt(13.)+1.)/8.,-(sqrt(13.)-1.)/8.])





# /// StateSpace construction ///
state_keys = ['x1','x2','pars','sigs','sigVp','sigVs']

Nstate = len(state_keys)





# /// Set parameters of initial dst  ///
x1_0 = zeros(nx)
x2_0 = zeros(nx**2)

pars_0=array([1.,1.,6.,0.75,0.8,1.5,0.25,0.8,0.9,0.8])
pars_s=array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001])

sigs_0=array([0.005,0.005,0.005,0.005])
sigs_s=array([0.001,0.001,0.001,0.001])


sigVp_0=pars_s
sigVp_s=array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001])

sigVs_0=sigs_s
sigVs_s=sigs_s/10



# /// Int dst of State Vars ///
init_dst=[]
init_dst.append( array([repeat(x1_0[i],NN) for i in range(nx)]).T  )
init_dst.append( array([repeat(x2_0[i],NN) for i in range(nx**2)]).T  )

init_dst.append( array([stats.norm.rvs(pars_0[i],pars_s[i],NN) for i in range(len(pars_0))]).T  )

init_dst.append( array([stats.norm.rvs(sigs_0[i],sigs_s[i],NN) for i in range(len(sigs_0))]).T  )

init_dst.append( array([stats.norm.rvs(sigVp_0[i],sigVp_s[i],NN) for i in range(len(sigVp_0))]).T  )

init_dst.append( array([stats.norm.rvs(sigVs_0[i],sigVs_s[i],NN) for i in range(len(sigVs_0))]).T  )


X = [
    dict(zip(
        state_keys,

        [init[i] for init in init_dst]

    )) for i in range(NN)
]








# ===================== Start of Particle Filtering ===========================

# skeleton
Xhat=[]
Xstd=[]
plpath=[]
resid=[]
othpath=[]

# /// Iteration ///
t=start
while t<=TT-1:

    # Observed values
    YY = DD[obslist].values[t]

    # reset skelton
    Xf=[]
    Xmm=[]
    Xnew=[]
    pl=[]
    oth=[]


    for xx in X:


# ====================== Dynamics of State Variables ==========================


        # Sig pars and sigs
        sigVp = abs(xx['sigVp'])
        sigVs = abs(xx['sigVs'])

        Vp = array([stats.norm.rvs(0.,sig) for sig in sigVp])
        Vs = array([stats.norm.rvs(0.,sig) for sig in sigVs])

        pars = xx['pars'] + Vp
        sigs = xx['sigs'] + Vs



# ###################### DSGE Model Specification #############################
# Fernandez et.al (May 2012) "Nonlinear Adventures at the Zero Lower Bound" NBER WP 18058; WP 12-10


# ====================== Parameter value setting===============================

        # PARAMETER VALUES
        psi=pars[0]#1.      # Inelastic hours worked
        omic=pars[1]#1.     # Inverse of Flisch labor supply elasticity
        eps=pars[2]#6.      # Elasticity of substitution of intermidiate goods
        theta=pars[3]#0.75  # % of failur to reoptimaze price

        rhor=pars[4]#0.8     # Persistence of nominal interest rate
        phip=pars[5]#1.5    # Inflation coeff in Taylor rule
        phiy=pars[6]#0.25   # Output gap coeff in Taylor rule

        rhob=pars[7]#0.8    # Persistence of discount rate process
        rhoa=pars[8]#0.9    # Persistence of productivity process
        rhog=pars[9]#0.8    # Persistence of goverment spending process

        sigb=sigs[0]#0.0025 # Inovation stdev of discount rate process
        siga=sigs[1]#0.0025 # Inovation stdev of productivity process
        sigm=sigs[2]#0.0025 # Inovation stdev of Taylor rule process (= monetarly shock)
        sigg=sigs[3]#0.0025 # Inovation stdev of govermetn spending process



        eta  = array([
        [0, 0, sigb, 0, 0, 0],
        [0, 0, 0, siga, 0, 0],
        [0, 0, 0, 0, sigm, 0],
        [0, 0, 0, 0, 0, sigg]]).transpose()



        etask= array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]]).transpose()



# ============================ Steady States ==================================

        # Exogeneouse Steady states
        Piss=1.005  # Steady state inflaiton rate = Inflation target
        betass=0.994# Steady state discount rate
        Ass=1.      # Steady state productivity
        Sgss=0.2    # Steady state share of government spending in output
        bndss=0.      # Steady state bond outstanding
        mpss=1.



        # Steady state variables
        Rss=Piss/betass
        Pioss=((1.-theta*(Piss**(eps-1.)))/(1.-theta))**(1./(1.-eps))
        vegass=((1.-theta)/(1-theta*(Piss**eps)))*(Pioss**(-eps))
        x2ss=(1./(1.-Sgss))*(Pioss/(1.-betass*theta*(Piss**(eps-1.))))
        x1ss=((eps-1.)/eps)*x2ss
        mcss=(1.-Sgss)*(1.-betass*theta*(Piss**eps))*x1ss
        wagess=mcss
        lss=((wagess/psi)*(vegass/(1.-Sgss)))**(1./(1.+theta))
        gdpss=lss/vegass
        conss=(1.-Sgss)*gdpss
        govss=Sgss*gdpss



        # Log transport of steady state value

        ##Rss=log(Rss)
        Piss=log(Piss)
        betass=log(betass)
        Ass=log(Ass)
        Sgss=log(Sgss)
        mpss=log(mpss)

        Pioss=log(Pioss)
        vegass=log(vegass)
        x2ss=log(x2ss)
        x1ss=log(x1ss)
        mcss=log(mcss)
        wagess=log(wagess)
        lss=log(lss)
        gdpss=log(gdpss)
        conss=log(conss)
        govss=log(govss)




        #Values for Substituting steady state symbols
        SSval=array([gdpss,conss,lss,mcss,x1ss,x2ss,wagess,Piss,Pioss,vegass,Rss,betass,Ass,mpss,govss,bndss,Sgss,gdpss,conss,lss,mcss,x1ss,x2ss,wagess,Piss,Pioss,vegass,Rss,betass,Ass,mpss,govss,bndss,Sgss]).astype(float64)




# ====================== Symbolic Model Setting ===============================
        #Sym Define variables
        gdp,con,l,mc,x1,x2,wage,Pi,Pio,vega,R,beta,A,mp,gov,bnd,Sg,gdpp,conp,lp,mcp,x1p,x2p,wagep,Pip,Piop,vegap,Rp,betap,Ap,mpp,govp,bndp,Sgp=var('gdp,con,l,mc,x1,x2,wage,Pi,Pio,vega,R,beta,A,mp,gov,bnd,Sg,gdpp,conp,lp,mcp,x1p,x2p,wagep,Pip,Piop,vegap,Rp,betap,Ap,mpp,govp,bndp,Sgp')



        # Model equations
        ## FOC of Household
        f1  = (betap/conp)*(R/Pip)-1./con
        f2  = psi*(l**omic)*con-wage

        ## Profit Maximazation
        f3  = mc - wage/Ap
        f4  = eps*x1 - (eps-1.)*x2
        f5  = 1./con*mc*gdp + theta*(betap*(Pip**eps)*x1p) - x1
        f6  = Pio*(1./con*gdp+theta*(betap*(Pip**(eps-1.))/(Piop)*x2p)) - x2

        ## Goverment policy
        f7  = (Rss**(1.-rhor))*(R**rhor)*((((Pi/Piss)**phip)*((gdp/gdpss)**phiy))**(1.-rhor))*mp - Rp
        f8  = gov - Sg*gdp
        f9  = bnd

        ## Inflation evolution
        f10 = theta*(Pi**(eps-1.)) + (1.-theta)*((Pio)**(1.-eps)) - 1.
        f11 = theta*(Pi**eps)*vega+(1.-theta)*(Pio**(-eps)) - vegap

        ## Market clearing
        f12 = gdp - con - gov
        f13 = gdp - Ap/vegap*l

        ## Stockastic process
        f14 = (log(betap)-log(betass)) - rhob*(log(beta)-log(betass))
        f15 = (log(Ap)-log(Ass)) - rhoa*(log(A)-log(Ass))
        f16 = log(mpp)
        f17 = (log(Sgp)-log(Sgss)) - rhog*(log(Sg)-log(Sgss))


        # Create function f
        f = Matrix([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17])


        # Define the vector of controls, y, and states, x
        x = [vega,R,beta,A,mp,Sg];
        y = [gdp,con,l,mc,x1,x2,wage,Pi,Pio,gov,bnd];
        xp = [vegap,Rp,betap,Ap,mpp,Sgp];
        yp = [gdpp,conp,lp,mcp,x1p,x2p,wagep,Pip,Piop,govp,bndp];



        # Make f a function of the logarithm of the state and control vector for SOME variables
        SSexp={'gdp':exp(gdp),'con':exp(con),'l':exp(l),'mc':exp(mc),'x1':exp(x1),'x2':exp(x2),'wage':exp(wage),'Pi':exp(Pi),'Pio':exp(Pio),'vega':exp(vega),'R':exp(R),'beta':exp(beta),'A':exp(A),'mp':exp(mp),'gov':exp(gov),'bnd':bnd,'Sg':exp(Sg),'gdpp':exp(gdpp),'conp':exp(conp),'lp':exp(lp),'mcp':exp(mcp),'x1p':exp(x1p),'x2p':exp(x2p),'wagep':exp(wagep),'Pip':exp(Pip),'Piop':exp(Piop),'vegap':exp(vegap),'Rp':exp(Rp),'betap':exp(betap),'Ap':exp(Ap),'mpp':exp(mpp),'govp':exp(govp),'bndp':bndp,'Sgp':exp(Sgp)}

        f=f.subs(SSexp)




        # Arguments for fomula evaluation
        args=(gdp,con,l,mc,x1,x2,wage,Pi,Pio,vega,R,beta,A,mp,gov,bnd,Sg,gdpp,conp,lp,mcp,x1p,x2p,wagep,Pip,Piop,vegap,Rp,betap,Ap,mpp,govp,bndp,Sgp)




# ################### Do Not Need to Touch below ##############################
        # This program is a modified version of solab.m by Paul Klein  (JEDC, 2000).
        # http://www.columbia.edu/~mu2166/2nd_order/gx_hx.m


# DNT============ Analytical derivatives and sstate evaluation ================

        nx = len(x);
        ny = len(y);
        nxp = len(xp);
        nyp = len(yp);
        n = len(f);


        # Array of steady state value
        xss=array(lambdify(args,x,'numpy')(*SSval)).reshape(nx,1)
        yss=array(lambdify(args,y,'numpy')(*SSval)).reshape(ny,1)


        # 1st derivatives
        fx = f.jacobian(x)
        nfx=array(lambdify(args,fx,'numpy')(*SSval))

        fxp = f.jacobian(xp)
        nfxp=array(lambdify(args,fxp,'numpy')(*SSval))

        fy = f.jacobian(y)
        nfy=array(lambdify(args,fy,'numpy')(*SSval))

        fyp = f.jacobian(yp)
        nfyp=array(lambdify(args,fyp,'numpy')(*SSval))


        # 2nd derivatives
        nfypyp=array(lambdify(args,[fyp[ii,:].jacobian(yp) for ii in range(n)],'numpy')(*SSval)).reshape(n,nyp,nyp)
        nfypy=array(lambdify(args,[fyp[ii,:].jacobian(y) for ii in range(n)],'numpy')(*SSval)).reshape(n,nyp,ny)
        nfypxp=array(lambdify(args,[fyp[ii,:].jacobian(xp) for ii in range(n)],'numpy')(*SSval)).reshape(n,nyp,nxp)
        nfypx=array(lambdify(args,[fyp[ii,:].jacobian(x) for ii in range(n)],'numpy')(*SSval)).reshape(n,nyp,nx)

        nfyyp=array(lambdify(args,[fy[ii,:].jacobian(yp) for ii in range(n)],'numpy')(*SSval)).reshape(n,ny,nyp)
        nfyy=array(lambdify(args,[fy[ii,:].jacobian(y) for ii in range(n)],'numpy')(*SSval)).reshape(n,ny,ny)
        nfyxp=array(lambdify(args,[fy[ii,:].jacobian(xp) for ii in range(n)],'numpy')(*SSval)).reshape(n,ny,nxp)
        nfyx=array(lambdify(args,[fy[ii,:].jacobian(x) for ii in range(n)],'numpy')(*SSval)).reshape(n,ny,nx)

        nfxpyp=array(lambdify(args,[fxp[ii,:].jacobian(yp) for ii in range(n)],'numpy')(*SSval)).reshape(n,nxp,nyp)
        nfxpy=array(lambdify(args,[fxp[ii,:].jacobian(y) for ii in range(n)],'numpy')(*SSval)).reshape(n,nxp,ny)
        nfxpxp=array(lambdify(args,[fxp[ii,:].jacobian(xp) for ii in range(n)],'numpy')(*SSval)).reshape(n,nxp,nxp)
        nfxpx=array(lambdify(args,[fxp[ii,:].jacobian(x) for ii in range(n)],'numpy')(*SSval)).reshape(n,nxp,nx)

        nfxyp=array(lambdify(args,[fx[ii,:].jacobian(yp) for ii in range(n)],'numpy')(*SSval)).reshape(n,nx,nyp)
        nfxy=array(lambdify(args,[fx[ii,:].jacobian(y) for ii in range(n)],'numpy')(*SSval)).reshape(n,nx,ny)
        nfxxp=array(lambdify(args,[fx[ii,:].jacobian(xp) for ii in range(n)])(*SSval)).reshape(n,nx,nxp)
        nfxx=array(lambdify(args,[fx[ii,:].jacobian(x) for ii in range(n)],'numpy')(*SSval)).reshape(n,nx,nx)




# DNT ==== First-order derivatives of the functions g and h ===================


        stake=1.0

        #Create system matrices A,B
        AA = c_[-nfxp,-nfyp]
        BB = c_[nfx,nfy]
        NK = nfx.shape[1]

        #Complex Schur Decomposition
        ss,tt,qq,zz = linalg.qz(AA,BB);

        #Pick non-explosive (stable) eigenvalues
        slt = (abs(diag(tt))<stake*abs(diag(ss)));
        noslt=logical_not(slt)
        nk=sum(slt);



        # Prep for QZ decomposition- qzswitch()

        def qzswitch(i,ss,tt,qq,zz):
            ssout = ss.copy(); ttout = tt.copy(); qqout = qq.copy(); zzout = zz.copy()
            ix = i-1 # from 1-based to 0-based indexing...

            # use all 1x1-matrices for convenient conjugate-transpose even if real:
            a = mat(ss[ix, ix]); d = mat(tt[ix, ix]); b = mat(ss[ix, ix+1]);
            e = mat(tt[ix, ix+1]); c = mat(ss[ix+1, ix+1]); f = mat(tt[ix+1, ix+1])
            wz = c_[c*e - f*b, (c*d - f*a).H]
            xy = c_[(b*d - e*a).H, (c*d - f*a).H]
            n = sqrt(wz*wz.H)
            m = sqrt(xy*xy.H)

            if n[0,0] == 0: return (ssout, ttout, qqout, zzout)
            wz = solve(n, wz)
            xy = solve(m, xy)
            wz = r_[ wz, c_[-wz[:,1].H, wz[:,0].H]]
            xy = r_[ xy, c_[-xy[:,1].H, xy[:,0].H]]
            ssout[ix:ix+2, :] = xy * ssout[ix:ix+2, :]
            ttout[ix:ix+2, :] = xy * ttout[ix:ix+2, :]
            ssout[:, ix:ix+2] = ssout[:, ix:ix+2] * wz
            ttout[:, ix:ix+2] = ttout[:, ix:ix+2] * wz
            zzout[:, ix:ix+2] = zzout[:, ix:ix+2] * wz
            qqout[ix:ix+2, :] = xy * qqout[ix:ix+2, :]
            return (ssout, ttout, qqout, zzout)


        # Prep for QZ decomposition- qzdiv()

        def qzdiv(stake,ss,tt,qq,zz):
            ssout = ss.copy(); ttout = tt.copy(); qqout = qq.copy(); zzout = zz.copy()
            n,jnk = ss.shape

            # remember diag returns 1d
            root = mat(abs(c_[diag(ss)[:,newaxis], diag(tt)[:,newaxis]]))
            root[:,1] /= where(root[:,0]<1e-13, -root[:,1], root[:,0])
            for i in range(1,n+1)[::-1]:        # always first i rows, decreasing
                m = None
                for j in range(1,i+1)[::-1]:    # search backwards in the first i rows
                    if root[j-1,1] > stake or root[j-1,1] < -0.1:
                        m = j                   # get last relevant row
                        break

                if m == None: return (ssout, ttout, qqout, zzout)

                for k in range(m,i):            # from relev. row to end of first part
                    (ssout, ttout, qqout, zzout) = qzswitch(k, ssout, ttout, qqout, zzout)
                    root[k-1:k+1, 1] = root[k-1:k+1, 1][::-1]

            return (ssout, ttout, qqout, zzout)


        # reordering of generalized eigenvalues with the block inside the unit circle in the upper left
        ss,tt,qq,zz = qzdiv(stake,ss,tt,qq,zz);


        #Split up the results appropriately
        z21 = zz[nk:,:nk]
        z11 = zz[:nk,:nk]



        #Compute the Solution
        z11i = linalg.solve(z11,identity(nk));

        s11 = ss[:nk,:nk]
        t11 = tt[:nk,:nk]

        gx = real(dot(z21,z11i));
        hx = real(dot(z11,(dot(linalg.solve(s11,t11),z11i))));



# DNT ==== Second-order derivatives of the functions g and h by x =============
        nx = hx.shape[0] #rows of hx and hxx
        ny = gx.shape[0] #rows of gx and gxx
        n = nx + ny; #length of f
        ngxx = nx**2*ny; #elements of gxx
        ne = eta.shape[1] #number of exogenous shocks (columns of eta)


        sg = array([ny, nx, nx]); #size of gxx
        sh = array([nx, nx, nx]); #size of hxx

        QQ = zeros((n*nx*nx,n*nx*nx));#zeros((n*nx*(nx+1)/2,n*nx*nx)) #
        q = zeros((n*nx*nx,1));#zeros((n*nx*(nx+1)/2,1))#
        gxx=zeros(sg);
        hxx=zeros(sh);
        GXX=zeros(sg);
        HXX=zeros(sh);

        Qg=zeros((n,ny))
        Qh=zeros((n,nx))
        qs=zeros((n,1))


        m=0
        for i in range(n):#i=0#
            for j in range(nx):#j=0#
                for k in range(nx):#k=0#

                    #First Term
                    q[m,0] = dot(dot(transpose(dot(dot(nfypyp[i,:,:],gx),hx[:,k])+dot(nfypy[i,:,:],gx[:,k])+dot(nfypxp[i,:,:],hx[:,k])+nfypx[i,:,k]),gx),hx[:,j])

                    # Second term
                    GXX[:]=kron(ones((nx**2,1)),nfyp[i,:]).reshape(sg)
                    GXX[:]=GXX[:]*kron(ones((nx*ny,1)),hx[:,k]).reshape(sg)
                    GXX[:]=GXX[:]*transpose(kron(ones((nx*ny,1)),hx[:,j])).reshape(sg)

                    QQ[m,:ngxx]=GXX.flatten()
                    GXX=0*GXX


                    # Third term
                    HXX[:,j,k]=dot(nfyp[i,:],gx)

                    QQ[m,ngxx:]=HXX[:].flatten()
                    HXX= 0*HXX


                    # Fourth Term
                    q[m,0] = q[m,0] + dot(transpose(dot(dot(nfyyp[i,:,:],gx),hx[:,k]) +  dot(nfyy[i,:,:],gx[:,k]) + dot(nfyxp[i,:,:],hx[:,k]) + nfyx[i,:,k]),gx[:,j])


                    # Fifth Term
                    GXX[:,j,k]=nfy[i,:].transpose()

                    QQ[m,:ngxx] = QQ[m,:ngxx] + GXX.flatten()
                    GXX = 0*GXX

                    # Sixth term
                    q[m,0] = q[m,0] + dot(transpose(dot(dot(nfxpyp[i,:,:],gx),hx[:,k]) + dot(nfxpy[i,:,:],gx[:,k]) + dot(nfxpxp[i,:,:],hx[:,k]) + nfxpx[i,:,k].transpose()),hx[:,j]);


                    # Seventh Term
                    HXX[:,j,k]=nfxp[i,:].transpose();

                    QQ[m,ngxx:] = QQ[m,ngxx:] + HXX[:].flatten();
                    HXX = 0*HXX;


                    # Eighth Term
                    q[m,0] = q[m,0] + dot(dot(nfxyp[i,j,:],gx),hx[:,k]) +  dot(nfxy[i,j,:],gx[:,k]) +  dot(nfxxp[i,j,:],hx[:,k]) + nfxx[i,j,k]

                    m+=1





        xsol = -linalg.solve(QQ,q)

        gxx[:]= xsol[:ngxx].reshape(sg);
        hxx[:]= xsol[ngxx:].reshape(sh);




# DNT ==== Second-order derivatives of the functions g and h by sig ===========

        i=0 # reset
        for i in range(n):

            #First Term
            Qh[i,:] = dot(nfyp[i,:],gx);


            #Second Term
            qs[i,0] = sum(diag(dot(dot(transpose(dot(dot(nfypyp[i,:,:],gx),eta)),gx),eta)));


            #Third Term
            qs[i,0] = qs[i,0] + sum(diag(dot(dot(transpose(dot(nfypxp[i,:,:],eta)),gx),eta)));


            #Fourth Term
            ans=dot(nfyp[i,:],gxx.reshape(ny,nx**2))
            qs[i,0] =  qs[i,0] + sum(diag(dot(transpose(dot(ans.reshape(nx,nx),eta)),eta)));


            #Fifth Term
            Qg[i,:] = nfyp[i,:]


            #Sixth Term
            Qg[i,:] = Qg[i,:] + nfy[i,:]


            #Seventh Term
            Qh[i,:] = Qh[i,:] + nfxp[i,:]


            #Eighth Term
            qs[i,0] = qs[i,0] + sum(diag(dot(transpose(dot(dot(nfxpyp[i,:,:],gx),eta)),eta)));

            #Nineth Term
            qs[i,0] = qs[i,0] + sum(diag(dot(transpose(dot(nfxpxp[i,:,:],eta)),eta)));



        xssol=-linalg.solve(c_[Qg,Qh],qs)

        gss = xssol[:ny]
        hss = xssol[ny:]



# ------------------------- End of solve proc ---------------------------------







# ========== COMPUTE THEORETICAL MOMENTS FOR THE SECOND-ORDER SOLUTION ========
        varshock = dot(eta,eta.transpose())


        # All moments of x
        meanx = zeros((nx,1));
        varx = zeros((nx,nx));
        Qt = zeros((nx+nx**2,1));

        C1 = 0.5*hss;
        F11 = hx;
        F12 = 0.5*kron(1,hxx);

        C2 = varshock[:];
        F21 = zeros((nx**2,nx));
        F22 = kron(hx,hx);

        F = r_[c_[F11,F12.reshape(F12.shape[0],F12.shape[0]**2)],c_[F21,F22]]
        C = r_[C1,C2.reshape(C2.shape[0]**2,1)]

        Qt = linalg.solve((identity(F.shape[0])-F),C)
        meanx[:]= Qt[:nx];
        varx[:] = Qt[nx:nx+nx**2].reshape(nx,nx)


        # Variance-covariance matrix of y
        vary = zeros((ny,ny));
        vary[:,:]=dot(kron(gx,gx),varx.flatten()).reshape((ny,ny))


        # Covariance of y and x
        varyx = dot(gx,varx)


        # Unconditional mean of y
        meany = zeros((ny,1));
        meany = meany + dot(gx,meanx) + dot((0.5*kron(1,gxx)).reshape(ny,nx**2),varx.reshape(nx**2,1)) + 0.5*gss;


        # First-order autocovariance of x
        autox = dot(hx,varx)

        # First-order autocovariance of y
        autoy = dot(dot(gx,autox),gx.transpose())




# ======================== Simulation proc  ===================================

        ev=stats.norm.rvs(size=ll)

        eta  = array([
        [0, 0, ev[0], 0, 0, 0],
        [0, 0, 0, ev[1], 0, 0],
        [0, 0, 0, 0, ev[2], 0],
        [0, 0, 0, 0, 0, ev[3]]]).transpose()

        pathe = array([sigb,siga,sigm,sigg])
        eta1 = dot(eta,pathe)
        eta2 = dot(dot(dot(eta,pathe).reshape(nx,1),pathe.reshape(1,ne)),eta.transpose());


        pathx1p = xx['x1'] #meanx.flatten()
        pathx2p = xx['x2'] #varx.flatten()


        # /// Calcurate x ///
        C1 = 0.5*hss;
        F11 = hx;
        F12 = 0.5*kron(1,hxx);

        pathx1=(C1.flatten() + dot(F11,pathx1p) + dot(F12.reshape(nx,nx**2),pathx2p)+eta1.flatten())

        C2 = varshock[:];
        F21 = zeros((nx**2,nx));
        F22 = kron(hx,hx);
        pathx2 = C2.flatten() + dot(F21,pathx1p) + dot(F22,pathx2p) + eta2.flatten()


        # /// Calcurate y ///
        CG = (0.5)*gss;
        FG1 = gx;
        FG2 = (0.5)*kron(1,gxx);

        pathy = CG.flatten() + dot(FG1,pathx1) + dot(FG2.reshape(ny,nx**2),pathx2)





# ============================= Keep Next xx ==================================
        values = [
            pathx1,
            pathx2,
            pars,
            sigs,
            sigVp,
            sigVs
        ]

        xf = dict(zip(state_keys, values))
        Xf.append(xf)


        # /// Residuals ///
        w = YY - pathy[obsYflag]
        oth.append(pathy)



# ============================= Resampleing ===================================


        # evaluate particle likelihood
        pl_xi=1.
        pl.append(array([log(1./sigw[i]*((1+(pl_xi*(abs(w[i])-0.))/sigw[i])**(-1/pl_xi-1))) for i in range(ll)]).sum())




    # /// Relative particle likelihood ///
    pl=array(pl).astype('float')
    plpath.append(pl.mean())
    dpl=np.exp((pl-max(pl)))
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






    t+=1
    print str(round(float(t-start)/float(TT-start)*100,2))+'%'

    ## End of t iteration


# Total log-likelihood
LOGLIK=sum(plpath)


Yhat = pd.DataFrame(array(othpath).T[obsYflag].T,index=DD.index[start:],columns=DD.columns)



# pd.Series([Xhat[t]['pars'][0] for t in range(len(Xhat))],index=pd.period_range(DD.index[start],periods=len(Xhat))).plot()


# Yhat['con'].plot();DD['con'][start:].plot()




















