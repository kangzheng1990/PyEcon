# -*- coding:utf-8 -*-


from TVP import *
import PFilter as PF
"""
必要なモジュールをimportしています。TVPというディレクトリの__init__.pyでは必要な外部モジュールと自作した計算を便利にする関数群をimportしています。PFilter.pyは粒子フィルターを実行する関数だけを含んだモジュールです。
"""



# Config & initialization ###################################################


# /// Data ///
"""
まずはいつも通り、Pandas.DataFrame形式でデータを読み込みます（ここではExcelから）。サンプル期間数（TT）と扱う変数の数（KK）を記録しておくと後々使いまわせて便利です。
"""


url=r"TVP.xlsx"

EXL=pd.ExcelFile(url)
Data = EXL.parse('Go',index_col="TIME")
DD=Data.copy()

TT,KK=DD.shape



# /// Number of particles ///
"""
粒子フィルターは、未知の状態変数の分布を離散表現し、データ適合度が高まるように学習的に変形（リサンプリング）していくものです。離散化に用いる粒子の数が多ければ多いほど柔軟に変形できますので、計算時間と適合度とのバランスを見ながら粒子数を決めていきます。
"""

NN = 5000


# /// Time iter Config ///
"""
状態空間モデルでラグ付きのデータを用いる場合、その分、データのサンプル期間が減りますから、フィルターの計算期間も対応して減らす必要があります。
"""

Lag = 1 # Maximum lag used in Obs or Exo function
start=Lag+1  #TT-1#
t=start
T=TT-Lag



# /// Std of Obs Vars to be used to calc loglik
"""
SVARモデルの内生変数（＝観測データ）を指定します。また、尤度（適合度）計算に使用する観測データの標本標準偏差も計算しておきます。
"""
obslist= ['Pi','LR','GAP']
sigw=std(DD[obslist].values,axis=0)
ll=len(obslist)





###########################################################################
#
#                Constant params SVAR for initialization
#
# ##########################################################################
"""
未知状態としての時変パラメターの初期分布として、固定パラメターモデルの推定値を使用します。そのために、Statsmodelsを使ってSVARを推定しています。詳しくはStatsmodelsのドキュメントを参照。
"""


# /// SVAR /// #
"""
構造ショックの識別制約を設定。最も単純なコレスキー分解（下三角行列A）を採用します。
"""
svar_type="A"
A=tril(ones(ll*ll).reshape(ll,ll)*2).astype('S1')
A[A=='2']="E"; fill_diagonal(A,1)
B=None

model = tsa.SVAR(DD, svar_type=svar_type, A=A,B=B)
rlt= model.fit(maxlags=1,trend="c")

# IRF #
IRF=rlt.irf(12)
##IRF.plot();plt.show()


# /// Ericiting initials for TVP /// #
a_0 = rlt.A[tril_indices(ll,-1)]
gam_0 = rlt.params.flatten()
gam_s = rlt.cov_params.diagonal()
lnh_0=log(diag(linalg.cholesky(rlt.sigma_u,lower=True))**2)




###########################################################################
#
#                      TVP Initialization Proc
#
# ##########################################################################


# /// StateSpace construction ///
"""
TVP-SVARの行列表記を念頭にパラメター行列の名前を定義し、その個数を記録しておきます。
"""
state_keys = ['a', 'gam', 'lnh', 'sgVa', 'sgVg', 'sgVh']

Nstate = len(state_keys)



# /// Set parameters of initial dst  ///
"""
それぞれの状態変数の初期分布をNN個サンプリングし、DictionaryのリストXとしてまとめます。
"""

a_0,   a_s   = (a_0,   0.05*ones(ll))
gam_0, gam_s = (gam_0, gam_s)
lnh_0, lnh_s = (lnh_0, 0.05*ones(ll))

sgVa_0, sgVa_s   =  (0.001*ones(ll), 0.01*ones(ll))
sgVg_0, sgVg_s   =  (0.005*ones(ll**2+ll), 0.05*ones(ll**2+ll))
sgVh_0, sgVh_s   =  (0.005*ones(ll), 0.05*ones(ll))



# /// Int dst of State Vars ///
init_dst=[]
init_dst.append( array([stats.norm.rvs(a_0[i],a_s[i],NN) for i in range(ll)]).T  )
init_dst.append( array([stats.norm.rvs(gam_0[i],gam_s[i],NN) for i in range(ll**2+ll)]).T )
init_dst.append( array([stats.norm.rvs(lnh_0[i],lnh_s[i],NN) for i in range(ll)]).T )

init_dst.append( array([stats.norm.rvs(sgVa_0[i],sgVa_s[i],NN) for i in range(ll)]).T  )
init_dst.append( array([stats.norm.rvs(sgVg_0[i],sgVg_s[i],NN) for i in range(ll**2+ll)]).T )
init_dst.append( array([stats.norm.rvs(sgVh_0[i],sgVh_s[i],NN) for i in range(ll)]).T )


X = [
    dict(zip(
        state_keys,

        [init[i] for init in init_dst]

    )) for i in range(NN)
]







###########################################################################
#
#                              Estimation Proc
#
##########################################################################

# Model Equation #########################################################
# /// fitted observation function ///

"""
状態推移式によって1期進められた状態変数のDictionaryと観測データとを受けとり、SVARをフィットさせて観測誤差を出力する関数（ww）を定義します。
"""

def ww(xf,DD,t):

    ## Endo vars, install
    Y = DD[obslist].values[t]

    ## Exog vars, install
    Yp = DD[obslist].values[t-1]


    ## State vars, install
    a   = xf['a']
    gam = xf['gam']
    lnh = xf['lnh']


    ## Observation equations
    sigy = exp(lnh/2.)
    iHsqt = linalg.inv(diag(sigy))

    A = identity(ll); A[tril_indices(ll,-1)] = a
    Ahat = dot(A,iHsqt)

    G0 = gam[:ll]
    G1 = gam[ll:].reshape(ll,ll)

    A0 = dot(Ahat,G0)
    A1 = dot(Ahat,G1)


    ## Observation equations
    resid = dot(Ahat,Y) - ( A0 + dot(A1,Yp) ) # This is SVAR model
    fitted = G0 + dot(G1,Yp) # Reduced VAR model


    ## Other variables to return
    oth_rlt=fitted


    ## Output
    return resid,oth_rlt




# /// models for system function ///
"""
未知状態（＝パラメター）のダイナミクスを指定する関数（Dyn）を定義します。現在の状態を引数で受けて、1期先の状態の予測を返します。状態変数の次元が揃っており、Dictionaryのkeyが正確にセットされていることが必要です。
"""

def Dyn(x,keys=state_keys):

    ## Prior, install
    sgVa = abs(x['sgVa'])
    sgVg = abs(x['sgVg'])
    sgVh = abs(x['sgVh'])


    ## System Inovations
    Va = array([stats.norm.rvs(0.,sig) for sig in sgVa])
    Vg = array([stats.norm.rvs(0.,sig) for sig in sgVg])
    Vh = array([stats.norm.rvs(0.,sig) for sig in sgVh])


    ## System equations
    a   = x['a'] + Va
    gam = x['gam'] + Vg
    lnh = x['lnh'] + Vh


    ## Output
    values = [
        a,
        gam,
        lnh,
        sgVa,
        sgVg,
        sgVh
    ]

    xf = dict(zip(keys, values))
    return xf





# Execute PFilter ######################################################
"""
PFilterモジュールを使って、粒子フィルターによって推定した未知状態（＝パラメター）の時系列を出力します。引数は上で定義したDD,X,ww,Dynです。
"""

X, Xhat, Xstd, resid, othpath, LOGLIK = PF.PFilter_D(DD,X,ww,Dyn, sigw=sigw, start=start)





# Output checking ######################################################
"""
適宜、結果を可視化するセクションです。
"""

fitted=array(othpath)
YY=DD[obslist].values

a_r=array([Xhat[t]['a'] for t in range(TT-start)])
gam_r=array([Xhat[t]['gam'] for t in range(TT-start)])
lnh_r=array([Xhat[t]['lnh'] for t in range(TT-start)])

# col=0;pd.DataFrame(c_[fitted[:,col],YY[start:,col]],columns=['Fit','Act'],index=DD.index[start:]).plot()


# pd.DataFrame(-1*a_r[:,0],columns=['Pi<-GAP'],index=DD.index[start:]).plot()

# pd.DataFrame(lnh_r,index=DD.index[start:]).plot()







