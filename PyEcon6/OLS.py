# -*- coding: utf-8 -*-

import numpy as np
from six.moves import zip
from bokeh.plotting import *

"""
Statsmodelsで回帰分析のテンプレート

Anacondaをインストールし、Pandasでデータの出し入れができたら、すぐにStasmodelsという統計解析パッケージが使えます。

オープンソースの統計解析パッケージと言えばRがありますが、StatsmodelsはRに比べると機能（パッケージ）が少なく、もの足りなく感じる人もいるかもしれません。

でも、サクサクと短時間でトライ＆エラーしながらデータ分析をする時は、比較的シンプルな統計解析を多用するでしょうし、そういう機能であればStatsmodelsで十分カバーされていると思います。

また、本格的に、まだパッケージ化されていないような分析を自分で書くなら、PythonはRなどよりも効率的だと思います（前回はその一例として、最新の時系列分析を実装するコードを紹介しました）。

そういう意味では、Rはやや中途半端な感じになってしまったように思えます。経済、金融、心理、医療、生物、工学など、各分野の専門的なパッケージがたくさん備わっていますが、どれも「最先端」とまではいきません。

最近はデータ分析をする人がRからPythonに移行しているという話をよく耳にしますが、納得できる面も多いです。Statsmodelsがそのトレンドの火付け役になった感もあるでしょう。

今回は、このStatsmodelsを使って、単純だけども結局よく使う、回帰分析のテンプレートを用意してみました。
"""



"""
■　セットアップ

まずは必要なモジュールをimportします。全てAnacondaに含まれていますので、新たにインストールする必要はありません。
"""

# /// Setup /// -----------------------------------------------------


from scipy import *

import pandas as pd
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import autocorrelation_plot

import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as oti
import statsmodels.tsa.api as tsa



"""
データのimportです。今回はCSV形式のデータをpandasのDataFrameとして読み込んでいます。また、サンプル期間を2000年以降に限っています。DataFrameのindex_colをTime Label（年月日形式）にしておくと、こうした直感的なサンプル分割ができるので便利です。
"""



# Data Installation
url=r"Data.csv"
DD=pd.DataFrame.from_csv(url)

#EXL=pd.ExcelFile(url)
#DD=EXL.parse('Go', index_col='TIME')




# Sellecting period
DD=DD['2000':] # Time Slice




"""
今回は、ラグ付き変数を回帰分析で使いたいと思います。以下の関数は、データフレームの全変数のラグ変数を作成し、既存のデータフレームの右端に追加するものです。今回は、1期ラグまでラグ変数を追加しました。
"""

# Adding Lagged variables
def lag_join(DD,nlag=1,NaN_treat='delete'):
    """
    join the lagged data to the right-end
    DD: pandas-DataFrame
    nlag: number of lag
    NaN_treat: missing value treatment
      None: do nothing
      delete: deleting the first nlag'th rows
      fill: filling backward
      rowmean: filling by fowmean
    """
    LD=DD.copy()
    for i in range(1,nlag+1):
        LD=LD.join(LD.shift(i),rsuffix="_"+str(i))

    # NaN treat
    if NaN_treat=="delete":
        LD=LD[nlag:]

    elif NaN_treat=="fill":
        LD=LD.bfill()

    elif NaN_treat=="rowmean":
        LD=LD.fillna(LD.mean())

    return LD

LD=lag_join(DD,1) # Create DataFrame with lagged variables

TT=LD.shape[0] #Keep sample length






"""
■　回帰分析の実行

データフレームの準備ができたので、回帰分析に入ります。


実際に回帰式を組み立てる前に、データの相関関係を確認しておいたほうがよいでしょう。Pandas組み込みの散布図マトリックス・プロットが便利です。
"""





# /// OLS /// -----------------------------------------------------

# Setting variable list, the 1st var should be endogeneouse variable
varlst=['Pic','Pic_1','PIM','GAP']

# scatter plot matrix
scatter_matrix(LD[varlst], diagonal='kde')


"""
次の関数は上で設定したvarlstをもとに、回帰式の文字データを作成するものです。直接書けばよいのですが、こうして変数名のlistから式を生成できるようにしておくと何かと便利なことがあります。
"""


def fml_build(varlst):
    """
    Binding OLS formula from a list of variable names
      varlst: variable names, the 1st var should be endogeneouse variable
    """
    varlst.reverse()
    fml=varlst.pop()+'~'
    while len(varlst)>0:
        fml=fml+'+'+varlst.pop()
    return fml

eq = fml_build(varlst)


"""
実際にOLSで線形回帰式を当てはめるには、（smfとしてimportした）statsmodels.formula.apiのolsというメソッドに回帰式とデータを入力し、fit()メソッドを使えば完了です。
"""
rlt=smf.ols(eq, data=LD).fit()




"""
■　線形回帰式の当てはめ

OLSの推定結果であるrltに様々な関数やメソッドを適応していくことで、そのモデルの信頼性を診断していきます。まずは、.summary()メソッドで主要な結果の一覧表を表示しましょう。
"""



# /// Diagnosis /// -----------------------------------------------------

rlt.summary() # show summary


"""
回帰係数とt値、決定係数、AICなど、最低限必要な結果をまとめてくれています。また、これらの数値はrltのアトリビュートになっていますので、rlt.paramsなどとすれば呼びだせます。



■　推計結果の診断：　系列相関と多重共線性

Summaryで特筆すべきはDurbin-Watson比率です。これが2よりも十分に大きい時は負の系列相関。2より十分小さいときには正の系列相関が疑われます。経済時系列データを使った回帰分析では、系列相関が頻繁に生じますから、特に注意が必要です。

ちなみに、系列相関をはじめ、古典的な回帰モデルの診断手続きは経済企画庁[1988]が詳しいです。だいぶ昔のレポートですが、線形回帰モデルは古典的な方法ですので、その基本は変わっていません。http://www.esri.go.jp/jp/archive/bun/bun112/bun112a.pdf

誤差項に系列相関が残っている場合、トレンドも含めて、モデルに含まれていない要因が大きい影響を持っている可能性がありますので、思い当たる説明変数を加えてみたり、タイム・トレンドやラグ項を足したり、変分を取るなりして、コントロールしたほうがよいでしょう。

このような系列相関のチェックには、ADF検定によって誤差項の定常性を確認するのも有効だと思います。

"""




# ADF test, H0: Non-stationary
tsa.adfuller(rlt.resid,regression='nc')


# Autocorrel plot of resid
autocorrelation_plot(rlt.resid) # Show ACF of residuals
ACF_resid=tsa.acf(rlt.resid) # Keep ACF of residuals

"""
誤差項が定常であれば、モデル内の説明変数と被説明変数との間に安定した（一時的に外れても帰ってくるような）関係があることが保証されます。また、多くの経済変数はそもそも非定常ですので、残差が定常の場合、重要な要因がモデルから脱落している可能性も低くなります。

系列相関以外に大切なのは、多重共線性（マルチコリニアリティ）のチェックでしょう。これは、説明変数の間に強い相関がある場合に生じるもので、推定される係数の符号が反転してしまったりしますので厄介です。

以下のようにVIF統計量を計算して、10を大きく上回っていなければ、ひとまず安心と考えます。また、VIFを参照して機械的に判定しなくても、想定される符号と逆の符号を持った説明変数が現れれば、経験的にマルチコに気づくと思います。もっとも、マルチコの解決策は強相関している説明変数のどれかを取り除くくらいしか解決策がありません。

リッジ回帰など、パラメター空間を制約するやり方はそもそもパラメターの不偏性を犠牲にする上に、必ずしもマルチコを解消させる保障がないため、歪めますので、計量経済学では推奨されていません。

"""


# Checking Multicolinearity by VIF
VIF=pd.DataFrame([oti.variance_inflation_factor(rlt.model.exog,i) for i in  range(1,rlt.model.exog.shape[1])],index=rlt.model.exog_names[1:],columns=['VIF']) # VIF>10 should be cared



"""
■　トライ＆エラーを補助してくれる可視化ツール

回帰分析にはトライ＆エラーが付き物です。むしろ、ほとんどの経済現象は、線形式で完全に記述できるはずがありませんから、色々な回帰式を当てはめたり、サンプル期間変えたりして初めて、経済現象の全体像を掴むことができるのだと思います。

statsmodelsには、こうしたトライ＆エラーをサポートする可視化ツールが付いています。以下のモジュールをimportしてみましょう。

"""


# /// Graphical Diagnostic Tools /// ---------------------------------
import statsmodels.graphics.regressionplots as regplot


"""
まず、以下のinfluence_plot()は、サンプルの中にはずれ値的な動きをした期間があるかを検出してくれます。
"""
# Checking Outlier effect
regplot.influence_plot(rlt) # Studentized Residual
regplot.plot_leverage_resid2(rlt) # Leverage vs. resid^2


"""
また、plot_regress_exog()は、個別の説明変数ごとに、誤差項と説明変数の関係や、他の要因をコントロールした上での当該変数の説明力を見る偏回帰プロットが表示されます。誤差や説明力が説明変数の値に連動して変化するなら、他変数からの影響や非線形性が現れていると考えられます。
"""

# Selected exog vs. other things controlled endog plot
regplot.plot_regress_exog(rlt,1)




N = DD.shape[0]

x = DD['Pic'].values*100
y = DD['GAP'].values*100


radii = np.random.random(size=N)/10
colors = ["#%02x%02x%02x" % (r, g, 150) for r, g in zip(np.floor(50+2*x), np.floor(30+2*y))]

output_file("color_scatter.html", title="color_scatter.py example")

scatter(x,y, radius=radii, radius_units="data",
       fill_color=colors, fill_alpha=0.6,
        line_color=None, Name="color_scatter_example")

show()  # open a browser

