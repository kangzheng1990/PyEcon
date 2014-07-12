# -*- coding: utf-8 -*-

"""
PythonからRを使おう：ウェーブレット解析

データ分析をしていると、たまにつぎのようなシチュエーションがあります。
1）既に普及している技術、2）自分で書くのは大変、3）Pythonには既存のモジュールがない。

こういうときは、Rに頼りましょう。ただし、Pythonから。

PythonにはRpy2というRのインターフェイスがあり、Rが導入された環境でPythonからRの関数を使えるようにできます。

今回は、このRpy2を使って、「ウェーブレット解析」を行なってみました。Full Codeはこちら。


■　RとRpy2のセットアップ

まずは使用する環境（PC）にRがインストールされている必要があります。こちらの手順に従って最新版を入れておけば大丈夫だと思います(http://www.okada.jp.org/RWiki/?R%20%A4%CE%A5%A4%A5%F3%A5%B9%A5%C8%A1%BC%A5%EB)。


また、今回は「ウェーブレット解析」を行なうので、そのための拡張パッケージをRに追加します。手順はいくつかありますが、最も単純なのがRのGUIのツールバーからブラウズする方法。Packages>>Install Package>>Japan(Tokyo)>>waveletsとすればOKです。

続いて、Rpy2のセットアップです。PythonモジュールであるRpy2は、Anacondaには含まれていません。新たに追加するには、コマンドでpip install rpy2　とすれば通常はOKです。

ただし、環境によっては上手くインストールできないこともあるようす。特に、Windowsでは、こちらで配布されているインストーラー（http://www.lfd.uci.edu/~gohlke/pythonlibs/#rpy2）を使った方が確実だと思います。


PythonにRのインストールされた場所を認識してもらうため、環境変数PATH、R_HOME、R_USERの設定が必要です。環境変数の設定方法についてはお使いのOSに合わせて適宜Googleなどで検索して頂ければ分かると思います。

PATHという環境変数は元からあると思うので、そこにRのR.dllファイルがあるアドレスを加えておきます（例えば以下のように）。
C:\Program Files\R\R-3.0.2\bin\i386

また、R_HOMEとR_USERには共に、Rのルートディレクトリを設定しておきます（例えば以下のように）。
C:\Program Files\R\R-3.0.2\

以上で、RとRpy2のセットアップは完了。Pythonから以下のようにRpy2をimportしてエラーが出なければOKです。

import rpy2.robjects as robjects




■　外部モジュールとデータの準備
それでは、実際に、PythonからRを使ってウェーブレット解析を実行していきます。


まずはいつも通り、外部もモジュールのimportから。今回必要なのは以下の通りです。

[python]"""

from scipy import *
import pandas as pd
import pandas.io.data as web

import pandas.rpy.common as com
import rpy2.robjects as robjects
r=robjects.r
r.library("wavelets")

from pylab import *

"""[/python]

次はデータのimportです。今回は、pandasの機能を使って（前回参照）Yahoo!Financeから日経平均株価の時系列をダウンロードして使います。

ウェーブレット解析には、月次のリターン系列を使うため、適宜変換を行なっておきます。


[python]"""

# /// Data instalation /// --------------------------------------------
DD=web.get_data_yahoo('^N225','1980') # Nikkei225 from Yahoo!Finance
DD=DD.resample('M',how='last')
Ret = DD[['Adj Close']].pct_change()[1:]

"""[/python]




■　データ型をRのDataFrameに

続いて、データをRで使えるように型変換します。現状はPandasのDataFrameになっていますので、以下のように、これをRのDataFrameに変えます。

また、今回は冗長ですが、1系列だけについてウェーブレット解析を行なうため、それを取り出してXとしています。

[python]"""

# Convert to R-DataFrame
RDD=com.convert_to_r_dataframe(Ret)
TT=RDD.nrow
X=r.ts(RDD[0])

"""[/python]




■　Rのウェーブレット解析を使う

いよいよRのウェーブレット解析パッケージの出番です。
まず、そもそも、「ウェーブレットとは何か」という問いにはこちらのペーパーにお任せします（http://www.imes.boj.or.jp/research/papers/japanese/kk23-1-1.pdf）。

今回、具体的にやることは、「1本の時系列データを複数の周波数成分に寄与度解する」ということです。これは、「多重解像度分析」と呼ばれることも多いです。

これによって、現在の上昇／下落局面が、何ヶ月程度の周期成分によるものかを理解できます。それが分かれば、「現在の局面が後どれくらい続くのか」という見通しを立てるのに役立つでしょう。

ここでは、原系列から、もっとも周期の短い波（レベル1）から順に、レベル5までの周期成分（ウェーブレット・ディテール）を抜き出して行きます。そして、最後に残った、レベル5より長い周期が全て含まれる残差をレベル5のウェーブレット・スムースと呼びます。

月次データを使用していますので、それぞれ、レベル1が2ヶ月、レベル2が4ヶ月、レベル3が8ヶ月、レベル4が16ヶ月、レベル5が32ヶ月の周期、となります。

なお、分解に使うアルゴリズムは、時系列サンプルの数を保存できるMODWT（maximal overlap discrete wavelet transform）という方法を採っています。詳しくは前述のペーパーをご参照下さい。

ちなみに、以下のコードでは、R関数の返り値を逐一array()に変換することで、Rの関数をr.xxxxの形で呼び出す以外は、通常のPythonの記法で済ませています。

[python]"""

# /// Univariate MODWT  /// ------------------------------------------
wX=r.modwt(X,boundary = "reflection") #r.dwt(X) #  haar, D4, D8 D12 Wavelet
wX=r.align(wX) # Keep Time series align

level= wX.do_slot("level")[0] # optimal level selection

# Wavelet transformation 
W=array([array(wX.do_slot("W")[i])[0:TT].reshape(TT,) for i in range(level)])
V=array([array(wX.do_slot("V")[i])[0:TT].reshape(TT,) for i in range(level)])


# /// Multi-resolution analysis /// ----------------------------------
mrX=r.mra(X,boundary = "reflection",method="modwt",**{"n.level":level})

# Getting wavelet details and smooth
D=array([array(mrX.do_slot("D")[i])[0:TT].reshape(TT,) for i in range(level)])
S=array([array(mrX.do_slot("S")[i])[0:TT].reshape(TT,) for i in range(level)])

"""[/python]

以上で、レベル1～5のウェーブレット・ディテールDと、ウェーブレット・スムースSが計算できました。最後にこれをPlotして可視化しておきましょう。


■　結果の可視化と解釈

一番上の緑色が原系列で、以下レベル1～5のディテール。最後が、レベル5のスムース（残差としてのトレンド）です。

最近の日経平均の月次リターンは6月+3.6%と、5月+2.3%と、4月-3.5%と比べて上向いていましたが、この傾向はレベル2の4ヶ月周期成分とレベル3の8ヶ月周期成分の寄与が大きかったようです。4ヶ月周期成分は2ヶ月で折り返しますので、こちらの寄与は7月以降低下するでしょう。一方、8ヶ月っ周期成分は4ヶ月で折り返しますが、これまでちょうど4ヶ月連続で上向いてきていたため、7月からは低下に転じそうです。

総じて考えると、7月以降、日経平均のリターンは低下していくことが見込まれるでしょう。

[python]"""
# /// Plotting /// ----------------------------------------------------
cyc=[2**(i+1) for i in range(level)]
X=array(X)

subplot(711);plot(range(TT),X,color='Green');subplot(712);plot(range(TT),D[0,:]);subplot(713);plot(range(TT),D[1,:]);subplot(714);plot(range(TT),D[2,:]);subplot(715);plot(range(TT),D[3,:]);subplot(716);plot(range(TT),D[4,:]);subplot(717);plot(range(TT),S[4,:]);show()


"""[/python]
"""

