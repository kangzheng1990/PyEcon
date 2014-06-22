# -*- coding: utf-8 -*-

"""
Pandasでデータハンドリング：便利機能まとめ20選

マクロ経済データを使って統計解析をするには、前処理や後処理が欠かせません。今回は、そんな補助的なデータ処理（データハンドリング）に役立つPandasの機能を10個ピックアップして紹介します。なお、ここではエコノミストが頻繁に扱うマクロ経済の時系列データを念頭に選びました。

■　入力
1) Excelからデータをimport
2) Yahoo!Financeからデータをimport
3) GithubのGistからデータをimport

■　時間処理
4) Period-Indexを利用
5) サンプル期間の分割（スライス）
6) データ頻度の低解像化（集計）
7) データ頻度の高解像化（補完）

■　データ値の処理
8) 欠損値処理
9) 条件選択でサブデータ作成
10) 列の追加
11) 行の追加
12) データ値の置き換え（上書き）
13) ソート

■　系列の変換
14) 記述統計の表示
15) 変化率の計算
16) 指数化
17) 対数変換
18) 四分位範囲で正規化
19) ラグ変数の追加
20) 移動平均の計算


まず、pandasをpdとしてimportしておきます。Scipyもよく使うので丸ごとimportします。
[python]
"""
import pandas as pd
from scipy import *

"""
[/python]


■　入力
1) Excelからデータをimport
・　単純にData.xlsxのsheet1を読み込む。
[python]
"""
URL=r"Data.xlsx" # Set a path
EXL=pd.ExcelFile(URL) # Open
Data=EXL.parse("sheet1") # Parse
"""
[/python]
・　シートの１行目を除いて読み込み、年月日の入ったTIMEという列をIndexとして使う。
[python]
"""
Data=EXL.parse("sheet1",skiprows=1,index_col="TIME") # Parse
"""
[/python]

・　B列～F列だけを切抜いて読み込む。
[python]
"""
Data=EXL.parse("sheet1",parse_cols="B:F") # Parse

"""
[/python]




2) Yahoo!Financeからデータをimport
・　日経平均株価(コード：^N225)の日次時系列を1984年から取得。
[python]
"""
import pandas.io.data as web # import I/O tool of pandas
Data=web.get_data_yahoo('^N225','1984') # Yahoo!Finance parser of pandas
"""
[/python]


3) GithubのGistからデータをimport

・　GistにData.csvがアップロードされているとして読み込み。単にGistのページ左下の"Link to this gist"に表示されたURLを現在位置<gist-URL>としてData.csvを読み込めばOK。
[python]
"""
URL=r"<gist-URL>/Data.csv"
Data=pd.DataFrame.from_csv(URL)
"""
[/python]


■　時間処理
4) Period-Indexを利用
・　TimestampデータをPeriodデータに変換する。年月日データの列"TIME"をindex_colとして読み込むと、Timestampインデックスが生成される。これはデータの観察周期が一様でない時にも使える汎用的なラベリングで、若干使い勝手が悪い。年次、四半期、月次、など観察周期が決まっている時はPeriodインデックスの方が便利。

[python]
"""
URL=r"Data.xlsx" # Set a path
EXL=pd.ExcelFile(URL) # Open
Data=EXL.parse("sheet1",index_col="TIME") # Parse
Data=Data.to_period(freq="M") # change timestamp to period; e.g. "M": Montly, "Q": quarterly, "A": annual
"""
[/python]


5) サンプル期間の分割（スライス）
・　2000年を境にデータを前半と後半に分割する。
[python]
"""
Data_1H = Data[:"2000"]
Data_2H = Data["2001":]
"""
[/python]

6) データ頻度の低解像化（集計）
・　月次データを四半期データに変換。変換方法は「期間平均」。

[python]
"""
Data=Data.resample('Q',how="mean") # how="sum","mean", "median", "max", "min", "last", "first"
"""
[/python]

7) データ頻度の高解像化（補完）
・　四半期データを月次データに変換。補完方法は直後値の繰り返し(fill-forward: 'ffill')。
[python]
"""
Data=Data.resample('M',fill_method='ffill') # fill_method= 'ffill', 'bfill'

"""
[/python]




■　データ値の処理
8) 欠損値処理
・　欠損値を一括して指定の値(例えば0)に置き換え。
[python]
"""
Data = Data.fillna(0)
"""
[/python]

・　欠損値を所属列の時間平均で置き換え。
[python]
"""
Data = Data.fillna(Data.mean())
"""
[/python]

・　欠損値の直後値で埋め合わせ。
[python]
"""
Data = Data.fillna(method='ffill')
"""
[/python]


・　欠損値の前後で線形補完。
[python]
"""
Data = Data.interpolate()
"""
[/python]


・　欠損値のを含む行（axis=0）か列（axis=1）を削除。
[python]
"""
Data = Data.dropna(axis=0)
"""
[/python]





9) 条件選択でサブデータ作成
・　ある列の値がプラスの時点だけを集めたデータセット（例ではインフレ率: 'Pi'）。
[python]
"""
Data = Data[Data['Pi']>0]
"""
[/python]

10) 列の追加
・　DataFrameに一様乱数列を'Random'という変数名で追加
[python]
"""
Data['Random']=rand(Data.shape[0])
"""
[/python]

11) 行の追加
・　直近時点と同じデータを1時点先の行として追加
[python]
"""
Data=Data.append(pd.DataFrame(Data[-1:].values,columns=Data[-1:].columns,index=Data[-1:].index+1))
"""
[/python]

12) データ値の置き換え（上書き）
・　１行目を乱数列に置き換え。
[python]
"""
Data.iloc[0]=rand(Data.shape[1])
"""
[/python]

・　１列目を乱数列に置き換え。
[python]
"""
Data.iloc[:,0]=rand(Data.shape[0])
"""
[/python]

・　1行1列～5行5列までを乱数に置き換え。
[python]
"""
Data.iloc[0:5,0:5]=rand(5*5).reshape((5,5))
"""
[/python]





13) ソート

・　1列目のデータ値に基づき昇順でソート
[python]
"""
Data.sort(columns=Data.columns[0],ascending=True)
"""
[/python]


・　1列目のデータ値が同じ行については、2列目に基づきソート（昇順）
[python]
"""
Data.sort(columns=list(Data.columns[0:2]),ascending=True)
"""
[/python]




■　系列の変換
14) 記述統計の表示
[python]
"""
Data.describe()
"""
[/python]

15) 変化率の計算
・　前期比
[python]
"""
Data.pct_change()
"""
[/python]

・　前年比（月次）
[python]
"""
Data.pct_change(periods=12)
"""
[/python]




16) 指数化
・　2010年＝100の指数を作成。
[python]
"""
Data/Data['2010'].mean()*100
"""
[/python]


17) 対数変換
[python]
"""
Data.apply(log)
"""
[/python]


18) 四分位範囲で正規化
[python]
"""
(Data - Data.quantile(0.5).values) / (Data.quantile(0.75)-Data.quantile(0.25)).values
"""
[/python]

19) ラグ変数の追加

・　Dataの全ての変数（列）について、ラグ変数を3期分追加したDataFrameを作成。
[python]
"""
nlag=3
for i in range(1,nlag+1):
    LD=Data.join(Data.shift(i),rsuffix="_"+str(i))

"""
[/python]

20) 移動平均の計算
・　後方3期移動平均を計算（均等ウェイト）。
[python]
"""
pd.rolling_mean(Data,3)
"""
[/python]

・　後方3期移動平均を計算（指数関数ウェイト）。
[python]
"""
pd.ewma(Data,span=3)
"""
[/python]

"""




