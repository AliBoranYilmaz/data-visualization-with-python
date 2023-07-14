import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

diamonds = sns.load_dataset("diamonds")
df = diamonds.copy()
print(df.head())
# outputs    carat      cut color clarity  depth  table  price     x     y     z
#         0   0.23    Ideal     E     SI2   61.5   55.0    326  3.95  3.98  2.43
#         1   0.21  Premium     E     SI1   59.8   61.0    326  3.89  3.84  2.31
#         2   0.23     Good     E     VS1   56.9   65.0    327  4.05  4.07  2.31
#         3   0.29  Premium     I     VS2   62.4   58.0    334  4.20  4.23  2.63
#         4   0.31     Good     J     SI2   63.3   58.0    335  4.34  4.35  2.75

print(df.info())
# outputs <class 'pandas.core.frame.DataFrame'>
#         RangeIndex: 53940 entries, 0 to 53939
#         Data columns (total 10 columns):
#          #   Column   Non-Null Count  Dtype
#         ---  ------   --------------  -----
#          0   carat    53940 non-null  float64
#          1   cut      53940 non-null  category
#          2   color    53940 non-null  category
#          3   clarity  53940 non-null  category
#          4   depth    53940 non-null  float64
#          5   table    53940 non-null  float64
#          6   price    53940 non-null  int64
#          7   x        53940 non-null  float64
#          8   y        53940 non-null  float64
#          9   z        53940 non-null  float64
#         dtypes: category(3), float64(6), int64(1)
#         memory usage: 3.0 MB

print(df.describe().T) # transposed
# outputs          count         mean          std    min     25%      50%      75%       max
#         carat  53940.0     0.797940     0.474011    0.2    0.40     0.70     1.04      5.01
#         depth  53940.0    61.749405     1.432621   43.0   61.00    61.80    62.50     79.00
#         table  53940.0    57.457184     2.234491   43.0   56.00    57.00    59.00     95.00
#         price  53940.0  3932.799722  3989.439738  326.0  950.00  2401.00  5324.25  18823.00
#         x      53940.0     5.731157     1.121761    0.0    4.71     5.70     6.54     10.74
#         y      53940.0     5.734526     1.142135    0.0    4.72     5.71     6.54     58.90
#         z      53940.0     3.538734     0.705699    0.0    2.91     3.53     4.04     31.80

print(df["cut"].value_counts())
# outputs cut
#         Ideal        21551
#         Premium      13791
#         Very Good    12082
#         Good          4906
#         Fair          1610
#         Name: count, dtype: int64

from pandas.api.types import CategoricalDtype

df.cut = df.cut.astype(CategoricalDtype(ordered=True))
# We specified that the variable cut is an ordinal variable

cut_categories = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
df.cut = df.cut.astype(CategoricalDtype(categories=cut_categories, ordered=True))
print(df.cut.head(1))
# outputs 0    Ideal
#         Name: cut, dtype: category
#         Categories (5, object): ['Fair' < 'Good' < 'Very Good' < 'Premium' < 'Ideal']

sns.barplot(x = "cut", y = df.cut.index, data = df)
plt.show()

sns.catplot(x = "cut", y = "price", data = df)
plt.show()

sns.barplot(x = "cut", y = "price", hue = "color", data = df)
plt.show()

print(df.groupby(["cut", "color"])["price"].mean()) # verifying the correctness of the plot above
# outputs cut        color
#         Fair       D        4291.061350
#                    E        3682.312500
#                    F        3827.003205
#                    G        4239.254777
#                    H        5135.683168
#                    I        4685.445714
#                    J        4975.655462
#         Good       D        3405.382175
#                    E        3423.644159
#                    F        3495.750275
#                    G        4123.482204
#                    H        4276.254986
#                    I        5078.532567
#                    J        4574.172638
#         Very Good  D        3470.467284
#                    E        3214.652083
#                    F        3778.820240
#                    G        3872.753806
#                    H        4535.390351
#                    I        5255.879568
#                    J        5103.513274
#         Premium    D        3631.292576
#                    E        3538.914420
#                    F        4324.890176
#                    G        4500.742134
#                    H        5216.706780
#                    I        5946.180672
#                    J        6294.591584
#         Ideal      D        2629.094566
#                    E        2597.550090
#                    F        3374.939362
#                    G        3720.706388
#                    H        3889.334831
#                    I        4451.970377
#                    J        4918.186384
#         Name: price, dtype: float64

sns.displot(df.price, kde = True)
plt.show()

sns.displot(df.price, bins = 100, kde = False) # bins = number of bars
plt.show()

sns.kdeplot(df.price, fill = True)
plt.show()

sns.FacetGrid(df, hue="cut", height=5, xlim=(0,10000)).map(sns.kdeplot, "price", fill=True).add_legend();
plt.show()

sns.catplot(x="cut", y="price", hue="color", kind="point", data=df);
plt.show()

tips = sns.load_dataset("tips")
df2 = tips.copy()

sns.boxplot(x = df2["total_bill"])
plt.show()

sns.boxplot(x="day", y="total_bill", data=df2)
plt.show()

sns.boxplot(x="time", y="total_bill", data=df2)
plt.show()

sns.boxplot(x="size", y="total_bill", data=df2)
plt.show()

sns.catplot(y="total_bill", kind="violin", data=df2)
plt.show()

sns.catplot(x="day", y="total_bill", kind="violin", data=df2)
plt.show()

sns.scatterplot(x="total_bill", y="tip", data=df2)
plt.show()

sns.scatterplot(x="total_bill", y="tip", hue="time", data=df2)
plt.show()

sns.lmplot(x="total_bill", y="tip", data=df2) # linear model plot
plt.show()

sns.lmplot(x="total_bill", y="tip", hue="smoker",data=df2)
plt.show()

iris = sns.load_dataset("iris")
df3 = iris.copy()

sns.pairplot(df3)
plt.show()

fmri = sns.load_dataset("fmri")
df4 = fmri.copy()

sns.lineplot(x="timepoint", y="signal", data=df4)
plt.show()

sns.lineplot(x="timepoint", y="signal", hue="event", data=df4)
plt.show()