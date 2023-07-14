import seaborn as sns
import pandas as pd
planets = sns.load_dataset("planets")
print(planets.head())

# This data set is prepared as a result of NASA's researches.
# It includes information about planets discovered by NASA.

df = planets.copy() # Copying the actual data set increases security.

print(df.info())
# outputs <class 'pandas.core.frame.DataFrame'>
#         RangeIndex: 1035 entries, 0 to 1034
#         Data columns (total 6 columns):
#              Column          Non-Null Count  Dtype
#         ---  ------          --------------  -----
#          0   method          1035 non-null   object
#          1   number          1035 non-null   int64
#          2   orbital_period  992 non-null    float64
#          3   mass            513 non-null    float64
#          4   distance        808 non-null    float64
#          5   year            1035 non-null   int64
#         dtypes: float64(3), int64(2), object(1)
#         memory usage: 48.6+ KB

df.method = pd.Categorical(df.method) # "object" data type converted to "categorical" data type
print(df.dtypes)
# outputs method            category
#         number               int64
#         orbital_period     float64
#         mass               float64
#         distance           float64
#         year                 int64

# df.shape returns the number of observations (rows) and variables (columns) of the data set
# df.columns returns the names of variables (columns) of the data set

print(df.describe().T)
# outputs                  count         mean           std          min         25%        50%       75%       max
#         number          1035.0     1.785507      1.240976     1.000000     1.00000     1.0000     2.000       7.0
#         orbital_period   992.0  2002.917596  26014.728304     0.090706     5.44254    39.9795   526.005  730000.0
#         mass             513.0     2.638161      3.818617     0.003600     0.22900     1.2600     3.040      25.0
#         distance         808.0   264.069282    733.116493     1.350000    32.56000    55.2500   178.500    8500.0
#         year            1035.0  2009.070531      3.972567  1989.000000  2007.00000  2010.0000  2012.000    2014.0

print(df.isnull().values.any()) # Is there any null value?
print(df.isnull().sum()) # returns number of null values for each variable
# outputs method              0
#         number              0
#         orbital_period     43
#         mass              522
#         distance          227
#         year                0

df["orbital_period"].fillna(0, inplace = True) # replaces null values with 0
print(df.isnull().sum())
# outputs method              0
#         number              0
#         orbital_period      0
#         mass              522
#         distance          227
#         year                0

df["mass"].fillna(df.mass.mean(), inplace = True) # replaces null values with the mean of desired variable
print(df.isnull().sum())
# outputs method              0
#         number              0
#         orbital_period      0
#         mass                0
#         distance          227
#         year                0

categorical_df = df.select_dtypes(include = ["category"]) # we selected only categorical variables from df
print(categorical_df.head())
# outputs             method
#         0  Radial Velocity
#         1  Radial Velocity
#         2  Radial Velocity
#         3  Radial Velocity
#         4  Radial Velocity

print(categorical_df.method.unique())
# outputs ['Radial Velocity', 'Imaging', 'Eclipse Timing Variations', 'Transit', 'Astrometry', 'Transit Timing Variations', 'Orbital Brightness Modulation', 'Microlensing', 'Pulsar Timing', 'Pulsation Timing Variations']

print(categorical_df.method.value_counts().count())
# outputs 10

print(categorical_df.method.value_counts()) # frequency
# outputs        method
#                Radial Velocity                  553
#                Transit                          397
#                Imaging                           38
#                Microlensing                      23
#                Eclipse Timing Variations          9
#                Pulsar Timing                      5
#                Transit Timing Variations          4
#                Orbital Brightness Modulation      3
#                Astrometry                         2
#                Pulsation Timing Variations        1

print(categorical_df.method.value_counts().plot.barh()) # outputs a bar chart of the data above

df_num = df.select_dtypes(include = ["float64", "int64"])
print(df_num.head())
# outputs    number  orbital_period   mass  distance  year
#         0       1         269.300   7.10     77.40  2006
#         1       1         874.774   2.21     56.95  2008
#         2       1         763.000   2.60     19.84  2011
#         3       1         326.030  19.40    110.62  2007
#         4       1         516.220  10.50    119.47  2009