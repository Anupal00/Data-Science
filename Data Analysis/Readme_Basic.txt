# Some basic tools for Data Science learn -
1. Scientific Computing Library -   a. Pandas (Data Structure and tools)
                                    b. Numpy (Arrays and matrices)
                                    c. SciPy (Integrals, solving differential equations, optimization)

2. Visualization Library -  a. Matplotlib (plots and graph, most popular)
                            b. Seaborn (plots: heat maps, time series, violin plots)

3. Algorithm Library -  a. Scikit-learn (Machine Learning : Regression, classification and so on)
                        b. Statsmodels (Explore data, estimate statistical models and perform statistical tests)

# How to import / export data for data analysis

import pandas as pd
url = "https://www.some.com/data"
df = pandas.read_csv(url, header = None)

print(df.head(4)) // first 4 rows
print(df.tail()) // last rows

// to set headers
header = ["h1", "h2", "h3"]
df.columns = headers
df.head(5)

// export csv
path = "C:\<actual path>"
df.to_csv(path)
// import and export other methods -
format      import              export
csv         pd.read_csv()       df.to_csv()
json        pd.read_json()      df.to_json()
excel       pd.read_excel()     df.to_excel()
sql         pd.read_sql()       df.to_sql()

# Understand the datasets

pandas datatypes - object, int64, float64, datetime64, timedelta[ns]
native python - string, int, float, N/A

df.dtypes // datatypes
df.describe() // statistical summary (by default omit object type)
df.describe(include = "all") // take object type also
df.info() // overall idea

# Accessing Database
DBMS api for connecting mysql

from dmodule import connect
connection = connect("databasename", "username", "pswd") // create connection object
cursor = connection.cursor() // create cursor object
cursor.execute("select * from table") // run queries
result = cursor.fetchall() // fetch data
cursor.close() // free resource
connection.close() //free resource