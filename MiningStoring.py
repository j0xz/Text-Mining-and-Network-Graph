# Mining & Storing Data - Text Mining ... building a spam filter using SMS data #
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
desired_width = 400                                   # variable for desired_width = 400
pd.set_option('display.width', desired_width)         # sets run screen width to 400
pd.set_option('display.max_columns', 20)              # sets run screen column display to 20
pd.set_option('display.max_rows', 100)                # sets run screen rows display to 100
data_file = 'SMSCollection'
sms_raw = pd.read_table(data_file, header=None)
sms_raw.columns = ('spam', 'message')
print(sms_raw.shape)
print(sms_raw.spam.value_counts())
print(sms_raw.head(5))
# ---------------------------------------------------------------------------- Simple text mining of message
spam_keywords = ['click', 'offer', 'winner', 'buy', 'Free', 'cash', 'urgent']  # list of spam key words
for key in spam_keywords:                             # for loop to add spam key word columns
    sms_raw[str(key)] = sms_raw.message.str.contains( # Add spaces around key to get word or pattern match with key  #
                        str(key) or ' '+str(key)+' ', case = False)  # key column assigned True is exists, else False #
print(sms_raw.head(3))                              # top three rows of table with keyword columns
sms_raw['allcaps'] = sms_raw.message.str.isupper()  # creates column named allcaps to check capitalization
print(sms_raw[sms_raw.allcaps==True].head(3))       # three rows of table with message in all caps
sns.heatmap(sms_raw.corr())               # map of correlation matrix for keywords & allcaps
plt.show()
# ---------------------------------------------------------------------------- slice sms_raw into data and target dfs
data = sms_raw[spam_keywords + ['allcaps']]
target = sms_raw['spam']
print(type(data),type(target))
print(data.head(2))
print(target.head(2))
# ------------------- data df is binary and target df is boolean, so Bernoulli distribution is applicable for model
from sklearn.naive_bayes import BernoulliNB                    # import Bernoulli Naive Bayes model classifier
from sklearn.metrics import confusion_matrix                   # import confusion matrix for type I & II errors
bnb = BernoulliNB()                                            # assign an instance of our model classifier to bnb
bnb.fit(data,target)                                           # fit our model to the data dfs
y_pred = bnb.predict(data)                                     # use classification and assign prediction to y_pred
print('Number of mislabeled messages out of a total {} points:  {}'.format(data.shape[0],(target != y_pred).sum()))
# the print statement takes the values from the two arguments and places them respectively in the {}s
print(confusion_matrix(target, y_pred))                        # prints the # Ham predicts , Type I error
#                                                              #            # Type II error, Spam predicts
print(sms_raw.spam.value_counts())                             # sms_raw data counts on ham vs spam to compare above

# ------------------------------------------------------------ # review of spam keywords to replace/change
sms_raw2 = sms_raw[target != y_pred]                           # new df from sms_raw where actual SMS tag != prediction
sms_raw2.to_csv('sms_mispredict.csv')                          # write SMS mispredictions to a .CSV file for viewing
for key in spam_keywords:                              # for loop to get stats on spam key words
        print('--------------------------')            # prints a line separator
        print('spam keyword is ', key)                 # print spam keyword
        print(sms_raw2[sms_raw2[str(key)] == True])    # print rows of sms_raw2 where spam keyword is True in messages
print ('-----------------------------')                               # prints a line separator
print (sms_raw2[sms_raw2.allcaps == True])                            # sms_raw2 rows with allcaps found in messages
print(sms_raw2.allcaps.value_counts(True) * (target != y_pred).sum()) # count of allcaps in messages= allcaps% * total
#-------------------------------------------------------------------- # end of Text Mining Python Code

# Mining & Storing Data - Network Mining ... making network graphs in Python #
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
desired_width = 400                                   # variable for desired_width = 400
pd.set_option('display.width', desired_width)         # sets run screen width to 400
pd.set_option('display.max_columns', 20)              # sets run screen column display to 20
pd.set_option('display.max_rows', 100)                # sets run screen rows display to 100
#---------------------------------------------------- # creating a directed graph
G = nx.DiGraph()                                      # assigning a variable to the DiGraph function
node_list = ['Child','Fiancee','Mom','Dad','Uncle']   # assign node names to a list node_list
G.add_nodes_from (nodes_for_adding = node_list)       # adding nodes - can be added individually with G.add_node()
#---------------------------------------------------- # adding edges - can be added individually with G.add_edge()
G.add_edges_from([('Fiancee', 'Child'), ('Child', 'Fiancee'), ('Child', 'Dad'), ('Child', 'Mom'), ('Mom','Fiancee')])
G.add_edges_from([('Mom', 'Dad'), ('Dad', 'Mom'), ('Mom','Child'), ('Dad','Uncle'), ('Uncle', 'Dad')])
#---------------------------------------------------- # drawing the graph using circular layout
fig1 = nx.draw_networkx(G,
                 pos = nx.circular_layout(G), # positions the node relative to each other #
                 node_size = 1600,            # size of the node
                 cmap = plt.cm.Reds,          # color palate to use {Greens, Blues, Purples, Reds} on the nodes
                 node_color = range(len(G)))  # the number of shades of color to use
plt.axis('off')
plt.savefig('fig1.png')
plt.show()
#---------------------------------------------------- # drawing the graph using spiral layout
fig2 = nx.draw_networkx(G,
                 pos = nx.spiral_layout(G),   # positions the node relative to each other #
                 node_size = 1600,            # size of the node
                 cmap = plt.cm.Purples,       # color palate to use {Greens, Blues, Purples, Reds} on the nodes
                 node_color = range(len(G)))  # the number of shades of color to use
plt.axis('off')
plt.savefig('fig2.png')
plt.show()
#---------------------------------------------------- # drawing the graph using spring layout
fig3 = nx.draw_networkx(G,
                 pos = nx.spiral_layout(G),   # positions the node relative to each other #
                 node_size = 1600,            # size of the node
                 cmap = plt.cm.Greens,        # color palate to use {Greens, Blues, Purples, Reds} on the nodes
                 node_color = range(len(G)))  # the number of shades of color to use
plt.axis('off')
plt.savefig('fig3.png')
plt.show()
#----------------------------------------------------- # pulling information from the network graph
print('This family network graph has {} nodes and {} edges.'.format(G.number_of_nodes(),G.number_of_edges())) # placing numbers in {}
print('The nodes are:  {}'.format(G.nodes()))           # displaying all the nodes on the network graph
print('The edges are:  {}'.format(G.edges()))           # displaying all the edges on the network graph
print('-------------------------------------')         # line separator
print('The Child node has in-degree of {} and an out-degree of {}'.format(G.in_degree('Child'), G.out_degree('Child')))
print('The Fiancee node has in-degree of {} and an out-degree of {}'.format(G.in_degree('Fiancee'), G.out_degree('Fiancee')))
print('The Mom node has in-degree of {} and an out-degree of {}'.format(G.in_degree('Mom'), G.out_degree('Mom')))
print('The Dad node has in-degree of {} and an out-degree of {}'.format(G.in_degree('Dad'), G.out_degree('Dad')))
print('The Uncle node has in-degree of {} and an out-degree of {}'.format(G.in_degree('Uncle'), G.out_degree('Uncle')))
print('-------------------------------------')         # line separator
print('The closeness centrality scores are:  {}'.format(nx.closeness_centrality(G))) # shortest path - most influence
print('The node degrees are:  {}'.format(G.degree()))    # Node degree is the number of edges (node relationships)
# ------------------------------------------------------ # End of Network Mining Python Code

# Mining & Storing Data - Matrix - Using arrays and matrices in NumPy & Pandas #
import numpy as np
import pandas as pd
desired_width = 400                                        # variable for desired_width = 400
pd.set_option('display.width', desired_width)              # sets run screen width to 400
pd.set_option('display.max_columns', 20)                   # sets run screen column display to 20
pd.set_option('display.max_rows', 100)                     # sets run screen rows display to 100
#--------------------------------------------------------- # using NumPy arrays
A = np.array([[1,2,3], [4,5,6]])                           # creating a NumPy two dimensional array A
print(A, ' = Array A ~ Array A type:  ',type(A))           # printing array A type and elements
print(A.T, ' = Array A transposed')                        # printing array A transposed
print(A.T.T, ' = Array A transpose,transpose = Array A')   # printing array A transpose, transpose (original A)
B = np.array([[7,8,9], [10,11,12]])                        # creating a NumPy two dimensional array B
print(B, ' = Array B ~ Array B type:  ',type(B))           # printing array B type and contents
AxBT = np.dot(A, B.T)                           # the np.dot() multiplies array A * array B (must transpose B)
C = A + 2                                       # create a NumPy array C that is array A +2 for each element
C2 = A ** 2                                     # creating a NumPy array C2 that is array A**2 for each element
print(AxBT, ' = Array AxBT = np.dot(A, B.T) ~ Array AxBT type:  ', type(AxBT))  # printing array AxBT
print(C, ' = Array C = Array A + 2 ~ Array C type:  ', type(C))                 # printing array C type and elements
print(C2, ' = Array C2 = Array A**2 ~ Array C2 type:  ', type(C2))              # printing array C2 type and elements
#------------------------------------------------------ # more NumPy arrays, NumPy matrix, & NumPy matrix as Pandas df
C3 = np.array([[1,2,3], [4,5,6], [7,8,9]])              # creating a NumPy two dimensional array C3 that is a 3x3
C4 = C3.diagonal()                                      # np.array.diagonal() returns the diagonal elements of an array
print(C3,' = Array C3 is a 3x3 ~ Array C3 type:  ', type(C3))         # print NumPy two dimensional array C3
print(C4, ' = Array C4 is the C3 diagonal ~ Array C4 type:  ', type(C4))  # print NumPy two dimensional array C4
A = np.matrix(A)                                        #
print(A, ' = matrix A = array A ~ matrix A type:  ', type(A))           # print NumPy matrix A
print('...... matrix A sum = ', A.sum(), ' | mean = ', A.mean(), ' | std = ', A.std(), ' | variance = ', A.var())
print('...... matrix.nonzero() returns indices of nonzero elements ~ ',A.nonzero())  # print indices of nonzero elements
print(A-B, ' = A-B results when array B is subtracted from matrix A')  # print element results of matrix A - array B
print(A+B, ' = A+B results when array B is added to matrix A')         # print element results of matrix A + array B
print(A.dot(B.T), ' = A.dot(B.T) results when matrix A is multiplied by array B transposed')  # print element results
A_df = pd.DataFrame(A)
print(A_df, ' = matrix A as a Pandas Dataframe ~ type(A_df) = ', type(A_df))  # print matrix A as a Pandas dataframe
# -----------------------------------------------------------------# end of Matrix

# Mining & Storing Data - reading and storing data - using sqlite3 package to access a SQL database#
import pandas as pd
import sqlite3
desired_width = 400                                        # variable for desired_width = 400
pd.set_option('display.width', desired_width)              # sets run screen width to 400
pd.set_option('display.max_columns', 20)                   # sets run screen column display to 20
pd.set_option('display.max_rows', 100)                     # sets run screen rows display to 100
#--------------------------------------------------------- # using sqlite3 package library to read a SQLite database
con = sqlite3.connect('test.db')                                   # establish connection to SQLite test.db
cur = con.cursor()                                                 # assign variable to con.cursor() function
for row in cur.execute('SELECT * FROM physician LIMIT 5;'):        # result of SQL script #1 execution iterated by row
    print(row)                                                     # print SQL projection row by row with the 'for loop'
print('                                      ')                    # line separator
for row in cur.execute('SELECT * FROM procedure LIMIT 5;'):        # result of SQL script #2 execution iterated by row
    print(row)                                                     # print SQL projection row by row with the 'for loop'
con.close()                                                        # close the database connection
#--------------------------------------------------------- # using a user defined function to execute a SQL script
def exec_sql():                                            # user defined function to execute a SQL script
    print('                             ')                 # separator line
    print(SQL_script)                                      # print the SQL script to execute
    cur = con.cursor()                                     # assign cur to the connection.cursor() function
    for row in cur.execute(SQL_script):                    # 'for loop' to  receive SQL projection by row
        print(row)                                         # print row by row of projection
con = sqlite3.connect('test.db')                           # establishes connection to SQLite database (open db)
SQL_script = 'SELECT * FROM physician lIMIT 5;'            # assign SQL_script variable to SQL script #1
exec_sql()                                                 # execute user defined function exec_sql() for SQL script #1
SQL_script = 'SELECT * FROM procedure LIMIT 5;'            # assign SQL_script variable to SQL script #2
exec_sql()                                                 # execute user defined function exec_sql() for SQL script #1
con.close()                                                # closes SQLite database connection
# ------------------------------ # reading a SQL script projection to a Pandas df and then writing the df to a CSV file
con = sqlite3.connect('test.db')                           # establish connection to SQLite database test.db
df = pd.read_sql_query("SELECT * FROM physician", con)     # execute SQL script and load project into Pandas df
print(df.head(4))                                          # print top four rows of df
print(df.shape)                                            # print df rows & columns
con.close()                                                # closes SQLite database connection
df.to_csv('physicians.csv', index=False)                   # writes df to a CSV file named physicians.csv
# ---------------------------------------------------------# end of sqlite3 examples

# Mining & Storing Data - reading and storing data - reading a NumPy array from a CSV and saving the array to a CSV #
import pandas as pd
import numpy as np
desired_width = 400                                        # variable for desired_width = 400
pd.set_option('display.width', desired_width)              # sets run screen width to 400
pd.set_option('display.max_columns', 20)                   # sets run screen column display to 20
pd.set_option('display.max_rows', 100)                     # sets run screen rows display to 100
#------------------------------------------------------- # read a 6x6 array, square elements, write a 6x6 matrix to CSV
my_array = np.loadtxt('numpy_6x6_array.csv',delimiter=",",skiprows=1)  # read CSV file into a NumPy array
print(my_array, type(my_array))                            # print my_array & the variable type of my_array
print('                              ')                    # separator line
my_matrix = np.matrix(my_array**2)                         # square each my_array element and convert to NumPy matrix
print(my_matrix, type(my_matrix))                          # print my_matrix & the variable type of my_matrix
np.savetxt("my_matrix_6x6.csv", my_matrix, delimiter=",")  # write my_matrix as a CSV in the Python code subdirectory
# ------------------------------------------------------- # end of reading CSV into NumPy array | writing to a CSV file

# Mining & Storing Data - reading and storing data - reading chunks of a df from CSV; writing/appending to a CSV file #
import pandas as pd
import numpy as np
desired_width = 400                                        # variable for desired_width = 400
pd.set_option('display.width', desired_width)              # sets run screen width to 400
pd.set_option('display.max_columns', 20)                   # sets run screen column display to 20
pd.set_option('display.max_rows', 100)                     # sets run screen rows display to 100
# ------------------------------------------------ # read a 6x8594 array in chunks; write/ append to a CSV file
chunks = 2000                                             # declare chunks variable and assign value of 1000
eof = 8594                                                # end of CSV file is 8594
i = 0
if i <= eof:
    for chunk in pd.read_csv('numpy_6x8594_array.csv',skiprows=i,header=0,chunksize=chunks):  # read CSV file into chunk
        my_array = chunk.to_numpy(dtype=int)                      # convert chunk df to numpy array (integers)
        my_matrix = np.matrix(my_array**2)                        # square elements of my_array as a matrix
        print(my_array[0:5,:], type(my_array))                    # print my_array (5 rows) & the variable type
        print('                              ')                   # separator line
        print(my_matrix[0:5,:], type(my_matrix))                  # print my_matrix (5 rows) & the variable type
        if i < 1:                                                 # if statement to CSV write i = 0, else append i > 0
           np.savetxt('my_matrix_6x8594.csv', my_matrix, delimiter=',')  # write my_matrix_6x8954.csv as a CSV file
        else:
           csv_file = open('my_matrix_6x8594.csv', 'a')                  # create a open file variable for appending
           np.savetxt(csv_file, my_matrix)                               # append to my_matrix_6x8594 as a CSV file
           csv_file.close()
        i = i + chunks                                                     # increment counter
# ---------------------------------------------------------------- # end of reading/writing/appending CSV file

