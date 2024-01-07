import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 20)
chunks = 2000
eof = 15001
i = 0
j = i
if i <= eof:
    for chunk in pd.read_csv('single_family_home_values.csv',skiprows=i,header=0,chunksize=chunks):  # read CSV file into chunk
        df = chunk[['id','address','city','state','zipcode','latitude','longitude','bedrooms','bathrooms','rooms','squareFootage','lotSize','yearBuilt','lastSaleDate','lastSaleAmount','priorSaleDate','priorSaleAmount','estimated_value',]]
        df2 = df[(df.bedrooms>3) & (df.bathrooms>3) & (df.squareFootage>1800)]
        j = j + 1
        if i < 1:
           df2.to_csv('SelectedHomes.csv', index=False)
           i = i + chunks
           rows_written = df2.shape[0]
           rows = df2.shape[0]
        else:
           csv_file = open('SelectedHomes.csv', 'a')
           df2.to_csv(csv_file, index=False, header=False)
           i = i + chunks
           rows = df2.shape[0]
           rows_written += rows
           csv_file.close()
        print('Iteration = ', j, ' Chunks = ', i, ' Diced rows = ', rows, ' Rows written = ', rows_written, ' Data Type = ', type(df2))