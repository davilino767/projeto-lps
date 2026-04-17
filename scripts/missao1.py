import pandas as dados

# Carrega os dados do CSV
df = dados.read_csv('subset.csv')

# Imprime as 5 primeiras linhas do dataframe
print("Dados:")
print(df.head())
