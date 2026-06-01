import pandas as pd
import numpy as np

# 1. Lê o arquivo de dados

df = pd.read_csv('../subset.csv')

# 2. Prepara as variáveis físicas
# Usamos o valor absoluto de eta (|eta|) porque a simetria do detector é o que importa
df['abs_eta'] = df['ElectronContainer.calo_cluster.eta'].abs()

# NOTA: No ATLAS, a energia geralmente vem em MeV. Dividimos por 1000 para ter em GeV.
df['ET_GeV'] = df['ElectronContainer.calo_cluster.et'] / 1000.0

# 3. Mapeia a coluna 'target' para nomes mais fáceis de ler
df['Tipo'] = df['target'].map({1: 'Sinal', 0: 'Ruido'})

# 4. Define os intervalos (bins) das regiões
# Você pode alterar esses números baseados nos cortes da sua missão/análise
bins_eta = [0.0, 1.37, 2.5]       # Ex: Separação típica Barrel / Endcap
bins_ET = [0, 20, 40, 100, 500]   # Regiões de energia em GeV

# 5. Corta os dados nessas caixas
df['Regiao_eta'] = pd.cut(df['abs_eta'], bins=bins_eta)
df['Regiao_ET'] = pd.cut(df['ET_GeV'], bins=bins_ET)

# 6. Cria a tabela cruzando as informações
tabela = pd.crosstab(
    index=[df['Regiao_eta'], df['Regiao_ET']], 
    columns=df['Tipo'],
    margins=True,          # Adiciona uma linha/coluna de "Total"
    margins_name="Total"
)

print("\n--- TABELA DE EVENTOS: SINAL VS RUÍDO ---")
print(tabela)
