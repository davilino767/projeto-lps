import pandas as pd
import matplotlib.pyplot as plt

print("Lendo os dados do CERN...")
df = pd.read_csv('subset.csv')

print("Procurando a coluna do mu...")
# Como o print anterior escondeu algumas colunas, esse código acha a do mu
coluna_mu = [col for col in df.columns if 'mu' in col.lower() or 'interact' in col.lower()]
print(f"Achei essas opções para o mu: {coluna_mu}")

# Vamos usar a primeira que ele achar (geralmente é EventInfo.mu ou algo assim)
nome_mu = coluna_mu[0] if coluna_mu else None

print("Gerando o painel com os 4 histogramas...")

# Criando uma figura grande com 4 espaços (2 linhas x 2 colunas)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. Histograma de E_T (Azul)
axs[0, 0].hist(df['ElectronContainer.calo_cluster.et'], bins=100, color='blue', alpha=0.7)
axs[0, 0].set_title('Energia Transversa ($E_T$)')
axs[0, 0].set_xlabel('Valor')
axs[0, 0].set_ylabel('Frequência')

# 2. Histograma de eta (Vermelho)
axs[0, 1].hist(df['ElectronContainer.calo_cluster.eta'], bins=100, color='red', alpha=0.7)
axs[0, 1].set_title('Pseudorapidez ($\eta$)')
axs[0, 1].set_xlabel('Valor')
axs[0, 1].set_ylabel('Frequência')


# 3. Histograma de phi (Verde)
axs[1, 0].hist(df['ElectronContainer.calo_cluster.phi'], bins=100, color='green', alpha=0.7)
axs[1, 0].set_title('Ângulo Azimutal ($\phi$)')
axs[1, 0].set_xlabel('Valor')
axs[1, 0].set_ylabel('Frequência')

# 4. Histograma de mu (Roxo)
if nome_mu:
    axs[1, 1].hist(df[nome_mu], bins=100, color='purple', alpha=0.7)
    axs[1, 1].set_title(r'Média de Interações ($\langle\mu\rangle$)')
    axs[1, 1].set_xlabel('Valor')
    axs[1, 1].set_ylabel('Frequência')
else:
    axs[1, 1].set_title('Coluna do mu não encontrada')

# Ajusta o espaçamento para não encavalar os textos
plt.tight_layout()

# Salva o painel completo
plt.savefig('histogramas_missao2.png')
print("Sucesso Absoluto! Gráfico salvo como 'histogramas_missao2.png'.")
