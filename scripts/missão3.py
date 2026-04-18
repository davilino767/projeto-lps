import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

print("Carregando e desempacotando os dados do calorímetro...")
dado = pd.read_csv('subset.csv')
dado=dado.copy()
dado['lista_aneis'] = dado['ElectronContainer.ringsE'].apply(ast.literal_eval)


#Histograma do anel 0
print("Gerando o gráfico do Anel 0...")
dado['anel_0'] = dado['lista_aneis'].apply(lambda x: x[0])

plt.figure(figsize=(10, 6))
plt.hist(dado['anel_0'], bins=100, color='royalblue', edgecolor='black', alpha=0.7)

plt.title('Foco Local: Distribuição de Energia no Anel 0 do Calorímetro', fontsize=14)
plt.xlabel('Energia Depositada [MeV]', fontsize=12)
plt.ylabel('Frequência (Número de Eventos)', fontsize=12)
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('histograma_anel_0.png', dpi=300, bbox_inches='tight')
plt.close() 

# Comparação Global: Boxplot de todos os Anéis
print("Gerando o Boxplot Global dos 100 anéis...")
df_aneis = pd.DataFrame(dado['lista_aneis'].tolist())

plt.figure(figsize=(14, 7))
sns.boxplot(data=df_aneis, showfliers=False, color='lightblue', 
            medianprops={'color': 'dimgrey', 'linewidth': 2})

plt.title('Comparação Global: Energia nos Anéis do Calorímetro', fontsize=14)
plt.xlabel('Número do Anel (0 a 99)', fontsize=12)
plt.ylabel('Energia Depositada [MeV] (Escala SymLog)', fontsize=12)

plt.yscale('symlog') 
plt.xticks(range(0, 100, 5))
plt.grid(True, axis='y', linestyle='--', alpha=0.5)


plt.tight_layout()
plt.savefig('boxplot_aneis_global.png', dpi=300, bbox_inches='tight')
plt.close()

print("Sucesso! As imagens 'histograma_anel_0.png' e 'boxplot_aneis_global.png' foram geradas.")
