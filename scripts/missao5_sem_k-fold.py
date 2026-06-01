import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy

df = pd.read_csv('../subset.csv')

y = df['target'].values

colunas_para_remover = [
    col for col in df.columns
    if 'ElectronContainer'    in col
    or 'HLTElectronContainer' in col
    or 'MonteCarloContainer'  in col
]
colunas_para_remover.extend(['target', 'id'])

X_base = df.drop(columns=colunas_para_remover)

rings_limpos     = X_base['TrigEMClusterContainer.ringsE'].str.strip('[]')
rings_expandidos = rings_limpos.str.split(',', expand=True).astype(float)
rings_expandidos.columns = [f'ring_{i}' for i in range(rings_expandidos.shape[1])]

X_base = X_base.drop(columns=['TrigEMClusterContainer.ringsE'])
X_df   = pd.concat([X_base, rings_expandidos], axis=1)

X_df = X_df.replace([np.inf, -np.inf], np.nan)
X_df = X_df.fillna(0)

X = X_df.values

X_resto, X_teste, y_resto, y_teste = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_treino, X_validacao, y_treino, y_validacao = train_test_split(
    X_resto, y_resto, test_size=0.25, random_state=42, stratify=y_resto
)

scaler      = StandardScaler()
X_treino    = scaler.fit_transform(X_treino)
X_validacao = scaler.transform(X_validacao)
X_teste     = scaler.transform(X_teste)

X_treino_t = torch.tensor(X_treino,    dtype=torch.float32)
y_treino_t = torch.tensor(y_treino,    dtype=torch.float32).view(-1, 1)

X_val_t    = torch.tensor(X_validacao, dtype=torch.float32)
y_val_t    = torch.tensor(y_validacao, dtype=torch.float32).view(-1, 1)

X_teste_t  = torch.tensor(X_teste,     dtype=torch.float32)
y_teste_t  = torch.tensor(y_teste,     dtype=torch.float32).view(-1, 1)

dataset_treino = TensorDataset(X_treino_t, y_treino_t)
loader_treino  = DataLoader(dataset_treino, batch_size=32, shuffle=True)


class ClassificadorRinger(nn.Module):

    def __init__(self, numero_de_entradas):
        super(ClassificadorRinger, self).__init__()
        self.camada_oculta = nn.Linear(numero_de_entradas, 5)
        self.relu          = nn.ReLU()
        self.camada_saida  = nn.Linear(5, 1)
        self.sigmoid       = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.camada_oculta(x))
        x = self.sigmoid(self.camada_saida(x))
        return x


model     = ClassificadorRinger(numero_de_entradas=X_treino.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)

epochs             = 50
patience           = 5
melhor_val_loss    = float('inf')
epocas_sem_melhora = 0
melhores_pesos     = copy.deepcopy(model.state_dict())
historico          = {'loss': [], 'val_loss': []}

for epoca in range(epochs):

    model.train()
    loss_acumulada = 0.0

    for lote_X, lote_y in loader_treino:
        optimizer.zero_grad()
        predicao       = model(lote_X)
        loss           = criterion(predicao, lote_y)
        loss.backward()
        optimizer.step()
        loss_acumulada += loss.item() * lote_X.size(0)

    loss_treino = loss_acumulada / len(dataset_treino)

    model.eval()
    with torch.no_grad():
        predicao_val = model(X_val_t)
        loss_val     = criterion(predicao_val, y_val_t).item()

    historico['loss'].append(loss_treino)
    historico['val_loss'].append(loss_val)

    if loss_val < melhor_val_loss:
        melhor_val_loss    = loss_val
        melhores_pesos     = copy.deepcopy(model.state_dict())
        epocas_sem_melhora = 0
    else:
        epocas_sem_melhora += 1
        if epocas_sem_melhora >= patience:
            break

model.load_state_dict(melhores_pesos)

model.eval()
with torch.no_grad():
    y_prob_tensor = model(X_teste_t)

y_prob = y_prob_tensor.numpy().flatten()
y_pred = (y_prob > 0.5).astype(int)

acuracia = np.mean(y_pred == y_teste)
print(f"Acurácia: {acuracia * 100:.2f}%")

cm = confusion_matrix(y_teste, y_pred)
tn, fp, fn, tp = cm.ravel()

pd_val = tp / (tp + fn)
fa_val = fp / (fp + tn)

print(f"PD: {pd_val * 100:.2f}%")
print(f"FA: {fa_val * 100:.2f}%")

plt.figure(figsize=(8, 6))
plt.plot(historico['loss'],     label='Treino',    color='steelblue', linewidth=2)
plt.plot(historico['val_loss'], label='Validação', color='tomato',    linewidth=2, linestyle='--')
plt.title('Curva de Loss por Época')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('grafico_loss.png')
plt.close()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Jato (0)', 'Elétron (1)'])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Matriz de Confusão')
plt.tight_layout()
plt.savefig('matriz_confusao.png')
plt.close()

fpr, tpr, _ = roc_curve(y_teste, y_prob)
roc_auc     = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('FA')
plt.ylabel('PD')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('curva_roc.png')
plt.close()
