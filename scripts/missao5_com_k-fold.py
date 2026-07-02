import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv('../subset.csv')
y = df['target'].values

colunas_offline = [
    col for col in df.columns
    if 'ElectronContainer' in col
    or 'HLTElectronContainer' in col
    or 'MonteCarloContainer' in col
]
X_base = df.drop(columns=colunas_offline + ['target', 'id'])

rings_como_string = X_base['TrigEMClusterContainer.ringsE'].str.strip('[]')
rings_expandidos = rings_como_string.str.split(',', expand=True).astype(float)
rings_expandidos.columns = [f'ring_{i}' for i in range(rings_expandidos.shape[1])]

X_base = X_base.drop(columns=['TrigEMClusterContainer.ringsE'])
X_df = pd.concat([X_base, rings_expandidos], axis=1)
X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
X = X_df.values


class RingerModel(nn.Module):
    def __init__(self, numero_de_features: int):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(numero_de_features, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.rede(x)


NUM_FOLDS = 5
MAX_EPOCAS = 50
PACIENCIA = 5

kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

acuracias = []
aucs = []
pds = []
fas = []

for fold_idx, (idx_treino_val, idx_teste) in enumerate(kfold.split(X, y)):
    fold_num = fold_idx + 1
    print(f"\n{'='*50}  FOLD {fold_num}/{NUM_FOLDS}  {'='*50}")

    X_treino_val, X_teste = X[idx_treino_val], X[idx_teste]
    y_treino_val, y_teste = y[idx_treino_val], y[idx_teste]

    X_treino, X_val, y_treino, y_val = train_test_split(
        X_treino_val, y_treino_val,
        test_size=0.20,
        random_state=42,
        stratify=y_treino_val,
    )

    scaler = StandardScaler()
    X_treino = scaler.fit_transform(X_treino)
    X_val = scaler.transform(X_val)
    X_teste = scaler.transform(X_teste)

    X_treino_t = torch.tensor(X_treino, dtype=torch.float32)
    y_treino_t = torch.tensor(y_treino, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_teste_t = torch.tensor(X_teste, dtype=torch.float32)


    loader_treino = DataLoader(
        TensorDataset(X_treino_t, y_treino_t),
        batch_size=128, 
        shuffle=True,
    )

    modelo = RingerModel(X_treino.shape[1])
    criterio = nn.BCELoss()
    
   
    otimizador = optim.Adam(modelo.parameters(), lr=0.001, weight_decay=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode='min', factor=0.5, patience=2
    )

    melhor_val_loss = float('inf')
    epocas_sem_melhora = 0
    melhores_pesos = None
    historico = {'loss_treino': [], 'loss_val': []}

    for epoca in range(MAX_EPOCAS):
        modelo.train()
        loss_treino_total = 0.0

        for batch_X, batch_y in loader_treino:
            otimizador.zero_grad()
            loss = criterio(modelo(batch_X), batch_y)
            loss.backward()
            otimizador.step()
            loss_treino_total += loss.item() * batch_X.size(0)

        loss_treino_media = loss_treino_total / len(loader_treino.dataset)

        modelo.eval()
        with torch.no_grad():
            loss_val = criterio(modelo(X_val_t), y_val_t).item()

        historico['loss_treino'].append(loss_treino_media)
        historico['loss_val'].append(loss_val)

        scheduler.step(loss_val)

        if loss_val < melhor_val_loss:
            melhor_val_loss = loss_val
            melhores_pesos = copy.deepcopy(modelo.state_dict())
            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= PACIENCIA:
                print(f"  Early stopping na época {epoca + 1}")
                break

    modelo.load_state_dict(melhores_pesos)

    modelo.eval()
    with torch.no_grad():
        y_prob = modelo(X_teste_t).numpy()

    y_pred = (y_prob > 0.5).astype(int)

    acuracia = np.mean(y_pred.flatten() == y_teste)
    acuracias.append(acuracia)

    cm = confusion_matrix(y_teste, y_pred)
    tn, fp, fn, tp = cm.ravel()

    pds.append(tp / (tp + fn))
    fas.append(fp / (fp + tn))

    fpr_roc, tpr_roc, _ = roc_curve(y_teste, y_prob)
    roc_auc = auc(fpr_roc, tpr_roc)
    aucs.append(roc_auc)

    print(f"  Acurácia: {acuracia * 100:.2f}%  |  AUC: {roc_auc:.4f}")
    print(f"  PD: {pds[-1] * 100:.2f}%  |  FA: {fas[-1] * 100:.2f}%")

    plt.figure(figsize=(6, 4))
    plt.plot(historico['loss_treino'], label='Treino', color='steelblue')
    plt.plot(historico['loss_val'], label='Validação', color='tomato', linestyle='--')
    plt.title(f'Curva de Loss — Fold {fold_num}')
    plt.xlabel('Épocas')
    plt.ylabel('BCELoss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'grafico_loss_fold_{fold_num}.png')
    plt.close()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Jato (0)', 'Elétron (1)'])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Matriz de Confusão — Fold {fold_num}')
    plt.tight_layout()
    plt.savefig(f'matriz_confusao_fold_{fold_num}.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(fpr_roc, tpr_roc, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Aleatório')
    plt.title(f'Curva ROC — Fold {fold_num}')
    plt.xlabel('Taxa de Falso Alarme (FA)')
    plt.ylabel('Prob. de Detecção (PD)')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'curva_roc_fold_{fold_num}.png')
    plt.close()


barra = "=" * 55
print(f"\n{barra}")
print("  RESULTADO FINAL — K-Fold (Média ± Desvio Padrão)".center(55))
print(barra)
print(f"  Acurácia : {np.mean(acuracias) * 100:.2f}% (± {np.std(acuracias) * 100:.2f}%)")
print(f"  AUC      : {np.mean(aucs):.4f}  (± {np.std(aucs):.4f})")
print(f"  PD (TPR) : {np.mean(pds) * 100:.2f}% (± {np.std(pds) * 100:.2f}%)")
print(f"  FA (FPR) : {np.mean(fas) * 100:.2f}% (± {np.std(fas) * 100:.2f}%)")
print(barra)
print(f"\n✅ {NUM_FOLDS * 3} gráficos salvos.")
