import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

# ==============================================================================
# O SEU CAMINHO CONFIGURADO (Utilizando 'r' na frente para evitar erros de barra)
# ==============================================================================
PASTA_DADOS = r"C:\Users\janai\Downloads\Dados"

print("Carregando microdados do ENEM 2024 do diretório local...")

# Montando os caminhos completos de forma segura
caminho_itens = os.path.join(PASTA_DADOS, 'ITENS_PROVA_2024.csv')
caminho_resultados = os.path.join(PASTA_DADOS, 'RESULTADOS_2024.csv')

# Lendo os arquivos locais
itens_prova_df = pd.read_csv(caminho_itens, encoding='latin1', sep=';')
resultados_df = pd.read_csv(caminho_resultados, encoding='latin1', sep=';')

print("Dados carregados com sucesso!")

# ==========================================
# 2. PROCESSAMENTO DA MATRIZ DE ACERTOS (MT)
# ==========================================
print("\nProcessando matriz de acertos para Matemática...")
df_mt = resultados_df[resultados_df['TP_PRESENCA_MT'] == 1].sample(10000, random_state=42).copy()

def calcular_vetor_acertos(row):
    resp = str(row['TX_RESPOSTAS_MT'])
    gabarito = str(row['TX_GABARITO_MT'])
    if len(resp) != len(gabarito) or resp == 'nan':
        return None
    return [1 if r == g else 0 for r, g in zip(resp, gabarito)]

acertos_series = df_mt.apply(calcular_vetor_acertos, axis=1).dropna()
matriz_acertos = pd.DataFrame(acertos_series.tolist())

# ==========================================
# 3. ESTRUTURAÇÃO DA REDE BAYESIANA
# ==========================================
itens_mt = itens_prova_df[itens_prova_df['SG_AREA'] == 'MT'].drop_duplicates('CO_POSICAO').sort_values('CO_POSICAO')

edges = []
for _, row in itens_mt.iterrows():
    habilidade_no = f"H_{int(row['CO_HABILIDADE'])}"
    item_no = f"Item_{int(row['CO_POSICAO'])}"
    edges.append((habilidade_no, item_no))

model = DiscreteBayesianNetwork(edges)
print(f"Rede Bayesiana Estruturada: {len(model.nodes())} nós.")

# ==========================================
# 4. TREINAMENTO DO MODELO
# ==========================================
colunas_itens = [f"Item_{int(pos)}" for pos in itens_mt['CO_POSICAO']]
dados_treino = matriz_acertos.copy()
dados_treino.columns = colunas_itens

for node in model.nodes():
    if node.startswith('H_'):
        itens_filhos = [edge[1] for edge in model.edges() if edge[0] == node]
        if itens_filhos:
            dados_treino[node] = (dados_treino[itens_filhos].mean(axis=1) >= 0.5).astype(int)

print("Treinando a rede (Aprendendo CPTs)...")
model.fit(dados_treino)
print("Parâmetros da rede aprendidos com sucesso!")

# ==========================================
# 5. MAPEAMENTO DE GABARITO REAL
# ==========================================
exemplo_gabarito_str = str(df_mt['TX_GABARITO_MT'].iloc[0])
gabarito_real_mapeado = {}
for idx, pos in enumerate(itens_mt['CO_POSICAO']):
    if idx < len(exemplo_gabarito_str):
        gabarito_real_mapeado[int(pos)] = exemplo_gabarito_str[idx]

# ==========================================
# 6. EXPORTAÇÃO DO MODELO E METADADOS
# ==========================================
dados_exportacao = {
    'model': model,
    'gabarito': gabarito_real_mapeado,
    'itens_posicoes': [int(pos) for pos in itens_mt['CO_POSICAO']]
}

# O arquivo será salvo na mesma pasta onde você rodar o script
with open('modelo_enem_rede.pkl', 'wb') as f:
    pickle.dump(dados_exportacao, f)
print("\nArquivo 'modelo_enem_rede.pkl' gerado com sucesso!")