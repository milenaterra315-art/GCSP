import streamlit as st
import pandas as pd
import base64
import pickle
import os
from pgmpy.inference import VariableElimination

# 1. Configuração da Página
st.set_page_config(page_title="Simulador ENEM - IA", layout="wide")

# Caminhos locais atualizados do seu computador
PASTA_DADOS = r"C:\Users\janai\Downloads\Dados"
PASTA_PROVAS = r"C:\Users\janai\Downloads\Dados\Provas 2024\Dia 2"

# ==============================================================================
# DICIONÁRIO DE CONFIGURAÇÃO DAS PROVAS DO DIA 2 (MATEMÁTICA)
# ==============================================================================
PROVAS_DISPONIVEIS = {
    "Azul (Caderno 7)": {"codigo": 1407, "pdf": "ENEM_2024_P2_CAD_07_DIA_2_AZUL.pdf"},
    "Amarelo (Caderno 5)": {"codigo": 1408, "pdf": "ENEM_2024_P2_CAD_05_DIA_2_AMARELO.pdf"},
    "Cinza (Caderno 6)": {"codigo": 1410, "pdf": "ENEM_2024_P2_CAD_06_DIA_2_CINZA.pdf"},
    "Verde (Caderno 8)": {"codigo": 1409, "pdf": "ENEM_2024_P2_CAD_08_DIA_2_VERDE.pdf"}
}

# Dicionário de Descrições das Habilidades (Mantenha o seu original aqui)
DESCRICOES_HABILIDADES = {
    "H_1": "H1 - Reconhecer, no contexto social, differentes significados...",
    # ... Pode manter todas as suas descrições originais de H_1 a H_30 aqui dentro ...
    "H_30": "H30 - Avaliar propostas de intervenção na realidade utilizando conhecimentos de estatística e probabilidade"
}

# ==============================================================================
# EXTRAÇÃO DINÂMICA DO GABARITO BASEADO NO CÓDIGO DA PROVA
# ==============================================================================
@st.cache_data
def extrair_gabarito_dinamico(codigo_prova):
    caminho_resultados = os.path.join(PASTA_DADOS, 'RESULTADOS_2024.csv')
    
    if not os.path.exists(caminho_resultados):
        st.error(f"❌ Arquivo de resultados não encontrado em: {caminho_resultados}")
        return {}, []

    lista_questoes = list(range(136, 181))
    gabarito_mapeado = {}

    with pd.read_csv(caminho_resultados, encoding='latin1', sep=';', chunksize=5000) as reader:
        for chunk in reader:
            linha_alvo = chunk[chunk['CO_PROVA_MT'] == codigo_prova]
            
            if not linha_alvo.empty:
                str_gabarito = str(linha_alvo['TX_GABARITO_MT'].iloc[0])
                gabarito_mapeado = {q: letra for q, letra in zip(lista_questoes, str_gabarito)}
                break
                
    return gabarito_mapeado, lista_questoes

# ==============================================================================
# BARRA LATERAL INTERATIVA (LISTA SUSPENSA)
# ==============================================================================
st.sidebar.header("📋 Configuração do Simulado")
prova_escolhida = st.sidebar.selectbox(
    "Selecione o caderno que você realizou:",
    options=list(PROVAS_DISPONIVEIS.keys())
)

# Resgata as configurações específicas da prova selecionada
codigo_prova_atual = PROVAS_DISPONIVEIS[prova_escolhida]["codigo"]
nome_pdf_atual = PROVAS_DISPONIVEIS[prova_escolhida]["pdf"]

# Executa a extração baseada no clique/seleção do usuário
gabarito_oficial, lista_questoes = extrair_gabarito_dinamico(codigo_prova_atual)

# ==========================================
# CARREGAMENTO DO MODELO BAYESIANO (.PKL)
# ==========================================
@st.cache_resource
def carregar_dados_rede():
    caminho_pkl = 'modelo_enem_rede.pkl'
    if os.path.exists(caminho_pkl):
        with open(caminho_pkl, 'rb') as f:
            dados = pickle.load(f)
            if isinstance(dados, dict):
                return dados['model']
            return dados
    return None

model = carregar_dados_rede()

if model is None:
    st.error("❌ Arquivo 'modelo_enem_rede.pkl' não encontrado!")
    st.info("Execute 'py treinamodelo.py' no terminal primeiro.")
    st.stop()

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================
st.title("🧠 Portal do Aluno: Avaliação via Rede Bayesiana (Matemática ENEM)")
st.markdown(f"Análise baseada no caderno: **{prova_escolhida}** (Código INEP: `{codigo_prova_atual}`)")

col_prova, col_respostas = st.columns([1, 1])

# --- COLUNA DA ESQUERDA: Exibição Dinâmica do PDF ---
with col_prova:
    st.subheader(f"📄 Caderno de Questões ({prova_escolhida})")
    CAMINHO_PDF = os.path.join(PASTA_PROVAS, nome_pdf_atual)
    
    if os.path.exists(CAMINHO_PDF):
        with open(CAMINHO_PDF, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_html = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></embed>'
        st.markdown(pdf_html, unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ Arquivo PDF não encontrado em: {CAMINHO_PDF}")

# --- COLUNA DA DIREITA: Entrada de Dados e Inferência ---
with col_respostas:
    st.subheader("🖋️ Cartão de Respostas Interativo")
    
    with st.expander("🔍 Conferir Gabarito Oficial Extraído"):
        st.write(gabarito_oficial)

    with st.form("gabarito_aluno"):
        respostas_aluno = {}
        st.write("Marque as alternativas assinaladas:")
        
        for q in sorted(lista_questoes):
            respostas_aluno[q] = st.radio(
                f"Questão {q}:",
                ('Não Respondida', 'A', 'B', 'C', 'D', 'E'),
                horizontal=True,
                key=f"q_{q}_{codigo_prova_atual}" # Chave dinâmica para limpar o form ao mudar de prova
            )
            
        st.markdown("---")
        botao_enviar = st.form_submit_button("Submeter Respostas para Análise Probabilística")

    # Processamento pós-submissão
    if botao_enviar:
        evidencias_rede = {}
        total_respondidas = 0
        acertos = 0
        
        for q, resp in respostas_aluno.items():
            if resp != 'Não Respondida':
                total_respondidas += 1
                gabarito_correto = gabarito_oficial.get(q, "X")
                acertou = 1 if resp == gabarito_correto else 0
                if acertou == 1:
                    acertos += 1
                
                evidencias_rede[f"Item_{q}"] = acertou

        if total_respondidas == 0:
            st.warning("Por favor, preencha o gabarito antes de processar.")
        else:
            st.markdown("### 📊 Resultado Geral")
            st.metric(
                label="Total de Acertos em Matemática", 
                value=f"{acertos} / {total_respondidas}", 
                delta=f"Aproveitamento: {(acertos/total_respondidas)*100:.1f}%"
            )
            
            st.markdown("---")
            st.markdown("### 🔮 Diagnóstico de Habilidades Latentes (Inferência IA)")
            
            try:
                inference = VariableElimination(model)
                nos_habilidade = [node for node in model.nodes() if str(node).startswith('H_')]
                
                perfil_diagnostico = []
                for hab in sorted(nos_habilidade, key=lambda x: int(x.split('_')[1])):
                    resultado = inference.query(variables=[hab], evidence=evidencias_rede, show_progress=False)
                    prob_dominio = resultado.values[1] 
                    perfil_diagnostico.append({'Habilidade': hab, 'Probabilidade': prob_dominio})
                
                df_diagnostico = pd.DataFrame(perfil_diagnostico)
                
                for _, row in df_diagnostico.iterrows():
                    hab_key = row['Habilidade']
                    hab_texto = DESCRICOES_HABILIDADES.get(hab_key, hab_key)
                    
                    st.write(f"**{hab_texto}**")
                    st.progress(float(row['Probabilidade']))
                    st.caption(f"Probabilidade de domínio real: **{row['Probabilidade']:.2%}**")
                    
            except Exception as error:
                st.error(f"Erro na inferência da rede pgmpy: {error}")