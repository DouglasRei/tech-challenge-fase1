
# TECH CHALLENGE FIAP - FASE 1
# ETAPA 8 - PASSO 33
# Interface inicial em Streamlit

import json
from pathlib import Path
from relatorio_pdf import gerar_relatorio_pdf
from validacao_entrada import validar_amostra, EntradaInvalida
import logging

logger = logging.getLogger(__name__)

import joblib
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.special import expit


# 1. Configuracao da pagina
st.set_page_config(
    page_title="Tech Challenge - IA Explicavel",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. Localizacao dos arquivos
RAIZ = Path(__file__).resolve().parent.parent

ARQUIVO_MODELO = (
    RAIZ / "models" / "modelo_final_logistica.joblib"
)

ARQUIVO_DECISAO = (
    RAIZ / "reports" / "decisao_modelo.json"
)

ARQUIVO_METRICAS = (
    RAIZ / "reports" / "metricas_finais_teste.json"
)


# 3. Carregar o modelo definitivo
# Carregar somente arquivos produzidos pelo projeto.
@st.cache_resource
def carregar_modelo(caminho):
    return joblib.load(caminho)


# 4. Ler arquivos JSON locais
def carregar_json(caminho):
    with caminho.open("r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


# 5. Validar os arquivos antes de carregar
arquivos = {
    "Modelo definitivo": ARQUIVO_MODELO,
    "Decisao metodologica": ARQUIVO_DECISAO,
    "Metricas finais": ARQUIVO_METRICAS,
}

arquivos_ausentes = [
    nome for nome, caminho in arquivos.items()
    if not caminho.is_file()
]

if arquivos_ausentes:
    st.error("Arquivos necessarios nao encontrados.")
    st.write("Verifique:", arquivos_ausentes)
    st.stop()



# 6. Carregar os artefatos

from sklearn.pipeline import Pipeline

try:
    artefato_carregado = carregar_modelo(
        str(ARQUIVO_MODELO)
    )

    decisao = carregar_json(ARQUIVO_DECISAO)
    metricas = carregar_json(ARQUIVO_METRICAS)

except Exception:
    st.error(
        "Nao foi possivel carregar os artefatos."
    )
    st.stop()


# 7. Identificar o pipeline salvo

if isinstance(artefato_carregado, Pipeline):

    # O arquivo contem diretamente o pipeline
    modelo = artefato_carregado

elif isinstance(artefato_carregado, dict):

    # O arquivo contem um dicionario.
    # Identificar pipelines validos dentro dele.
    candidatos = [
        (nome, valor)
        for nome, valor in artefato_carregado.items()
        if isinstance(valor, Pipeline)
        and callable(
            getattr(valor, "predict_proba", None)
        )
    ]

    if len(candidatos) != 1:

        st.error(
            "Nao foi possivel identificar um "
            "pipeline unico no arquivo."
        )

        st.write(
            "Chaves encontradas:",
            list(artefato_carregado.keys())
        )

        st.write(
            "Tipos encontrados:",
            {
                chave: type(valor).__name__
                for chave, valor
                in artefato_carregado.items()
            }
        )

        st.stop()

    nome_pipeline, modelo = candidatos[0]

    st.sidebar.caption(
        f"Pipeline identificado: {nome_pipeline}"
    )

else:

    st.error(
        "Formato de artefato nao reconhecido."
    )

    st.write(
        "Tipo encontrado:",
        type(artefato_carregado).__name__
    )

    st.stop()


# 8. Validar previsoes probabilisticas

if not callable(
    getattr(modelo, "predict_proba", None)
):
    st.error(
        "O pipeline nao permite previsoes "
        "probabilisticas."
    )
    st.stop()

st.sidebar.success(
    "Pipeline identificado e validado!"
)


#correcao aqui 

# Validar a quantidade de variaveis do modelo

quantidade_variaveis = getattr(
    modelo,
    "n_features_in_",
    None
)

if quantidade_variaveis is None:
    st.error(
        "Nao foi possivel identificar a "
        "quantidade de variaveis do modelo."
    )
    st.stop()

if quantidade_variaveis != 30:
    st.error(
        "O modelo nao possui as 30 "
        "variaveis esperadas."
    )
    st.stop()

# Recuperar o limiar de classificacao

if not isinstance(decisao, dict):
    st.error("Configuracao metodologica invalida.")
    st.stop()

limiar = decisao.get("limiar")

if (
    isinstance(limiar, bool)
    or not isinstance(limiar, (int, float))
    or not 0 <= limiar <= 1
):
    st.error(
        "Limiar de classificacao invalido."
    )
    st.stop()

print("Quantidade de variaveis:", quantidade_variaveis)
print("Limiar:", limiar)



# 8. Barra lateral
with st.sidebar:
    st.title("🧬 Tech Challenge")

    st.caption("FIAP | Fase 1")

    st.divider()

    st.success("Modelo carregado")

    st.write("**Algoritmo:** Regressao Logistica")
    st.write(f"**Variaveis:** {quantidade_variaveis}")
    st.write(f"**Limiar:** {limiar:.0%}")

    st.divider()

    st.caption(
        "Sistema demonstrativo academico. "
        "Nao destinado a uso clinico."
    )


# 9. Cabecalho principal
st.title("Inteligencia Artificial Explicavel")

st.write(
    "Sistema demonstrativo de classificacao "
    "de caracteristicas morfologicas do dataset "
    "Breast Cancer Wisconsin."
)

st.warning(
    "Esta aplicacao e um prototipo academico. "
    "Os resultados nao constituem diagnostico "
    "medico e nao devem orientar decisoes clinicas."
)


# 10. Indicadores principais
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Modelo",
        "Regressao Logistica"
    )

with col2:
    st.metric(
        "Caracteristicas",
        quantidade_variaveis
    )

with col3:
    st.metric(
        "Limiar de classificacao",
        f"{limiar:.0%}"
    )


# 11. Abas da aplicacao
aba1, aba2, aba3, aba4 = st.tabs([
    "Visao geral",
    "Modelo e metodo",
    "Auditoria",
    "Simulador de IA"
])


# 12. Visao geral
with aba1:
    st.subheader("Sobre o projeto")

    st.write(
        "O projeto utiliza aprendizado de maquina "
        "supervisionado para estudar a classificacao "
        "de amostras em benignas e malignas."
    )

    st.write(
        "O modelo foi treinado e avaliado "
        "utilizando dados publicos do dataset "
        "Breast Cancer Wisconsin."
    )

    st.info(
        "O módulo de previsões interativas já está disponível "
        "na aba Simulador de IA."
    )


# 13. Modelo e metodo
with aba2:
    st.subheader("Configuracao metodologica")

    st.write(
        "A aplicacao utiliza o modelo definitivo "
        "salvo apos o processo experimental."
    )

    st.json(decisao)

    st.subheader("Estrutura do pipeline")

    st.code(
        str(modelo),
        language="text"
    )


# 14. Auditoria
with aba3:
    st.subheader("Resultados da avaliacao")

    st.write(
        "Metricas registradas na avaliacao "
        "independente do conjunto de teste."
    )

    st.json(metricas)

    st.subheader("Verificacoes")

    st.success("Arquivo do modelo encontrado")
    st.success("Configuracao metodologica carregada")
    st.success("Metricas finais carregadas")
    st.success("Quantidade de variaveis validada")
###########

# ============================================
# PASSO 35 - EXPLICABILIDADE COM SHAP
# ============================================

def mostrar_explicacao_shap(
    modelo,
    X_entrada,
    probabilidade_maligna
):
    st.divider()
    st.subheader("4. Explicacao da previsao com SHAP")

    st.write(
        "O grafico mostra como cada caracteristica "
        "contribuiu para a pontuacao da classificacao."
    )

    # Recuperar os componentes do pipeline
    if not hasattr(modelo, "named_steps"):
        st.error("O modelo nao possui um pipeline valido.")
        return

   
    # Identificar os componentes pelo tipo,
    # independentemente dos nomes utilizados no pipeline.

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    etapas = modelo.named_steps

    padronizador = next(
        (
            componente
            for componente in etapas.values()
            if isinstance(componente, StandardScaler)
        ),
        None
    )

    classificador = next(
        (
            componente
            for componente in etapas.values()
            if isinstance(componente, LogisticRegression)
        ),
        None
    )

    # Confirmar que os componentes foram encontrados
    if padronizador is None or classificador is None:
        st.error(
            "Nao foi possivel identificar os componentes "
            "do pipeline."
        )

        st.write(
            "Etapas encontradas:",
            [
                (nome, type(componente).__name__)
                for nome, componente in modelo.steps
            ]
        )

        return


    # Garantir que a classe positiva e malignidade
    if list(classificador.classes_) != [0, 1]:
        st.error("Classes do classificador inesperadas.")
        return

    try:
        # Padronizar usando o transformador ja treinado
        X_padronizado = padronizador.transform(
            X_entrada
        )

        # O vetor zero corresponde as medias
        # aprendidas pelo StandardScaler.
        # Referencia: SHAP linear independente.
        referencia = np.zeros(
            (1, X_padronizado.shape[1])
        )

        explicador = shap.LinearExplainer(
            classificador,
            referencia
        )

        valores = explicador(X_padronizado)

        contribuicoes = np.asarray(
            valores.values[0],
            dtype=float
        )

        valor_base = float(
            np.asarray(valores.base_values).reshape(-1)[0]
        )

        # Validar a propriedade aditiva do SHAP
        log_odds_shap = (
            valor_base + contribuicoes.sum()
        )

        log_odds_modelo = float(
            classificador.decision_function(
                X_padronizado
            )[0]
        )

        if not np.isclose(
            log_odds_shap,
            log_odds_modelo,
            atol=1e-6
        ):
            raise ValueError(
                "A explicacao nao reconstruiu "
                "a saida do modelo."
            )

        # Confirmar a consistencia da probabilidade
        probabilidade_reconstruida = float(
            expit(log_odds_shap)
        )

        if not np.isclose(
            probabilidade_reconstruida,
            probabilidade_maligna,
            atol=1e-6
        ):
            raise ValueError(
                "A probabilidade explicada "
                "nao corresponde a previsao."
            )

        # Exibir valores originais no grafico,
        # mantendo contribuicoes na escala log-odds
        explicacao = shap.Explanation(
            values=contribuicoes,
            base_values=valor_base,
            data=X_entrada.iloc[0].to_numpy(
                dtype=float
            ),
            feature_names=list(X_entrada.columns)
        )

    except (ValueError, TypeError, AttributeError, IndexError):
        st.error(
            "Nao foi possivel validar a explicacao SHAP. "
            "Verifique os componentes e os dados do modelo."
        )
        return

    # Gerar o grafico waterfall
    st.markdown("#### Contribuicoes das caracteristicas")

    shap.plots.waterfall(
        explicacao,
        max_display=11,
        show=False
    )

    figura = plt.gcf()
    st.pyplot(figura)
    plt.close(figura)

    st.caption(
        "Contribuicoes na escala de log-odds. "
        "Valores positivos favorecem malignidade; "
        "valores negativos favorecem benignidade. "
        "A referencia representa as medias de "
        "treinamento apos padronizacao."
    )

    # Tabela das principais contribuicoes
    tabela = pd.DataFrame({
        "Caracteristica": X_entrada.columns,
        "Contribuicao SHAP": contribuicoes
    })

    tabela["Impacto absoluto"] = (
        tabela["Contribuicao SHAP"].abs()
    )

    tabela = tabela.sort_values(
        "Impacto absoluto",
        ascending=False
    ).head(10)

    st.markdown(
        "#### Dez caracteristicas de maior impacto"
    )

    st.dataframe(
        tabela,
        hide_index=True,
        use_container_width=True
    )

    st.success(
        "Explicacao SHAP calculada e "
        "validada com sucesso!"
    )




#Modificação do código para adicionar a aba de simulador interativo de IA

# ================================================
# ETAPA 8 - PASSO 34
# SIMULADOR INTERATIVO DE INTELIGENCIA ARTIFICIAL
# ================================================

with aba4:

    st.header("Simulador de Inteligencia Artificial")

    st.write(
        "Selecione uma amostra publica do dataset "
        "Breast Cancer Wisconsin e altere suas "
        "caracteristicas para explorar o modelo."
    )

    st.info(
        "A simulacao utiliza o modelo definitivo "
        "ja treinado. Nenhum novo treinamento sera realizado."
    )

    # 1. Nomes das 10 caracteristicas fundamentais

    caracteristicas_base = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension"
    ]

    # 2. Construir as 30 caracteristicas na ordem
    # original utilizada no treinamento

    colunas_entrada = [
        f"{nome}_{sufixo}"
        for sufixo in ["mean", "se", "worst"]
        for nome in caracteristicas_base
    ]

    assert len(colunas_entrada) == 30

    # 3. Verificar a ordem das colunas do modelo

    nomes_modelo = getattr(
        modelo,
        "feature_names_in_",
        None
    )

    if nomes_modelo is None:
        st.error(
            "O pipeline nao informa os nomes das "
            "caracteristicas. Verifique a ordem "
            "do treinamento antes de realizar previsoes."
        )
        st.stop()

    if list(nomes_modelo) != colunas_entrada:
        st.error(
            "A ordem das caracteristicas nao "
            "corresponde ao modelo treinado."
        )

        st.write(
            "Colunas esperadas pelo modelo:",
            list(nomes_modelo)
        )

        st.stop()

    # 4. Localizar o dataset original

    arquivo_dataset = (
        RAIZ / "data" / "raw" / "wdbc.data"
    )

    if not arquivo_dataset.is_file():
        st.error(
            "O arquivo data/raw/wdbc.data "
            "nao foi encontrado."
        )
        st.stop()

    # 5. Carregar os exemplos publicos

    try:
        dados_originais = pd.read_csv(
            arquivo_dataset,
            header=None,
            names=[
                "id",
                "diagnosis",
                *colunas_entrada
            ]
        )

    except (OSError, ValueError, pd.errors.ParserError):
        st.error(
            "Nao foi possivel carregar o dataset."
        )
        st.stop()

    # 6. Validar a estrutura do dataset

    if dados_originais.shape != (569, 32):
        st.error(
            "O dataset nao possui as dimensoes "
            "originais esperadas."
        )
        st.stop()

    if set(dados_originais["diagnosis"]) != {"B", "M"}:
        st.error("Classes inesperadas no dataset.")
        st.stop()

    # PASSO 39 - Validar o referencial numérico antes de gerar limites.
    dados_referencia = dados_originais[colunas_entrada].apply(
        pd.to_numeric, errors="coerce"
    )
    valores_referencia = dados_referencia.to_numpy(dtype=float)
    if (
        not np.isfinite(valores_referencia).all()
        or (valores_referencia < 0).any()
    ):
        st.error("O dataset de referência contém valores numéricos inválidos.")
        st.stop()

    minimos_referencia = dados_referencia.min(axis=0)
    maximos_referencia = dados_referencia.max(axis=0)

    # 7. Selecionar uma amostra inicial

    st.subheader("1. Escolha uma amostra")

    tipo_amostra = st.radio(
        "Exemplo para iniciar a simulacao:",
        [
            "Exemplo benigno",
            "Exemplo maligno"
        ],
        horizontal=True
    )

    rotulo_original = (
        "B" if tipo_amostra == "Exemplo benigno"
        else "M"
    )

    # Evitar que o PDF de um exemplo anterior permaneça disponível
    # quando o usuário troca o tipo de amostra.
    if st.session_state.get("tipo_amostra_anterior") not in (None, rotulo_original):
        st.session_state.pop("relatorio_pdf", None)
    st.session_state["tipo_amostra_anterior"] = rotulo_original

    amostra = dados_originais.loc[
        dados_originais["diagnosis"] == rotulo_original
    ].iloc[0]

    st.caption(
        "Os valores iniciais pertencem ao dataset "
        "publico e podem ter participado do "
        "desenvolvimento do modelo. "
        "Nao representam uma nova avaliacao independente."
    )

    # 8. Construir o formulario

    st.subheader("2. Caracteristicas morfologicas")

    entradas = {}

    with st.form(
        key=f"form_simulador_{rotulo_original}"
    ):

        grupos = [
            ("mean", "Medidas medias"),
            ("se", "Erros-padrao"),
            ("worst", "Maiores medidas")
        ]

        for sufixo, titulo in grupos:

            with st.expander(
                titulo,
                expanded=(sufixo == "mean")
            ):

                esquerda, direita = st.columns(2)

                nomes_grupo = [
                    nome for nome in colunas_entrada
                    if nome.endswith(f"_{sufixo}")
                ]

                for i, nome in enumerate(nomes_grupo):

                    coluna = (
                        esquerda if i % 2 == 0
                        else direita
                    )

                    with coluna:
                        entradas[nome] = st.number_input(
                            label=nome,
                            min_value=0.0,
                            value=float(amostra[nome]),
                            step=0.0001,
                            format="%.6f",
                            key=f"{rotulo_original}_{nome}"
                        )

        executar = st.form_submit_button(
            "Executar previsao",
            type="primary"
        )

    # 9. Realizar a previsao

    if executar:

        # Remover relatorio anterior antes de processar uma nova previsao.
        st.session_state.pop("relatorio_pdf", None)

        X_entrada = pd.DataFrame(
            [[
                entradas[nome]
                for nome in colunas_entrada
            ]],
            columns=colunas_entrada
        )

        # 10. Validar 30 entradas, ordem, finitude e limites operacionais.
        try:
            X_entrada = validar_amostra(
                X_entrada,
                colunas_entrada,
                maximos_referencia
            )
        except EntradaInvalida as erro:
            st.error(f"Não foi possível executar a previsão: {erro}")
            st.stop()

        # Alertar quando a simulação extrapolar os valores vistos no dataset.
        # O aviso é informativo; não é critério clínico.
        fora_da_referencia = [
            nome for nome in colunas_entrada
            if (
                X_entrada.iloc[0][nome] < minimos_referencia[nome]
                or X_entrada.iloc[0][nome] > maximos_referencia[nome]
            )
        ]
        if fora_da_referencia:
            nomes_fora = ", ".join(fora_da_referencia[:5])
            if len(fora_da_referencia) > 5:
                nomes_fora += ", entre outras"
            st.warning(
                f"Atenção: esta simulação usa valores fora da faixa do "
                f"dataset de referência ({nomes_fora}). "
                "A confiabilidade da previsão pode ser menor."
            )

        # 11. Identificar a classe maligna

        classes_modelo = list(modelo.classes_)

        if 1 not in classes_modelo:
            st.error(
                "A classe maligna nao foi encontrada "
                "no classificador."
            )
            st.stop()

        indice_maligno = classes_modelo.index(1)

        # 12. Gerar a probabilidade

        try:
            probabilidades = modelo.predict_proba(
                X_entrada
            )

            probabilidade_maligna = float(
                probabilidades[0, indice_maligno]
            )

        except (ValueError, TypeError, OverflowError, FloatingPointError):
            st.error(
                "Nao foi possivel processar "
                "os valores informados."
            )
            st.stop()

        # 13. Validar a probabilidade

        if not (
            np.isfinite(probabilidade_maligna)
            and 0 <= probabilidade_maligna <= 1
        ):
            st.error(
                "A probabilidade retornada e invalida."
            )
            st.stop()

        # 14. Aplicar o limiar definitivo

        previsao = int(
            probabilidade_maligna >= limiar
        )

        nome_previsao = (
            "Maligno" if previsao == 1
            else "Benigno"
        )

        # 15. Apresentar o resultado

        st.divider()

        st.subheader("3. Resultado da simulacao")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Classificacao estimada",
                nome_previsao
            )

        with c2:
            st.metric(
                "Probabilidade de malignidade",
                f"{probabilidade_maligna * 100:.4f}%"
            )

        with c3:
            st.metric(
                "Limiar utilizado",
                f"{limiar:.0%}"
            )

        st.progress(
            probabilidade_maligna,
            text="Probabilidade estimada de malignidade"
        )

        if previsao == 1:
            st.warning(
                "O modelo classificou a amostra "
                "como maligna."
            )
        else:
            st.success(
                "O modelo classificou a amostra "
                "como benigna."
            )

        # 16. Verificar se os dados foram alterados

        valores_originais = (
            amostra[colunas_entrada]
            .to_numpy(dtype=float)
        )
        
        dados_alterados = not np.allclose(
            X_entrada.iloc[0].to_numpy(dtype=float),
            valores_originais,
            rtol=0,
            atol=1e-9
)

       

        if dados_alterados:

            st.info(
                "Os valores foram modificados. "
                "O diagnostico do registro original "
                "nao pode ser atribuido a este "
                "novo cenario simulado."
            )

        else:

            classe_original = (
                0 if rotulo_original == "B" else 1
            )

            st.write(
                "**Diagnostico do registro original:**",
                "Benigno" if classe_original == 0
                else "Maligno"
            )

            st.write(
                "**Previsao coincide com o registro:**",
                "Sim" if previsao == classe_original
                else "Nao"
            )

        st.caption(
            "Resultado exclusivamente experimental. "
            "As probabilidades nao sao uma estimativa "
            "clinica validada e nao devem orientar "
            "decisoes medicas."
        )

        print(
            "Simulacao executada com sucesso."
        )
        mostrar_explicacao_shap(
            modelo,
            X_entrada,
            probabilidade_maligna
        )

        # PASSO 37 - Gerar o PDF com a mesma entrada da previsao.
        try:
            st.session_state["relatorio_pdf"] = gerar_relatorio_pdf(
                modelo=modelo,
                X_entrada=X_entrada.copy(),
                limiar=limiar
            )

            st.success("Relatorio PDF preparado com sucesso!")

        except Exception:
            logger.exception("Falha ao gerar o relatório PDF")
            st.session_state.pop("relatorio_pdf", None)
            st.warning(
                "Não foi possível preparar o PDF desta simulação. "
                "Confira o terminal da aplicação para diagnóstico."
            )

    # Botao de download fora do st.form e depois da geracao do PDF.
    if st.session_state.get("relatorio_pdf") is not None:
        st.download_button(
            label="Baixar relatorio da simulacao (PDF)",
            data=st.session_state["relatorio_pdf"],
            file_name="relatorio_simulacao_fiap.pdf",
            mime="application/pdf",
            on_click="ignore",
            key="download_relatorio_pdf"
        )


# PASSO 36 - Guia de interpretação do SHAP

with st.expander(
    "Como interpretar a explicação da inteligência artificial?",
    expanded=False
):

    st.markdown(
        """
        ### Entendendo o resultado

        O SHAP permite identificar quais características
        influenciaram a classificação realizada pelo modelo.

        **Contribuições positivas (vermelho):**
        Aumentam a pontuação associada à classe maligna.

        **Contribuições negativas (azul):**
        Diminuem a pontuação associada à classe maligna,
        favorecendo a classificação benigna.

        ### O que significam os valores?

        Os números SHAP apresentados estão na escala
        de log-odds, e não em porcentagem.

        A soma das contribuições com o valor de referência
        reconstrói a pontuação do modelo, que é convertida
        em probabilidade pela função logística.

        ### Limitações importantes

        - Uma contribuição SHAP não demonstra causalidade.
        - A explicação descreve o comportamento do modelo.
        - Uma probabilidade elevada não representa certeza.
        - Os resultados não substituem exames, avaliação
          médica ou diagnóstico clínico.

        **Finalidade:** demonstração acadêmica de
        inteligência artificial explicável.
        """
    )








# 15. Rodape
st.divider()

st.caption(
    "Tech Challenge FIAP - Fase 1 | "
    "Inteligencia Artificial Explicavel | "
    "Uso exclusivamente educacional"
)
