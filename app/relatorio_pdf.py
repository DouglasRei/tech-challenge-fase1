from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


def gerar_relatorio_pdf(modelo, X_entrada, limiar=0.50):
    """Cria um PDF acadêmico de uma única previsão."""
    if not isinstance(X_entrada, pd.DataFrame) or len(X_entrada) != 1:
        raise ValueError("X_entrada deve ser um DataFrame com uma amostra.")

    colunas = list(getattr(modelo, "feature_names_in_", X_entrada.columns))
    X = X_entrada.loc[:, colunas].copy()
    valores = X.to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("Existem valores inválidos na amostra.")

    padronizador = next(
        (etapa for _, etapa in modelo.steps
         if isinstance(etapa, StandardScaler)), None
    )
    classificador = next(
        (etapa for _, etapa in modelo.steps
         if isinstance(etapa, LogisticRegression)), None
    )
    if padronizador is None or classificador is None:
        raise ValueError("Pipeline incompatível com o relatório SHAP.")

    indice_maligno = list(modelo.classes_).index(1)
    probabilidade = float(modelo.predict_proba(X)[0, indice_maligno])
    classificacao = "Maligno" if probabilidade >= limiar else "Benigno"

    # Mesma referência: média do conjunto usado para ajustar o StandardScaler.
    # Na escala padronizada essa referência é zero.
    X_padronizado = padronizador.transform(X)
    explicacao = shap.LinearExplainer(
        classificador, np.zeros((1, len(colunas)))
    )(X_padronizado)
    contribuicoes = np.asarray(explicacao.values)[0]
    referencia = float(np.asarray(explicacao.base_values).reshape(-1)[0])
    logit = float(classificador.decision_function(X_padronizado)[0])
    if not np.isclose(referencia + contribuicoes.sum(), logit, atol=1e-5):
        raise ValueError("Contribuições SHAP não reconstruíram a saída.")

    principais = sorted(
        zip(colunas, contribuicoes),
        key=lambda item: abs(item[1]), reverse=True
    )[:10]

    memoria = BytesIO()
    documento = SimpleDocTemplate(
        memoria, pagesize=A4,
        leftMargin=42, rightMargin=42, topMargin=45, bottomMargin=45
    )
    estilo = getSampleStyleSheet()
    elementos = [
        Paragraph("Tech Challenge FIAP - Fase 1", estilo["Title"]),
        Paragraph("Relatório de simulação e explicabilidade", estilo["Heading2"]),
        Paragraph(
            f"Gerado em: {datetime.now():%d/%m/%Y %H:%M}",
            estilo["Normal"]
        ),
        Spacer(1, 15),
        Paragraph("1. Resultado do modelo", estilo["Heading2"]),
        Paragraph(f"Classificação estimada: <b>{classificacao}</b>", estilo["Normal"]),
        Paragraph(f"Probabilidade de malignidade: <b>{probabilidade * 100:.4f}%</b> (arredondada)", estilo["Normal"]),
        Paragraph(f"Limiar de classificação: {limiar * 100:.0f}%", estilo["Normal"]),
        Paragraph("Modelo: Regressão Logística; 30 características morfológicas.", estilo["Normal"]),
        Spacer(1, 12),
        Paragraph("2. Interpretação SHAP", estilo["Heading2"]),
        Paragraph(
            "As contribuições estão na escala log-odds. Valores positivos "
            "favorecem a classificação maligna; valores negativos favorecem a benigna. "
            "Elas explicam a saída do modelo e não demonstram causalidade.",
            estilo["BodyText"]
        ),
        Spacer(1, 8),
    ]
    tabela_shap = [["Característica", "SHAP (log-odds)"]]
    tabela_shap += [[escape(nome), f"{valor:+.4f}"] for nome, valor in principais]
    tabela = Table(tabela_shap, colWidths=[335, 175], repeatRows=1)
    tabela.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    elementos += [tabela, Spacer(1, 14),
        Paragraph("3. Limitações", estilo["Heading2"]),
        Paragraph(
            "Protótipo acadêmico. Este resultado não é um diagnóstico médico, "
            "não equivale a uma probabilidade clínica validada e não deve "
            "orientar decisões de saúde. Um resultado numérico próximo de "
            "100% não representa certeza.", estilo["BodyText"]
        ),
        PageBreak(),
        Paragraph("4. Características utilizadas", estilo["Heading2"]),
    ]
    tabela_entrada = [["Variável", "Valor informado"]]
    tabela_entrada += [
        [escape(nome), f"{float(valor):.6g}"]
        for nome, valor in X.iloc[0].items()
    ]
    tabela2 = Table(tabela_entrada, colWidths=[335, 175], repeatRows=1)
    tabela2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    elementos.append(tabela2)
    documento.build(elementos)
    return memoria.getvalue()


