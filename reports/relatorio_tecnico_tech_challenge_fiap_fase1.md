# Relatório Técnico — Tech Challenge FIAP | Fase 1

## Classificação experimental de amostras do Breast Cancer Wisconsin com Inteligência Artificial Explicável

**Natureza do projeto:** acadêmica e experimental  
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Modelo selecionado:** Regressão Logística  
**Classe positiva:** 1 — Maligno  
**Limiar de classificação:** 0,50  
**Características utilizadas:** 30

> **Aviso de uso responsável:** este projeto não é um dispositivo médico, não realiza diagnóstico clínico validado e não deve fundamentar decisões de saúde. As probabilidades produzidas representam saídas de um modelo treinado sobre um conjunto de dados público.

## 1. Resumo executivo

O projeto implementou uma solução de aprendizado de máquina supervisionado para classificação experimental de amostras como benignas ou malignas, utilizando 30 características morfológicas do dataset Breast Cancer Wisconsin (Diagnostic). Após a comparação de estratégias de modelagem, foi selecionado um pipeline de Regressão Logística com padronização de características. A solução inclui um simulador interativo em Streamlit, explicações locais por SHAP, geração de relatório em PDF, testes automatizados e empacotamento Docker.

Na avaliação final sobre **86 amostras reservadas**, o modelo obteve **100,00% de acurácia, precisão, recall, F1-score e especificidade**, com 54 verdadeiros negativos, 32 verdadeiros positivos e nenhum erro observado nesse conjunto. Esses resultados são específicos dessa amostra de teste: **não** demonstram validade clínica, desempenho perfeito em outros dados ou garantia de generalização.

## 2. Objetivo

Desenvolver e documentar um classificador experimental reprodutível e interpretável, permitindo explorar o ciclo de trabalho de inteligência artificial: análise de dados, preparação, treinamento, escolha de modelo, avaliação, explicação e disponibilização de uma interface demonstrativa.

## 3. Base de dados e partições

O dataset contém **569 registros** e **30 características preditoras**. Os registros foram organizados em um conjunto de desenvolvimento de **483 amostras** e um conjunto de teste reservado de **86 amostras**. Na organização registrada para o desenvolvimento, aparecem **398 amostras de treinamento** e **85 de validação**.

O conjunto de teste reservado contém **54 registros benignos** e **32 malignos**. A classe positiva adotada é **maligno (1)**; a classe negativa é **benigno (0)**.

As métricas *out-of-fold* (OOF), utilizadas na avaliação comparativa durante o desenvolvimento, e as métricas de teste reservado são relatadas **separadamente**, porque representam etapas de avaliação diferentes.

## 4. Estratégia de modelagem e escolha do classificador

O modelo selecionado foi a **Regressão Logística**, organizada em um pipeline que contempla a **padronização das características** e a etapa de classificação. A saída probabilística referente à classe maligna é comparada ao limiar de **0,50** para obter a classe estimada.

Segundo o registro de decisão do projeto (`reports/decisao_modelo.json`), o critério de seleção priorizou o **maior F1-score e a maior acurácia entre as estratégias avaliadas**, considerando também a ocorrência de **zero falsos positivos nas previsões OOF**. A decisão foi formulada para avaliação acadêmica; o arquivo registra expressamente que a estratégia não deve ser alterada em função dos resultados do teste final.

Essa escolha deve ser interpretada em conjunto com as demais métricas, especialmente os falsos negativos, que correspondem a registros malignos classificados incorretamente como benignos.

## 5. Resultados da avaliação durante o desenvolvimento (OOF)

O registro de seleção documenta os seguintes indicadores *out-of-fold*:

| Métrica | Resultado OOF |
|---|---:|
| Acurácia | **98,24%** |
| F1-score | **97,58%** |
| Falsos positivos | **0** |
| Falsos negativos | **7** |

Os **sete falsos negativos** constituem uma limitação relevante: mesmo com desempenho agregado elevado, a avaliação de desenvolvimento registrou erros na identificação da classe maligna. Esse achado reforça o caráter experimental da solução e impede interpretações de segurança clínica.

## 6. Resultados no teste reservado

A avaliação final foi registrada no arquivo `reports/metricas_finais_teste.json` e compreendeu **86 amostras**, mantidas como conjunto de teste reservado.

### 6.1. Matriz de confusão

| Classe real / Classe prevista | Benigno (0) | Maligno (1) |
|---|---:|---:|
| **Benigno (0)** | **54** (verdadeiros negativos) | **0** (falsos positivos) |
| **Maligno (1)** | **0** (falsos negativos) | **32** (verdadeiros positivos) |

### 6.2. Métricas finais

| Métrica | Resultado no teste reservado |
|---|---:|
| Acurácia | **100,00%** |
| Precisão | **100,00%** |
| Recall / Sensibilidade | **100,00%** |
| F1-score | **100,00%** |
| Especificidade | **100,00%** |

**Interpretação:** o modelo classificou corretamente todos os 86 exemplos específicos do teste reservado. A ausência de erros nesse grupo não elimina a possibilidade de erros em novas amostras, populações distintas ou condições diferentes de coleta. A dimensão do teste e a inexistência de validação clínica externa limitam as conclusões.

**Evidência gráfica:** `reports/figures/10_matriz_confusao_final_teste.png`.

## 7. Inteligência artificial explicável — SHAP

O simulador utiliza explicações SHAP para decompor a pontuação de cada previsão em contribuições de características. Para a Regressão Logística empregada, as contribuições apresentadas na aplicação são interpretadas na escala de **log-odds**, e não como percentuais de probabilidade.

Contribuições positivas aumentam a pontuação associada à classe **maligna**; contribuições negativas a reduzem, favorecendo a classe **benigna**. A soma das contribuições com o valor de referência reconstrói a pontuação explicada do modelo; a conversão pela função logística permite relacioná-la à probabilidade exibida.

As explicações revelam **como o modelo calculou uma classificação**, mas não demonstram causalidade biológica, validade de um diagnóstico ou certeza individual. As evidências produzidas incluem a análise global de importância SHAP e gráficos de explicação de exemplos benigno e maligno:

- `reports/figures/12_importancia_global_shap.png`
- `reports/figures/13_shap_beeswarm.png`
- `reports/figures/14_shap_waterfall_maligno.png`
- `reports/figures/15_shap_waterfall_benigno.png`

## 8. Aplicação, engenharia e reprodutibilidade

A interface demonstrativa foi construída com **Streamlit** e possibilita iniciar simulações com exemplos públicos, editar características, executar previsões e inspecionar explicações SHAP. O projeto também incorpora um módulo de **geração de relatório PDF** e procedimentos de validação de entrada.

Os artefatos de reprodução incluem o código-fonte, o arquivo do pipeline treinado (`models/modelo_final_logistica.joblib`), a documentação, a declaração de dependências, os notebooks de desenvolvimento e um `Dockerfile`. Foram realizadas verificações em contêiner e uma reconstrução a partir de cópia limpa/arquivo de entrega. Os registros de terminal apresentados confirmam **14 testes automatizados aprovados** nesse ambiente.

A disponibilização do código e do modelo facilita a auditoria e a reprodução técnica, mas não substitui verificações independentes das métricas nem validação externa para outros contextos.

## 9. Limitações e uso responsável

Os resultados decorrem de **um dataset público**, e as métricas finais observadas abrangem **um único conjunto reservado de 86 registros**. O desempenho pode variar diante de outros dados, diferenças de distribuição, novos instrumentos de medição ou situações não representadas na base original.

O registro OOF contém **sete falsos negativos**, dado particularmente importante para a interpretação dos riscos de erro na classe maligna. A obtenção de 100% no teste reservado não invalida esses resultados anteriores e não comprova ausência futura de erros.

O SHAP oferece transparência sobre as contribuições do modelo, mas não comprova relações causais. Não foi demonstrada validação clínica independente, avaliação prospectiva, estudo de segurança ou aprovação regulatória. Portanto, **o sistema deve permanecer restrito à demonstração acadêmica e à pesquisa educacional**.

## 10. Conclusão

A Fase 1 entregou um pipeline de classificação experimental com 30 variáveis, seleção documentada de Regressão Logística, avaliação OOF, teste reservado, explicações SHAP, interface Streamlit, relatórios PDF, testes automatizados e ambiente Docker. A avaliação final não registrou erros nas 86 amostras reservadas, enquanto a avaliação OOF apresentou acurácia de 98,24%, F1-score de 97,58% e sete falsos negativos.

A contribuição principal do projeto é **a integração reprodutível entre aprendizado de máquina, explicabilidade e engenharia de software**, acompanhada da explicitação das limitações de interpretação. Os resultados não autorizam utilização em decisões clínicas.

## 11. Fontes de evidência do repositório

| Evidência | Arquivo ou diretório |
|---|---|
| Métricas do teste reservado | `reports/metricas_finais_teste.json` |
| Justificativa de seleção do modelo | `reports/decisao_modelo.json` |
| Notebook de treinamento e comparação | `notebooks/03_treinamento_modelos.ipynb` |
| Artefato treinado | `models/modelo_final_logistica.joblib` |
| Interface e geração de PDF | `app/app.py` e `app/relatorio_pdf.py` |
| Validação de dados de entrada | `app/validacao_entrada.py` |
| Testes automatizados | `tests/test_validacao_entrada.py` |
| Ambiente de execução | `Dockerfile`, `requirements.txt`, `requirements-lock.txt` |
| Gráficos e matrizes de avaliação | `reports/figures/` |

---

**Nota metodológica:** texto elaborado a partir das métricas e dos registros de decisão exibidos nas capturas de tela do projeto, sem reexecução independente do treinamento ou auditoria dos arquivos originais.
