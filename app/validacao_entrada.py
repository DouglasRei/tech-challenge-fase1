"""Validação de entradas para o simulador acadêmico do Tech Challenge.

Os limites numéricos são uma proteção operacional contra entradas extremas,
não limites clínicos nem critérios diagnósticos.
"""

import numpy as np
import pandas as pd


class EntradaInvalida(ValueError):
    """O formulário recebeu dados incompatíveis com o modelo."""


def validar_amostra(X_entrada, colunas_esperadas, maximos_referencia):
    """Retorna uma cópia segura de uma amostra com 30 características.

    Exige ordem exata das colunas, um registro, números finitos e
    não negativos. Por segurança operacional, cada valor deve ser
    <= 10 vezes o maior valor observado no dataset de referência.
    """
    colunas = list(colunas_esperadas)
    if len(colunas) != 30 or len(set(colunas)) != 30:
        raise EntradaInvalida("Esquema de características inválido.")

    if not isinstance(X_entrada, pd.DataFrame):
        raise EntradaInvalida("A entrada deve ser uma tabela de dados.")

    if X_entrada.shape != (1, 30):
        raise EntradaInvalida("A entrada deve conter 1 amostra e 30 características.")

    if list(X_entrada.columns) != colunas:
        raise EntradaInvalida("A ordem ou os nomes das características estão incorretos.")

    if any(isinstance(valor, (bool, np.bool_))
           for valor in X_entrada.to_numpy().ravel()):
        raise EntradaInvalida("Valores booleanos não são permitidos nas características.")

    try:
        amostra = X_entrada.astype('float64')
        valores = amostra.to_numpy(dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise EntradaInvalida("Todas as características devem ser numéricas.") from exc

    if not np.isfinite(valores).all():
        raise EntradaInvalida("Há valores ausentes, infinitos ou inválidos.")

    negativos = np.where(valores[0] < 0)[0]
    if len(negativos):
        raise EntradaInvalida(
            f"A característica '{colunas[int(negativos[0])]}' não pode ser negativa."
        )

    if not isinstance(maximos_referencia, pd.Series):
        raise EntradaInvalida("Os limites de referência não foram configurados.")
    if list(maximos_referencia.index) != colunas:
        raise EntradaInvalida("Os limites de referência não seguem a ordem do modelo.")

    try:
        maximos = maximos_referencia.to_numpy(dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise EntradaInvalida("Limites de referência inválidos.") from exc

    if not np.isfinite(maximos).all() or np.any(maximos < 0):
        raise EntradaInvalida("Limites de referência inválidos.")

    # Evitar overflow no cálculo do limite de segurança.
    limites = np.minimum(maximos, np.finfo(float).max / 10.0) * 10.0
    limites = np.maximum(limites, 1e-8)
    excessos = np.where(valores[0] > limites)[0]
    if len(excessos):
        nome = colunas[int(excessos[0])]
        raise EntradaInvalida(
            f"O valor de '{nome}' excede o limite operacional do simulador."
        )

    return amostra
