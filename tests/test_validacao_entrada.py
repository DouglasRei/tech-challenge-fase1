"""Execute na raiz: python -m unittest discover -s tests -v"""
import unittest
import numpy as np
import pandas as pd
from app.validacao_entrada import validar_amostra, EntradaInvalida

COLUNAS = [f"caracteristica_{i}" for i in range(30)]
MAXIMOS = pd.Series([10.0] * 30, index=COLUNAS)

def exemplo():
    return pd.DataFrame([[2.5] * 30], columns=COLUNAS)

class TestValidacaoEntrada(unittest.TestCase):
    def test_amostra_valida(self):
        obtido = validar_amostra(exemplo(), COLUNAS, MAXIMOS)
        self.assertEqual(obtido.shape, (1, 30))
        self.assertEqual(obtido.iloc[0, 0], 2.5)

    def test_nao_altera_o_original(self):
        original = exemplo()
        resultado = validar_amostra(original, COLUNAS, MAXIMOS)
        resultado.iloc[0, 0] = 3.0
        self.assertEqual(original.iloc[0, 0], 2.5)

    def test_coluna_ausente(self):
        with self.assertRaises(EntradaInvalida):
            validar_amostra(exemplo().iloc[:, :-1], COLUNAS, MAXIMOS)

    def test_duas_amostras(self):
        x = pd.concat([exemplo(), exemplo()], ignore_index=True)
        with self.assertRaises(EntradaInvalida):
            validar_amostra(x, COLUNAS, MAXIMOS)

    def test_ordem_trocada(self):
        colunas_trocadas = [COLUNAS[1], COLUNAS[0], *COLUNAS[2:]]
        with self.assertRaises(EntradaInvalida):
            validar_amostra(exemplo()[colunas_trocadas], COLUNAS, MAXIMOS)

    def test_valor_nan(self):
        x = exemplo()
        x.iloc[0, 2] = np.nan
        with self.assertRaises(EntradaInvalida):
            validar_amostra(x, COLUNAS, MAXIMOS)

    def test_valor_infinito(self):
        x = exemplo()
        x.iloc[0, 2] = np.inf
        with self.assertRaises(EntradaInvalida):
            validar_amostra(x, COLUNAS, MAXIMOS)

    def test_valor_negativo(self):
        x = exemplo()
        x.iloc[0, 2] = -1
        with self.assertRaises(EntradaInvalida):
            validar_amostra(x, COLUNAS, MAXIMOS)

    def test_valor_extremo(self):
        x = exemplo()
        x.iloc[0, 2] = 101.0  # > 10 vezes o máximo (10)
        with self.assertRaises(EntradaInvalida):
            validar_amostra(x, COLUNAS, MAXIMOS)

    def test_valor_no_limite(self):
        x = exemplo()
        x.iloc[0, 2] = 100.0  # <= limite permitido
        validar_amostra(x, COLUNAS, MAXIMOS)

    def test_valor_texto(self):
        x = exemplo().astype(object)
        x.iloc[0, 2] = 'texto'
        with self.assertRaises(EntradaInvalida):
            validar_amostra(x, COLUNAS, MAXIMOS)

    def test_valor_booleano(self):
        x = exemplo().astype(object)
        x.iloc[0, 2] = True
        with self.assertRaises(EntradaInvalida):
            validar_amostra(x, COLUNAS, MAXIMOS)

    def test_limites_na_ordem_errada(self):
        with self.assertRaises(EntradaInvalida):
            validar_amostra(exemplo(), COLUNAS, MAXIMOS.iloc[::-1])

    def test_limites_nan(self):
        limites = MAXIMOS.copy()
        limites.iloc[0] = np.nan
        with self.assertRaises(EntradaInvalida):
            validar_amostra(exemplo(), COLUNAS, limites)

if __name__ == '__main__':
    unittest.main()
