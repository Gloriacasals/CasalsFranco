"""
Module: arms/armbernoulli.py
Description: Contains the implementation of the ArmBernoulli class for the Bernoulli distribution arm.
"""

import numpy as np
from arms import Arm

class ArmBernoulli(Arm):
    def __init__(self, p: float):
        """
        Inicializa el brazo con distribución de Bernoulli.

        :param p: Probabilidad de éxito (recompensa = 1). Debe estar en [0, 1].
        """
        assert 0.0 <= p <= 1.0, "La probabilidad p debe estar entre 0 y 1."
        self.p = p

    def pull(self):
        """
        Genera una recompensa siguiendo una distribución de Bernoulli.
        Retorna 1 con probabilidad p, y 0 con probabilidad 1-p.

        :return: Recompensa obtenida (0 o 1).
        """
        # Usamos binomial con n=1 que es equivalente a Bernoulli
        return np.random.binomial(1, self.p)

    def get_expected_value(self) -> float:
        """
        Devuelve el valor esperado de la distribución Bernoulli.
        E[X] = p

        :return: Valor esperado (p).
        """
        return self.p

    def __str__(self):
        return f"ArmBernoulli(p={self.p:.2f})"

    @classmethod
    def generate_arms(cls, k: int):
        """
        Genera k brazos con probabilidades p únicas en el rango (0, 1).
        
        :param k: Número de brazos a generar.
        :return: Lista de brazos generados.
        """
        assert k > 0, "El número de brazos k debe ser mayor que 0."

        # Generar k valores únicos de p
        p_values = set()
        while len(p_values) < k:
            p = np.random.uniform(0.05, 0.95) # Evitamos 0 y 1 puros para evitar determinismo total
            p = round(p, 4)
            p_values.add(p)
        
        p_values = list(p_values)
        arms = [ArmBernoulli(p) for p in p_values]
        
        return arms
