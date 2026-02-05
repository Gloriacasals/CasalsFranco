"""
Module: algorithms/softmax.py
Description: Implementación del algoritmo Softmax (Boltzmann Exploration).
"""

import numpy as np
from algorithms.algorithm import Algorithm

class Softmax(Algorithm):
    def __init__(self, k: int, tau: float = 0.1):
        """
        Inicializa el algoritmo Softmax.

        :param k: Número de brazos.
        :param tau: Temperatura. Controla la aleatoriedad de la elección.
                    Debe ser > 0.
        """
        assert tau > 0, "La temperatura tau debe ser mayor que 0."
        super().__init__(k)
        self.tau = tau

    def select_arm(self) -> int:
        """
        Selecciona un brazo basándose en la distribución de probabilidad de Boltzmann.
        """
        
        # Para estabilidad numérica:
        # Restamos el valor máximo para evitar overflow en la exponencial (e^1000 da error).
        # Matemáticamente no cambia las probabilidades relativas.
        z = (self.values - np.max(self.values)) / self.tau
        
        # Calcular numeradores
        exp_values = np.exp(z)
        
        # Calcular probabilidades (Softmax)
        probs = exp_values / np.sum(exp_values)
        
        # Seleccionar brazo basado en las probabilidades calculadas
        chosen_arm = np.random.choice(self.k, p=probs)
        
        return chosen_arm
