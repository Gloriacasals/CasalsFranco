"""
Module: algorithms/epsilon_greedy.py
Description: Implementación del algoritmo epsilon-greedy para el problema de los k-brazos.

Author: Luis Daniel Hernández Molinero
Email: ldaniel@um.es
Date: 2025/01/29

This software is licensed under the GNU General Public License v3.0 (GPL-3.0),
with the additional restriction that it may not be used for commercial purposes.

For more details about GPL-3.0: https://www.gnu.org/licenses/gpl-3.0.html
"""

import numpy as np

from algorithms.algorithm import Algorithm

class EpsilonGreedy(Algorithm):

    def __init__(self, k: int, epsilon: float = 0.1):
        """
        Inicializa el algoritmo epsilon-greedy.

        :param k: Número de brazos.
        :param epsilon: Probabilidad de exploración (seleccionar un brazo al azar).
        :raises ValueError: Si epsilon no está en [0, 1].
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."

        super().__init__(k)
        self.epsilon = epsilon

   # def select_arm(self) -> int:
    #    """
     #   Selecciona un brazo basado en la política epsilon-greedy.

      #  :return: índice del brazo seleccionado.
       # """

        # Observa que para para epsilon=0 solo selecciona un brazo y no hace un primer recorrido por todos ellos.
        # ¿Podrías modificar el código para que funcione correctamente para epsilon=0?

        #if np.random.random() < self.epsilon:
            # Selecciona un brazo al azar
         #   chosen_arm = np.random.choice(self.k)
        #else:
            # Selecciona el brazo con la recompensa promedio estimada más alta
         #   chosen_arm = np.argmax(self.values)

        #return chosen_arm


def select_arm(self) -> int:
        """
        Selecciona un brazo basado en la política epsilon-greedy.
        Incluye un barrido inicial: si hay brazos sin probar, los selecciona primero.

        :return: índice del brazo seleccionado.
        """
        
        # 1. Barrido inicial: Si hay algún brazo que nunca se ha tocado, se selecciona
        # Esto garantiza que tengamos una estimación inicial para todos (evita el problema de epsilon=0)
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        # 2. Política Epsilon-Greedy estándar
        if np.random.random() < self.epsilon:
            # Exploración: Selecciona un brazo al azar
            chosen_arm = np.random.choice(self.k)
        else:
            # Explotación: Selecciona el brazo con mayor valor estimado
            # NOTA: np.argmax devuelve el primero en caso de empate.
            # en caso de empate en los valores máximos, deberíamos elegir al azar entre ellos.
            
            max_value = np.max(self.values)
            # Encontramos los índices de todos los brazos que tienen ese valor máximo
            best_arms = np.where(self.values == max_value)[0]
            
            # Si hay empate, elegimos uno al azar entre los mejores. Si no, coge el único.
            chosen_arm = np.random.choice(best_arms)

        return chosen_arm
