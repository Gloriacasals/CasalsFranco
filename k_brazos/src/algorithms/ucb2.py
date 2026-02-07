import numpy as np

from algorithms.algorithm import Algorithm


class UCB2(Algorithm):
    """
    Algoritmo Upper Confidence Bound 2 (UCB2).
    Basado en exploración por épocas controlada por el parámetro alpha.
    """

    def __init__(self, k: int, alpha: float = 0.5):
        """
        Inicializa el algoritmo UCB2.

        :param k: Número de brazos.
        :param alpha: Parámetro de control de exploración (alpha > 0).
        """
        assert alpha > 0, "El parámetro alpha debe ser positivo."

        super().__init__(k)

        self.alpha = alpha
        self.r = np.zeros(k)  # contador de épocas por brazo
        self.t = 0            # tiempo global

        # Variables para el mecanismo de bloques de UCB2
        self.current_arm = None   # brazo que estamos jugando actualmente
        self.remaining = 0        # cuántas jugadas quedan en el bloque actual

    def tau(self, r: float) -> float:
        """
        Función tau(r) usada en UCB2.

        :param r: época del brazo
        :return: número de tiradas recomendadas hasta época r
        """
        return int(np.ceil((1 + self.alpha) ** r))

    def bonus(self, arm: int) -> float:

        r_i = self.r[arm]

        tau_r = self.tau(r_i)
        tau_r_next = self.tau(r_i + 1)

        inside_log = max((np.e * self.t) / tau_r, 1.000001)

        return np.sqrt(
            (1 + self.alpha) *
            np.log(inside_log) /
            (2 * tau_r_next)
        )


    def select_arm(self) -> int:

        self.t += 1

        # 1. Barrido inicial: probar todos los brazos al menos una vez
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        # 2. Si estamos dentro de un bloque, repetimos el brazo actual
        if self.remaining > 0:
            self.remaining -= 1
            return self.current_arm

        # 3. Si no, calculamos UCB2 para todos los brazos
        ucb_values = np.zeros(self.k)

        for arm in range(self.k):
            ucb_values[arm] = self.values[arm] + self.bonus(arm)

        # Elegimos el mejor brazo (desempate aleatorio)
        max_value = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_value)[0]
        chosen_arm = np.random.choice(best_arms)

        # 4. Actualizamos época del brazo elegido
        r_i = self.r[chosen_arm]
        self.r[chosen_arm] += 1

        # 5. Calculamos tamaño del bloque
        block_len = self.tau(r_i + 1) - self.tau(r_i)

        # Guardamos brazo actual y cuántas veces repetirlo
        self.current_arm = chosen_arm
        self.remaining = block_len - 1

        return chosen_arm
