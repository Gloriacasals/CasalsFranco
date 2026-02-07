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

    def tau(self, r: float) -> float:
        """
        Función tau(r) usada en UCB2.

        :param r: época del brazo
        :return: número de tiradas recomendadas hasta época r
        """
        return int(np.ceil((1 + self.alpha) ** r))

    def bonus(self, arm: int) -> float:
        """
        Calcula el término de exploración UCB2 para un brazo.

        :param arm: índice del brazo
        :return: bonus de exploración
        """
        r_i = self.r[arm]

        tau_r = self.tau(r_i)
        tau_r_next = self.tau(r_i + 1)

        return np.sqrt(
            (1 + self.alpha) *
            np.log(np.e * self.t / tau_r) /
            (2 * tau_r_next)
        )

    def select_arm(self) -> int:
        """
        Selecciona un brazo usando la política UCB2.

        :return: índice del brazo seleccionado.
        """

        self.t += 1

        # 1. Barrido inicial: probar todos los brazos al menos una vez
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        # 2. Selección según UCB2
        ucb_values = np.zeros(self.k)

        for arm in range(self.k):
            ucb_values[arm] = self.values[arm] + self.bonus(arm)

        # Elegimos el brazo con mayor UCB (desempate aleatorio)
        max_value = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_value)[0]
        chosen_arm = np.random.choice(best_arms)

        # Actualizamos época del brazo elegido
        self.r[chosen_arm] += 1

        return chosen_arm
