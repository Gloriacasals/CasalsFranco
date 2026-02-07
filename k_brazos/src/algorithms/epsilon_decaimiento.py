import numpy as np

from algorithms.algorithm import Algorithm


class EpsilonDecay(Algorithm):
    """
    Algoritmo epsilon-greedy con decaimiento.
    La probabilidad de exploración disminuye con el tiempo.
    """

    def __init__(self, k: int, epsilon: float = 1.0, decay: float = 0.001):
        """
        Inicializa el algoritmo epsilon-decay.

        :param k: Número de brazos.
        :param epsilon: Valor inicial de epsilon.
        :param decay: Factor de decaimiento (controla la velocidad de reducción).
        """
        assert 0 <= epsilon <= 1, "El parámetro epsilon debe estar entre 0 y 1."
        assert decay > 0, "El parámetro decay debe ser positivo."

        super().__init__(k)

        self.epsilon0 = epsilon
        self.decay = decay
        self.t = 0  # contador de pasos

    def select_arm(self) -> int:
        """
        Selecciona un brazo usando epsilon-greedy con decaimiento.

        :return: índice del brazo seleccionado.
        """

        # Incrementamos el paso temporal
        self.t += 1

        # Calculamos epsilon actual
        epsilon_t = self.epsilon0 / (1 + self.decay * self.t)

        # 1. Barrido inicial: probar todos los brazos al menos una vez
        for i in range(self.k):
            if self.counts[i] == 0:
                return i

        # 2. Exploración vs Explotación
        if np.random.random() < epsilon_t:
            # Exploración
            chosen_arm = np.random.choice(self.k)
        else:
            # Explotación con desempate aleatorio
            max_value = np.max(self.values)
            best_arms = np.where(self.values == max_value)[0]
            chosen_arm = np.random.choice(best_arms)

        return chosen_arm
