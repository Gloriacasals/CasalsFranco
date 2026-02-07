"""
Module: plotting/plotting.py
Description: Contiene funciones para generar gráficas de comparación de algoritmos.
"""

from typing import List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from algorithms import Algorithm, EpsilonGreedy, UCB1, Softmax, EpsilonDecay, UCB2

def get_algorithm_label(algo: Algorithm) -> str:
    """
    Genera una etiqueta descriptiva para el algoritmo incluyendo sus parámetros.

    :param algo: Instancia de un algoritmo.
    :type algo: Algorithm
    :return: Cadena descriptiva para el algoritmo.
    :rtype: str
    """
    label = type(algo).__name__
    if isinstance(algo, EpsilonGreedy):
        label += f" (epsilon={algo.epsilon})"
    elif isinstance(algo, EpsilonDecay):
        label += f" (epsilon={algo.epsilon0}, decay={algo.decay})"
    elif isinstance(algo, UCB1):
        label += f" ($c$={algo.c})"
    elif isinstance(algo, UCB2):
        label += f" (alpha={algo.alpha})"
    elif isinstance(algo, Softmax):
        label += f" ($\\tau$={algo.tau})"
    else:
        raise ValueError("El algoritmo debe ser de la clase Algorithm o una subclase.")
    return label


def plot_average_rewards(steps: int, rewards: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Recompensa Promedio vs Pasos de Tiempo.

    :param steps: Número de pasos de tiempo.
    :param rewards: Matriz de recompensas promedio.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    plt.figure(figsize=(14, 7))
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        plt.plot(range(steps), rewards[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('Recompensa Promedio', fontsize=14)
    plt.title('Recompensa Promedio vs Pasos de Tiempo', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.tight_layout()
    plt.show()




def plot_optimal_selections(steps: int, optimal_selections: np.ndarray, algorithms: List[Algorithm]):
    """
    Genera la gráfica de Porcentaje de Selección del Brazo Óptimo vs Pasos de Tiempo.
    :param steps: Número de pasos de tiempo.
    :param optimal_selections: Matriz de porcentaje de selecciones óptimas.
    :param algorithms: Lista de instancias de algoritmos comparados.
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 7))
    
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        # Multiplicamos por 100 para mostrar porcentaje
        plt.plot(range(steps), optimal_selections[idx] * 100, label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo', fontsize=14)
    plt.ylabel('% Selección Óptima', fontsize=14)
    plt.title('Porcentaje de Selección de la Acción Óptima', fontsize=16)
    plt.legend(title='Algoritmos')
    plt.ylim(-5, 105) # Fijar límites para ver bien el porcentaje
    plt.tight_layout()
    plt.show()




# DOS FUNCIONES ADICIONALES

def plot_regret(steps: int, regret_accumulated: np.ndarray, algorithms: List[Algorithm], *args):
    """
    Genera la gráfica de Regret Acumulado vs Pasos de Tiempo.
    
    :param steps: Número de pasos de tiempo.
    :param regret_accumulated: Matriz de regret acumulado (algoritmos x pasos).
    :param algorithms: Lista de instancias de algoritmos comparados.
    :param args: Opcional. Parámetros extra (ej. cotas teóricas).
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 7))
    
    for idx, algo in enumerate(algorithms):
        label = get_algorithm_label(algo)
        # Asumimos que regret_accumulated ya viene acumulado desde el experimento
        # Si viniera como regret instantáneo, haríamos np.cumsum(regret_accumulated[idx])
        plt.plot(range(steps), regret_accumulated[idx], label=label, linewidth=2)

    plt.xlabel('Pasos de Tiempo (t)', fontsize=14)
    plt.ylabel('Regret Acumulado R(T)', fontsize=14)
    plt.title('Evolución del Rechazo (Regret) Acumulado', fontsize=16)
    plt.legend(title='Algoritmos')
    
    # RIGOR: Si se pasa una cota teórica en args, se pinta (útil para UCB más adelante)
    if args:
        # Ejemplo: args[0] podría ser una función lambda t: C * log(t)
        pass 

    plt.tight_layout()
    plt.show()




# Para que plot_arm_statistics funcione, necesitamos modificar en bandit_experiment.ipynb la función run_experiment y que devuelva una estructura de datos arm_stats.
# El código del profesor solo guarda promedios temporales. Vamos a añadir acumuladores para contar cuántas veces se elige cada brazo en total 

def plot_arm_statistics(arm_stats: List[dict], algorithms: List[Algorithm], *args):
    """
    Genera gráficas de estadísticas de cada brazo: Promedio de ganancias y número de selecciones.
    
    :param arm_stats: Lista de diccionarios. Cada diccionario corresponde a un algoritmo y contiene:
                      - 'counts': array con veces que se eligió cada brazo.
                      - 'rewards': array con recompensa promedio obtenida por cada brazo.
                      - 'optimal_index': índice del brazo óptimo real.
    :param algorithms: Lista de algoritmos.
    """
    sns.set_theme(style="white", font_scale=1.1)
    
    num_algos = len(algorithms)
    # Creamos una figura con tantos subplots como algoritmos haya
    fig, axes = plt.subplots(num_algos, 1, figsize=(12, 6 * num_algos), constrained_layout=True)
    
    if num_algos == 1:
        axes = [axes] # Asegurar que sea iterable si solo hay 1 algoritmo

    for idx, algo in enumerate(algorithms):
        ax = axes[idx]
        stats = arm_stats[idx]
        
        counts = stats['counts']     # Cuántas veces se eligió cada brazo
        rewards = stats['rewards']   # Recompensa promedio real obtenida
        optimal_idx = stats['optimal_index']
        k = len(counts)
        
        # Colores: Verde para el óptimo, Azul para el resto
        colors = ['green' if i == optimal_idx else 'steelblue' for i in range(k)]
        
        # Crear gráfico de barras (Bar Chart)
        bars = ax.bar(range(k), rewards, color=colors, alpha=0.8)
        
        # Etiquetas del Eje X complejas como pide la guía
        # "Brazo i \n (N=Veces) \n [OPTIMO]"
        x_labels = []
        for i in range(k):
            label = f"Arm {i}\n(N={int(counts[i])})"
            if i == optimal_idx:
                label += "\n[OPTIMAL]"
            x_labels.append(label)
            
        ax.set_xticks(range(k))
        ax.set_xticklabels(x_labels)
        
        ax.set_ylabel('Recompensa Promedio Obtenida')
        ax.set_title(f'Estadísticas por Brazo - {get_algorithm_label(algo)}')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Añadir el valor exacto encima de cada barra
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.show()


