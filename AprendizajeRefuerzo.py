"""
ReinforcementLearning.py
Implementación de Deep Q-Learning (DQN) para CartPole usando TensorFlow/Keras

Autor: Sistema de ML Supervisado
Fecha: 2025
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Para usar matplotlib sin GUI
import matplotlib.pyplot as plt
import pickle
import os
from collections import deque
import base64
from io import BytesIO

# TensorFlow y Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suprimir warnings de TensorFlow
import tensorflow as tf
import keras
from keras import layers
from keras import metrics

# Gym para el entorno
import gymnasium as gym


class DQNAgent:
    """
    Agente de Deep Q-Learning para el entorno CartPole.

    Parámetros:
    -----------
    state_size : int
        Tamaño del espacio de estados (4 para CartPole)
    action_size : int
        Número de acciones posibles (2 para CartPole: izquierda/derecha)
    learning_rate : float
        Tasa de aprendizaje (α)
    gamma : float
        Factor de descuento (γ)
    epsilon : float
        Tasa de exploración inicial (ε)
    epsilon_decay : float
        Decaimiento de epsilon por episodio
    epsilon_min : float
        Valor mínimo de epsilon
    """

    def __init__(
        self,
        state_size=4,
        action_size=2,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        # Hiperparámetros
        self.gamma = gamma  # Factor de descuento
        self.epsilon = epsilon  # Tasa de exploración
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.batch_size = 32

        # Red neuronal
        self.model = self._build_model()

        # Historial de entrenamiento
        self.rewards_history = []
        self.epsilon_history = []
        self.avg_rewards_history = []

    def _build_model(self):
        """
        Construye la red neuronal para aproximar la función Q.

        Arquitectura:
        - Capa de entrada: state_size neuronas
        - Capa oculta 1: 24 neuronas, activación ReLU
        - Capa oculta 2: 24 neuronas, activación ReLU
        - Capa de salida: action_size neuronas, activación lineal
        """
        model = keras.Sequential(
            [
                layers.Dense(24, input_dim=self.state_size, activation="relu"),
                layers.Dense(24, activation="relu"),
                layers.Dense(self.action_size, activation="linear"),
            ]
        )

        model.compile(
            loss="mse",
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
        )

        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Almacena una experiencia en la memoria de replay.

        Parameters:
        -----------
        state : array
            Estado actual
        action : int
            Acción tomada
        reward : float
            Recompensa recibida
        next_state : array
            Estado siguiente
        done : bool
            Si el episodio terminó
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selecciona una acción usando política epsilon-greedy.

        Con probabilidad epsilon: acción aleatoria (exploración)
        Con probabilidad 1-epsilon: acción según Q-value (explotación)

        Parameters:
        -----------
        state : array
            Estado actual

        Returns:
        --------
        int : Acción seleccionada
        """
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)

        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """
        Entrena la red neuronal usando experiencias aleatorias de la memoria.
        Implementa Experience Replay para romper correlaciones temporales.
        """
        if len(self.memory) < self.batch_size:
            return

        # Muestreo aleatorio de experiencias
        minibatch = np.random.choice(len(self.memory), self.batch_size, replace=False)

        states = []
        targets = []

        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]

            target = reward
            if not done:
                # Q-Learning: Q(s,a) = r + γ * max(Q(s',a'))
                target = reward + self.gamma * np.amax(
                    self.model.predict(next_state, verbose=0)[0]
                )

            # Obtener Q-values actuales
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            states.append(state[0])
            targets.append(target_f[0])

        # Entrenar la red
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, n_episodes=500, max_steps=500):
        """
        Entrena el agente en el entorno.

        Parameters:
        -----------
        env : gym.Env
            Entorno de Gymnasium
        n_episodes : int
            Número de episodios de entrenamiento
        max_steps : int
            Número máximo de pasos por episodio

        Returns:
        --------
        dict : Diccionario con historial de entrenamiento
        """
        print(f"Iniciando entrenamiento: {n_episodes} episodios")

        for episode in range(n_episodes):
            state, _ = env.reset()
            state = np.reshape(state, [1, self.state_size])

            total_reward = 0

            for step in range(max_steps):
                # Seleccionar acción
                action = self.act(state)

                # Ejecutar acción
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                next_state = np.reshape(next_state, [1, self.state_size])

                # Almacenar experiencia
                self.remember(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

                if done:
                    break

            # Entrenar con experiencias
            self.replay()

            # Guardar historial
            self.rewards_history.append(total_reward)
            self.epsilon_history.append(self.epsilon)

            # Calcular recompensa promedio de los últimos 100 episodios
            if len(self.rewards_history) >= 100:
                avg_reward = np.mean(self.rewards_history[-100:])
            else:
                avg_reward = np.mean(self.rewards_history)

            self.avg_rewards_history.append(avg_reward)

            # Mostrar progreso cada 50 episodios
            if (episode + 1) % 10 == 0:
                print(
                    f"Episodio {episode + 1}/{n_episodes} | "
                    f"Recompensa: {total_reward:.0f} | "
                    f"Promedio: {avg_reward:.2f} | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        print("✓ Entrenamiento completado")

        return {
            "rewards": self.rewards_history,
            "avg_rewards": self.avg_rewards_history,
            "epsilon": self.epsilon_history,
        }

    def save_model(self, filepath="models/dqn_cartpole.h5"):
        """Guarda el modelo entrenado."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Modelo guardado en: {filepath}")

    def load_model(self, filepath="models/dqn_cartpole.h5"):
        """Carga un modelo previamente entrenado."""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(
                filepath,
                custom_objects={
                    "mse": metrics.mean_squared_error  # O solo metrics.mse si está disponible
                },
            )
            self.epsilon = self.epsilon_min  # Modo explotación
            print(f"Modelo cargado desde: {filepath}")
            return True
        return False

    def save_training_data(self, filepath="models/training_data.pkl"):
        """Guarda el historial de entrenamiento."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "rewards": self.rewards_history,
            "avg_rewards": self.avg_rewards_history,
            "epsilon": self.epsilon_history,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Datos de entrenamiento guardados en: {filepath}")

    def load_training_data(self, filepath="models/training_data.pkl"):
        """Carga el historial de entrenamiento."""
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                data = pickle.load(f)
            self.rewards_history = data["rewards"]
            self.avg_rewards_history = data["avg_rewards"]
            self.epsilon_history = data["epsilon"]
            print(f"Datos de entrenamiento cargados desde: {filepath}")
            return True
        return False


# ============================================================================
# FUNCIONES PARA LA INTERFAZ WEB
# ============================================================================


def initialize_agent():
    """
    Inicializa el agente DQN.
    Carga el modelo si existe, si no, crea uno nuevo.
    """
    agent = DQNAgent()

    # Intentar cargar modelo existente
    if agent.load_model() and agent.load_training_data():
        return agent, True  # Modelo cargado

    return agent, False  # Modelo nuevo


def train_agent(n_episodes=500):
    """
    Entrena el agente y guarda el modelo.

    Parameters:
    -----------
    n_episodes : int
        Número de episodios de entrenamiento

    Returns:
    --------
    dict : Resultados del entrenamiento
    """
    # Crear entorno
    env = gym.make("CartPole-v1")

    # Crear agente
    agent = DQNAgent()

    # Entrenar
    results = agent.train(env, n_episodes=n_episodes)

    # Guardar modelo y datos
    agent.save_model()
    agent.save_training_data()

    env.close()

    return results


def test_agent(n_episodes=5):
    """
    Prueba el agente entrenado.

    Parameters:
    -----------
    n_episodes : int
        Número de episodios de prueba

    Returns:
    --------
    dict : Resultados de las pruebas
    """
    # Crear entorno
    env = gym.make("CartPole-v1")

    # Cargar agente
    agent = DQNAgent()
    if not agent.load_model():
        return {"error": "No hay modelo entrenado disponible"}

    test_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 500:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = np.reshape(next_state, [1, agent.state_size])
            total_reward += reward
            steps += 1

        test_rewards.append(total_reward)

    env.close()

    return {
        "rewards": test_rewards,
        "avg_reward": np.mean(test_rewards),
        "max_reward": np.max(test_rewards),
        "min_reward": np.min(test_rewards),
    }


def plot_training_progress():
    """
    Genera un gráfico del progreso de entrenamiento.

    Returns:
    --------
    str : Imagen en formato base64
    """
    agent = DQNAgent()
    if not agent.load_training_data():
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Gráfico 1: Recompensas
    episodes = range(1, len(agent.rewards_history) + 1)
    ax1.plot(
        episodes, agent.rewards_history, alpha=0.3, label="Recompensa por episodio"
    )
    ax1.plot(
        episodes,
        agent.avg_rewards_history,
        linewidth=2,
        label="Promedio móvil (100 episodios)",
    )
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Recompensa Total")
    ax1.set_title("Evolución de la Recompensa durante el Entrenamiento")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Epsilon
    ax2.plot(episodes, agent.epsilon_history, color="orange")
    ax2.set_xlabel("Episodio")
    ax2.set_ylabel("Epsilon (Tasa de Exploración)")
    ax2.set_title("Decaimiento de Epsilon")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convertir a base64
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return image_base64


def get_training_stats():
    """
    Obtiene estadísticas del entrenamiento actual.

    Returns:
    --------
    dict : Estadísticas de entrenamiento
    """
    agent = DQNAgent()

    if not agent.load_training_data():
        return {
            "trained": False,
            "n_episodes": 0,
            "final_epsilon": 1.0,
            "avg_reward": 0,
            "max_reward": 0,
        }

    return {
        "trained": True,
        "n_episodes": len(agent.rewards_history),
        "final_epsilon": agent.epsilon_history[-1] if agent.epsilon_history else 1.0,
        "avg_reward": (
            np.mean(agent.avg_rewards_history[-100:])
            if len(agent.avg_rewards_history) >= 100
            else 0
        ),
        "max_reward": np.max(agent.rewards_history) if agent.rewards_history else 0,
        "final_avg": agent.avg_rewards_history[-1] if agent.avg_rewards_history else 0,
    }


def reset_model():
    """
    Elimina el modelo entrenado y los datos guardados.

    Returns:
    --------
    bool : True si se eliminó correctamente
    """
    import os

    model_path = "models/dqn_cartpole.h5"
    data_path = "models/training_data.pkl"

    removed = False
    if os.path.exists(model_path):
        os.remove(model_path)
        removed = True
    if os.path.exists(data_path):
        os.remove(data_path)
        removed = True

    print(f"✓ Modelo reiniciado correctamente")
    return removed


def test_agent_with_trajectory(n_episodes=5):
    """
    Prueba el agente entrenado y retorna la trayectoria del último episodio.

    Parameters:
    -----------
    n_episodes : int
        Número de episodios de prueba

    Returns:
    --------
    tuple : (test_results, trajectory_data)
    """
    import gymnasium as gym
    import numpy as np

    # Crear entorno
    env = gym.make("CartPole-v1")

    # Cargar agente
    agent = DQNAgent()
    if not agent.load_model():
        return {"error": "No hay modelo entrenado disponible"}, None

    test_rewards = []
    last_states = []
    last_actions = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, agent.state_size])

        total_reward = 0
        done = False
        steps = 0

        episode_states = []
        episode_actions = []

        while not done and steps < 500:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Guardar estados y acciones del último episodio
            if episode == n_episodes - 1:
                episode_states.append(state[0])
                episode_actions.append(action)

            state = np.reshape(next_state, [1, agent.state_size])
            total_reward += reward
            steps += 1

        test_rewards.append(total_reward)

        # Guardar la trayectoria del último episodio
        if episode == n_episodes - 1:
            last_states = episode_states
            last_actions = episode_actions

    env.close()

    test_results = {
        "rewards": test_rewards,
        "avg_reward": np.mean(test_rewards),
        "max_reward": np.max(test_rewards),
        "min_reward": np.min(test_rewards),
    }

    trajectory_data = (
        {"states": last_states, "actions": last_actions} if last_states else None
    )

    return test_results, trajectory_data


def plot_trajectory(states, actions):
    """
    Genera un gráfico de la trayectoria del agente.

    Parameters:
    -----------
    states : list
        Lista de estados (posición, velocidad, ángulo, velocidad angular)
    actions : list
        Lista de acciones tomadas

    Returns:
    --------
    str : Imagen en formato base64
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO

    if not states or len(states) == 0:
        return None

    positions = [s[0] for s in states]
    angles = [s[2] for s in states]
    steps = list(range(len(states)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Posición del carrito
    ax1.plot(steps, positions, "b-", linewidth=2)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(y=2.4, color="red", linestyle="--", alpha=0.3, label="Límite")
    ax1.axhline(y=-2.4, color="red", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Paso")
    ax1.set_ylabel("Posición del Carrito")
    ax1.set_title("Posición del Carrito a lo Largo del Tiempo")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ángulo del poste
    ax2.plot(steps, angles, "r-", linewidth=2)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.fill_between(
        steps, -0.2095, 0.2095, alpha=0.2, color="green", label="Zona segura (±12°)"
    )
    ax2.axhline(y=0.2095, color="red", linestyle="--", alpha=0.3)
    ax2.axhline(y=-0.2095, color="red", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Paso")
    ax2.set_ylabel("Ángulo del Poste (radianes)")
    ax2.set_title("Ángulo del Poste a lo Largo del Tiempo")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return image_base64


# ============================================================================
# SCRIPT DE PRUEBA
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DEEP Q-LEARNING - CARTPOLE")
    print("=" * 70)

    # Entrenar agente
    print("\n1. Entrenando agente...")
    results = train_agent(n_episodes=100)

    # Probar agente
    print("\n2. Probando agente...")
    test_results = test_agent(n_episodes=10)
    print(f"Recompensa promedio en pruebas: {test_results['avg_reward']:.2f}")

    # Generar gráfico
    print("\n3. Generando gráfico...")
    plot_training_progress()
    print("Gráfico generado exitosamente")

    print("\n✓ Proceso completado")
