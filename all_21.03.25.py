import os
import io
import csv
import math
import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from brian2 import *
from mpl_toolkits.mplot3d import Axes3D
import time
import networkx as nx
from scipy.signal import resample

# Установка параметров вывода NumPy
np.set_printoptions(threshold=np.inf)

# -- Создание/проверка директории для сохранения результатов --
directory_path = 'results_ext_test1'
if os.path.exists(directory_path):
    print(f"Директория '{directory_path}' существует")
else:
    os.mkdir(directory_path)
    print(f"Директория '{directory_path}' создана")

# -- Инициализация параметров сети --
n_neurons = 500
num_of_clusters = 2
cluster_sizes = [250, 250]

# Основные диапазоны параметров
p_within_values = np.arange(0.15, 0.16, 0.05)
p_input_values = np.arange(0.1, 0.21, 0.05)

num_tests = 10
J = 1
J2 = 1
rate_tick_step = 50
t_range = [0, 1000] # in ms
rate_range = [0, 200] # in Hz
max_rate_on_graph = 10
time_window_size = 1000  # in ms
refractory_period = 10*ms # -> max freq=100Hz
use_stdp_values = [False]

# Для осцилляций/стимулов
oscillation_frequencies = [10]
I0_values = [1000] # pA

measure_names = [
    "degree",  
    "betweenness",
    "closeness",
    "random",
    "eigenvector",
    "percolation",
    "harmonic",
]


# measure_names = [
#     "degree",
#     "random"
# ]

boost_factor_list = [1.8, 100, 50, 0, 20, 30, 30]


simulation_times = [5000] # in ms


# Определение границ кластеров на основе размеров кластеров
def fram(cluster_sizes):
    frames = np.zeros(len(cluster_sizes))
    frames[0] = cluster_sizes[0]
    for i in range(1, len(cluster_sizes)):
        frames[i] = frames[i - 1] + cluster_sizes[i]
    return frames

# Проверка принадлежности узла к определенному кластеру
def clcheck(a, cluster_sizes):
    frames = fram(cluster_sizes)
    if a >= 0 and a < frames[0]:
        return 0
    else:
        for i in range(len(frames) - 1):
            if a >= frames[i] and a < frames[i + 1]:
                return i + 1
        return len(frames) - 1

# -- Формируем вектор меток кластеров --
cluster_labels = []
for i in range(n_neurons):
    cluster_labels.append(clcheck(i, cluster_sizes))

def measure_connectivity(C, cluster_sizes):
    """
    Функция для вычисления фактических долей внутрикластерных (p_in_measured)
    и межкластерных (p_out_measured) связей по итоговой матрице C.
    """
    n_neurons = C.shape[0]
    labels = np.empty(n_neurons, dtype=int)
    start = 0
    for idx, size in enumerate(cluster_sizes):
        labels[start:start + size] = idx
        start += size

    intra_possible = 0
    intra_actual = 0
    inter_possible = 0
    inter_actual = 0

    # Перебор пар нейронов (i < j)
    for i in range(n_neurons):
        for j in range(i+1, n_neurons):
            if labels[i] == labels[j]:
                intra_possible += 1
                if C[i, j]:
                    intra_actual += 1
            else:
                inter_possible += 1
                if C[i, j]:
                    inter_actual += 1

    p_in_measured = intra_actual / intra_possible if intra_possible > 0 else 0
    p_out_measured = inter_actual / inter_possible if inter_possible > 0 else 0
    return p_in_measured, p_out_measured

def generate_sbm_with_high_centrality(
    n, cluster_sizes, p_intra, p_inter,
    target_cluster_index, proportion_high_centrality=0.2,
    centrality_type="degree", boost_factor=2
):
    """
    Generate an SBM where a proportion of nodes in a fixed cluster have high centrality.

    Parameters:
        n: Total number of nodes.
        cluster_sizes: List of sizes for each cluster.
        p_intra: Probability of intra-cluster edges.
        p_inter: Probability of inter-cluster edges.
        target_cluster_index: Index of the cluster to modify (0-based).
        proportion_high_centrality: Proportion of nodes in the target cluster to have high centrality.
        centrality_type: Type of centrality to boost.
        boost_factor: Factor by which to increase the centrality of selected nodes.

    Returns:
        G: Generated graph.
    """
    # Step 1: Generate the initial SBM
    G = nx.Graph()
    current_node = 0
    clusters = []

    for size in cluster_sizes:
        cluster = list(range(current_node, current_node + size))
        clusters.append(cluster)
        current_node += size


    # Устанавливаем связи внутри каждого кластера
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if np.random.rand() < p_intra:
                    G.add_edge(cluster[i], cluster[j])

    # Устанавливаем связи между кластерами
    for idx1 in range(len(clusters)):
        for idx2 in range(idx1 + 1, len(clusters)):
            for u in clusters[idx1]:
                for v in clusters[idx2]:
                    if np.random.rand() < p_inter:
                        G.add_edge(u, v)

    # for i in range(len(clusters)):
    #     for j in range(len(clusters)):
    #         for u in clusters[i]:
    #             for v in clusters[j]:
    #                 if u != v:
    #                     prob = p_intra if i == j else p_inter
    #                     if np.random.rand() < prob:
    #                         G.add_edge(u, v)

    # Step 2: Identify the target cluster
    target_cluster = clusters[target_cluster_index]
    num_high_centrality_nodes = int(proportion_high_centrality * len(target_cluster))
    high_centrality_nodes = np.random.choice(target_cluster, size=num_high_centrality_nodes, replace=False)

    # Precompute centralities if needed
    if centrality_type in ["eigenvector", "pagerank", "betweenness", "closeness", "harmonic"]:
        if centrality_type == "eigenvector":
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        elif centrality_type == "pagerank":
            centrality = nx.pagerank(G)
        elif centrality_type == "betweenness":
            centrality = nx.betweenness_centrality(G)
        elif centrality_type == "closeness":
            centrality = nx.closeness_centrality(G)
        elif centrality_type == "harmonic":
            centrality = nx.harmonic_centrality(G)
        
        # Select top nodes from other clusters
        central_nodes = sorted(centrality, key=centrality.get, reverse=True)
        other_cluster_central_nodes = [node for node in central_nodes if node not in target_cluster][:int(0.1 * n)]

    # Step 3: Boost centrality of selected nodes
    if centrality_type == "degree":
        for node in high_centrality_nodes:
            current_degree = G.degree[node]
            desired_degree = int(current_degree * boost_factor)
            potential_neighbors = [v for v in G.nodes if v != node and not G.has_edge(node, v)]
            num_new_edges = min(desired_degree - current_degree, len(potential_neighbors))
            if num_new_edges > 0:
                new_neighbors = np.random.choice(potential_neighbors, size=num_new_edges, replace=False)
                G.add_edges_from([(node, nn) for nn in new_neighbors])

    elif centrality_type in ["eigenvector", "pagerank"]:
        for node in high_centrality_nodes:
            for neighbor in other_cluster_central_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < p_inter * boost_factor:
                    G.add_edge(node, neighbor)

    elif centrality_type in ["betweenness", "closeness", "harmonic"]:
        for node in high_centrality_nodes:
            for neighbor in other_cluster_central_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < p_inter * boost_factor:
                    G.add_edge(node, neighbor)

    elif centrality_type == "local_clustering":
        for node in high_centrality_nodes:
            neighbors = list(G.neighbors(node))
            np.random.shuffle(neighbors)
            current_cc = nx.clustering(G, node)
            max_edges = len(neighbors) * (len(neighbors) - 1) // 2
            current_edges = int(current_cc * max_edges)
            desired_edges = min(int(current_edges * boost_factor), max_edges)
            edges_needed = desired_edges - current_edges
            if edges_needed <= 0:
                continue
            # Generate all possible non-edges among neighbors
            non_edges = [(u, v) for i, u in enumerate(neighbors) for v in neighbors[i+1:] if not G.has_edge(u, v)]
            # Sample edges to add
            add_edges = min(edges_needed, len(non_edges))
            if add_edges > 0:
                selected_edges = np.random.choice(len(non_edges), size=add_edges, replace=False)
                G.add_edges_from([non_edges[idx] for idx in selected_edges])

    elif centrality_type == "percolation":
        # Connect to high-degree nodes across clusters
        all_high_degree_nodes = sorted(G.nodes, key=lambda x: G.degree[x], reverse=True)[:int(0.1 * n)]
        for node in high_centrality_nodes:
            for neighbor in all_high_degree_nodes:
                if not G.has_edge(node, neighbor) and np.random.rand() < p_inter * boost_factor:
                    G.add_edge(node, neighbor)

    elif centrality_type == "cross_clique":
        cliques = list(nx.find_cliques(G))
        for node in high_centrality_nodes:
            # Find cliques not containing the node
            other_cliques = [c for c in cliques if node not in c]
            for clique in other_cliques:
                candidates = [v for v in clique if not G.has_edge(node, v)]
                if candidates:
                    neighbor = np.random.choice(candidates)
                    if np.random.rand() < p_inter * boost_factor:
                        G.add_edge(node, neighbor)
    elif centrality_type == "random":
        pass
    else:
        raise ValueError("Unsupported centrality type.")

    C_total_matrix = nx.to_numpy_array(G)
    p_in_measured, p_out_measured = measure_connectivity(C_total_matrix, cluster_sizes)
    print(f"Фактическая p_in (доля связей внутри кластера): {p_in_measured:.3f}")
    print(f"Фактическая p_out (доля связей между кластерами): {p_out_measured:.3f}")

    return G, p_in_measured, p_out_measured


class GraphCentrality:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.graph = nx.from_numpy_array(adjacency_matrix)

    def calculate_betweenness_centrality(self):
        return nx.betweenness_centrality(self.graph)

    def calculate_eigenvector_centrality(self):
        return nx.eigenvector_centrality(self.graph)

    def calculate_pagerank_centrality(self, alpha=0.85):
        return nx.pagerank(self.graph, alpha=alpha)

    def calculate_flow_coefficient(self):
        flow_coefficient = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if len(neighbors) < 2:
                flow_coefficient[node] = 0.0
            else:
                edge_count = 0
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if self.graph.has_edge(neighbors[i], neighbors[j]):
                            edge_count += 1
                flow_coefficient[node] = (2 * edge_count) / (len(neighbors) * (len(neighbors) - 1))
        return flow_coefficient

    def calculate_degree_centrality(self):
        return nx.degree_centrality(self.graph)

    def calculate_closeness_centrality(self):
        return nx.closeness_centrality(self.graph)

    def calculate_harmonic_centrality(self):
        return nx.harmonic_centrality(self.graph)

    def calculate_percolation_centrality(self, attribute=None):
        if attribute is None:
            attribute = {node: 1 for node in self.graph.nodes()}
        return nx.percolation_centrality(self.graph, states=attribute)

    def calculate_cross_clique_centrality(self):
        cross_clique_centrality = {}
        cliques = list(nx.find_cliques(self.graph))
        for node in self.graph.nodes():
            cross_clique_centrality[node] = sum(node in clique for clique in cliques)
        return cross_clique_centrality



def plot_spikes(ax_spikes, spike_times, spike_indices, time_window_size, t_range, sim_time, oscillation_frequency, use_stdp, measure_name):
    
    ax_spikes.scatter(spike_times, spike_indices, marker='|')
    step_size = time_window_size
    total_time_ms = sim_time / ms
    time_steps = np.arange(0, total_time_ms + step_size, step_size)
    for t in time_steps:
        ax_spikes.axvline(x=t, color='grey', linestyle='--', linewidth=0.5)

    ax_spikes.set_xlim([0,5000])
    ax_spikes.set_xlabel("t [ms]")
    ax_spikes.set_ylabel("Neuron index")
    ax_spikes.set_title(f"Spike Raster Plot\nFrequency={oscillation_frequency} Hz, STDP={'On' if use_stdp else 'Off'}, Window={time_window_size} ms, {measure_name}", fontsize=16)
def plot_rates(ax_rates, N1, N2, rate_monitor, t_range):
    ax_rates.set_title(f'Num. of spikes\n neurons\n 0-{N1}', fontsize=16)
    ax_rates.plot(rate_monitor.t / ms, rate_monitor.rate / Hz, label=f'Group 1 (0-{n_neurons/2})')
    ax_rates.set_xlim(t_range)
    ax_rates.set_ylim([0,500])
    ax_rates.set_xlabel("t [ms]")


def plot_rates2(ax_rates2, N1, N2, rate_monitor2, t_range):
    ax_rates2.set_title(f'Num. of spikes\n neurons\n {N1}-{N2}', fontsize=16)
    ax_rates2.plot(rate_monitor2.t / ms, rate_monitor2.rate / Hz, label=f'Group 2 ({n_neurons/2}-{n_neurons})')
    ax_rates2.set_xlim(t_range)
    ax_rates2.set_ylim([0,500])
    ax_rates2.set_xlabel("t [ms]")    

def plot_psd(rate_monitor, N1, N2, ax_psd):
    rate = rate_monitor.rate / Hz - np.mean(rate_monitor.rate / Hz)
    from numpy.fft import rfft, rfftfreq
    N = len(rate_monitor.t)
    if N > 0:
        # Определение частоты дискретизации на основе временного шага
        dt = float((rate_monitor.t[1] - rate_monitor.t[0]) / ms)
        sampling_rate = 1000 / dt
        window = np.hanning(N)
        rate_windowed = rate * window
        # Ограничение до нужного диапазона частот
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate_windowed)) / N 
        yn = yn[:max_point]
    
        ax_psd.set_title(f"PSD neurons\n 0-{N1}", fontsize=16)
        ax_psd.plot(x, yn, c='k', label='Function')
        ax_psd.set_xlim([0,50])
        ax_psd.set_xlabel('Hz')


def plot_psd2(rate_monitor2, N1, N2, ax_psd2):
    rate = rate_monitor2.rate / Hz - np.mean(rate_monitor2.rate / Hz)
    from numpy.fft import rfft, rfftfreq
    N = len(rate_monitor2.t)
    if N > 0:
        # Определение частоты дискретизации на основе временного шага
        dt = float((rate_monitor2.t[1] - rate_monitor2.t[0]) / ms) 
        sampling_rate = 1000 / dt 
        window = np.hanning(N)
        rate_windowed = rate * window
        # Ограничение до нужного диапазона частот
        max_point = int(N * 300 / sampling_rate)
        x = rfftfreq(N, d=1 / sampling_rate)
        x = x[:max_point]
        yn = 2 * np.abs(rfft(rate_windowed)) / N 
        yn = yn[:max_point]
    
        ax_psd2.set_title(f"PSD neurons\n {N1}-{N2}", fontsize=16)
        ax_psd2.plot(x, yn, c='k', label='Function')
        ax_psd2.set_xlim([0,50])
        ax_psd2.set_xlabel('Hz')

def plot_spectrogram(rate_monitor, rate_monitor2, N1, N2, ax_spectrogram):
    dt = float((rate_monitor.t[1] - rate_monitor.t[0]) / second)  # dt в секундах
    N_freq = len(rate_monitor.t)
    xf = rfftfreq(len(rate_monitor.t), d=dt)[:N_freq]
    ax_spectrogram.plot(xf, np.abs(rfft(rate_monitor.rate / Hz)), label=f'{0}-{N1} neurons')
    
    dt2 = float((rate_monitor2.t[1] - rate_monitor2.t[0]) / second)
    N_freq2 = len(rate_monitor2.t)
    xf2 = rfftfreq(len(rate_monitor2.t), d=dt2)[:N_freq2]
    ax_spectrogram.plot(xf2, np.abs(rfft(rate_monitor2.rate / Hz)), label=f'{N1}-{N2} neurons')
    
    ax_spectrogram.set_xlim(0, 50)
    ax_spectrogram.legend()
    ax_spectrogram.set_title(f'Global\n Frequencies', fontsize=16)


# def plot_connectivity(n_neurons, S_intra1, S_intra2, S_12, S_21, connectivity2, centrality, p_in_measured, p_out_measured, percent_central):




def print_centrality(C_total, N1, cluster_nodes, p_input, boost_factor, test_num, num_tests, measure_name, direct_1_2=True):
    """
    Вычисляет указанную меру центральности (measure_name), выводит список топ-узлов 
    по этой метрике (но только среди узлов одного кластера), а также возвращает этот список.

    Параметры:
    ----------
    C_total : numpy.ndarray
        Квадратная матрица смежности, описывающая граф.
    cluster_nodes : int или iterable
        Число узлов или список узлов (индексов), составляющих кластер, 
        по которым выбираем топ-узлы.
    p_input : float
        Доля от числа узлов кластера, которая берётся для формирования топ-листа.
        Если cluster_nodes - целое число, то используется именно это число для определения размера кластера.
    measure_name : str
        Название метрики центральности, которую нужно вычислить.
    direct_1_2 : bool, optional
        Если True, то анализ проводится для первой половины узлов кластера, иначе – для узлов с индексами 250-499.

    Возвращает:
    ----------
    list
        Список индексов узлов, являющихся топ-узлами по заданной метрике (только внутри выбранного кластера).
    """
    if measure_name != 'random':
        graph_centrality = GraphCentrality(C_total)
        measure_func_map = {
            "betweenness": graph_centrality.calculate_betweenness_centrality,
            "eigenvector": graph_centrality.calculate_eigenvector_centrality,
            "pagerank": graph_centrality.calculate_pagerank_centrality,
            "flow": graph_centrality.calculate_flow_coefficient,
            "degree": graph_centrality.calculate_degree_centrality,
            "closeness": graph_centrality.calculate_closeness_centrality,
            "harmonic": graph_centrality.calculate_harmonic_centrality,
            "percolation": graph_centrality.calculate_percolation_centrality,
            "cross_clique": graph_centrality.calculate_cross_clique_centrality
        }
        if measure_name not in measure_func_map:
            raise ValueError(
                f"Метрика '{measure_name}' не поддерживается. "
                f"Доступные метрики: {list(measure_func_map.keys())}."
            )
        measure_values = measure_func_map[measure_name]()
        for node in measure_values:
            measure_values[node] = round(measure_values[node], 5)

        # Определяем список узлов для анализа в зависимости от значения direct_1_2
        if not direct_1_2:
            # Если direct_1_2 == False, выбираем нейроны с индексами 250-499
            cluster_list = list(range(N1, cluster_nodes))
        else:
            if isinstance(cluster_nodes, int):
                cluster_list = list(range(cluster_nodes))
            else:
                cluster_list = list(cluster_nodes)
            # Выбираем первую половину узлов кластера (как в исходной логике)
            cluster_list = cluster_list[:int(len(cluster_list)/2)]
        
        # Формируем словарь значений метрики для выбранных узлов
        measure_values_cluster = {
            node: measure_values[node] for node in cluster_list if node in measure_values
        }
        # Определяем количество топ-узлов для выборки
        top_k = int(p_input * len(cluster_list))
        sorted_neurons_cluster = sorted(
            measure_values_cluster,
            key=lambda n: measure_values_cluster[n],
            reverse=True
        )
        top_neurons = sorted_neurons_cluster[:top_k]
        # print(measure_values_cluster)
        # print(sorted_neurons_cluster)
        # print(top_neurons)
        if test_num == num_tests-1:
            plot_centrality_by_neuron_number(measure_values_cluster, top_neurons, measure_name, boost_factor, top_percent=p_input*100)
    else:
        # Обработка случайной выборки
        if not direct_1_2:
            cluster_indices = np.arange(N1, cluster_nodes)
        else:
            # Здесь предполагается, что общее число нейронов задано переменной n_neurons,
            # а для direct_1_2 == True выбираются нейроны первой половины
            cluster_indices = np.arange(0, int(n_neurons/2))
        num_chosen = int(p_input * len(cluster_indices))
        top_neurons = np.random.choice(cluster_indices, size=num_chosen, replace=False)

    return top_neurons


def plot_centrality_by_neuron_number(measure_values_cluster, top_neurons, measure_name, boost_factor, top_percent=10):
    """
    Строит график зависимости значения метрики центральности от номера нейрона.
    
    Параметры:
    -----------
    measure_values_cluster : dict
         Словарь, где ключ – номер нейрона, а значение – метрика центральности.
    top_neurons : list
         Список номеров нейронов, отобранных как топ по заданной метрике.
    top_percent : float, optional
         Процент для выделения порогового значения (по умолчанию 10%).
    """
    # Сортируем нейроны по их номерам для корректного отображения по оси x
    neurons = sorted(measure_values_cluster.keys())
    values = [measure_values_cluster[n] for n in neurons]
    
    plt.figure(figsize=(12, 6))
    # Линейный график, показывающий зависимость значения метрики от номера нейрона
    plt.plot(neurons, values, 'bo-', label='Значение метрики')
    plt.xlabel('Номер нейрона')
    plt.ylabel('Значение метрики центральности')
    plt.title(f'Зависимость {measure_name} от номера нейрона после увеличения boost_factor={boost_factor}')
    plt.grid(True)
    
    # Определим пороговое значение метрики для топ-нейронов.
    # Для этого сначала сортируем значения метрики по возрастанию.
    sorted_values = sorted(values)
    n = len(sorted_values)
    threshold_index = int(np.ceil((1 - top_percent/100) * n))
    threshold_value = sorted_values[threshold_index] if threshold_index < n else sorted_values[-1]
    
    # Отмечаем горизонтальной пунктирной линией пороговое значение метрики
    plt.axhline(threshold_value, color='green', linestyle='--', linewidth=2,
                label=f'Пороговая метрика (топ {top_percent}%) = {threshold_value:.3f}')
    
    # Выделяем на графике нейроны, входящие в топ (их номера берём из top_neurons)
    top_values = [measure_values_cluster[n] for n in top_neurons]
    plt.scatter(top_neurons, top_values, color='red', zorder=5, label='Топ-нейроны')
    
    plt.legend()
    plt.savefig(f"results_ext_test1/{measure_name}_{top_percent}centrality_plot_after_{boost_factor}.png")
    plt.close()

def save_csv_data(csv_filename, data_for_csv):
    """
    Сохранение данных в CSV
    """
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for row in data_for_csv:
            writer.writerow(row)

def save_gif(images, filename, duration=3000, loop=0):
    """
    Создание анимации из последовательности изображений
    """
    imageio.mimsave(filename, images, duration=duration, loop=loop)


def plot_3d_spike_data(
    detailed_spike_data_for_3d,
    measure_names,
    p_within_values,
    I0_value,
    oscillation_frequency,
    use_stdp,
    current_time,
    time_window_size,
    max_rate,
    directory_path='results_ext_test1',
):
    """
    Для каждого measure_name строится свой 3D-график (при фиксированном p_within).
    """
    os.makedirs(directory_path, exist_ok=True)
    
    if I0_value not in detailed_spike_data_for_3d:
        print(f"Нет данных для I0={I0_value}pA.")
        return
    
    # Перебираем, например, все метрики, включая "random"
    for measure_name in measure_names:
        if measure_name not in detailed_spike_data_for_3d[I0_value]:
            print(f"Нет данных для measure_name={measure_name} при I0={I0_value}. Пропуск.")
            continue
        
        for p_within in p_within_values:
            p_within_str = f"{p_within:.2f}"
            if p_within_str not in detailed_spike_data_for_3d[I0_value][measure_name]:
                print(f"Нет данных для p_within={p_within_str}. Пропуск.")
                continue

            p_between_list = sorted(detailed_spike_data_for_3d[I0_value][measure_name][p_within_str].keys())
            if len(p_between_list) == 0:
                print(f"Нет p_between для p_within={p_within_str}. Пропуск.")
                continue

            sample_p_between = p_between_list[0]
            time_array = detailed_spike_data_for_3d[I0_value][measure_name][p_within_str][sample_p_between].get("time", np.array([]))

            if time_array.size == 0:
                print(f"Нет временных данных для measure_name={measure_name}, p_within={p_within_str}, p_between={sample_p_between}.")
                continue

            # Проверяем согласованность временных окон
            consistent_time = True
            for p_btw in p_between_list:
                current_time_array = detailed_spike_data_for_3d[I0_value][measure_name][p_within_str][p_btw].get("time", np.array([]))
                if current_time_array.size == 0 or len(current_time_array) != len(time_array):
                    consistent_time = False
                    print(f"Несоответствие временных окон для measure_name={measure_name}, p_within={p_within_str}, p_between={p_btw}.")
                    break
            if not consistent_time:
                continue

            # Создаём сетки для поверхности
            Time, P_between = np.meshgrid(time_array, p_between_list)
            Z = np.zeros(Time.shape)

            # Заполняем Z
            for i, p_btw in enumerate(p_between_list):
                spikes_arr = detailed_spike_data_for_3d[I0_value][measure_name][p_within_str][p_btw].get("spikes_list", [])
                if not spikes_arr:
                    Z[i, :] = 0
                else:
                    spikes_arr = np.array(spikes_arr)
                    if len(spikes_arr) < Z.shape[1]:
                        spikes_arr = np.pad(spikes_arr, (0, Z.shape[1] - len(spikes_arr)), 'constant')
                    elif len(spikes_arr) > Z.shape[1]:
                        spikes_arr = spikes_arr[:Z.shape[1]]
                    Z[i, :] = spikes_arr
            
            if not np.isfinite(Z).all():
                print(f"Некоторые значения в Z не являются конечными для measure_name={measure_name}, p_within={p_within_str}.")
                continue

            # Построение 3D
            fig_3d = plt.figure(figsize=(10, 8))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            surf = ax_3d.plot_surface(
                Time,        # X
                P_between,   # Y
                Z,           # Z
                cmap='viridis',
                edgecolor='none'
            )
            ax_3d.set_xlabel('Time [ms]')
            ax_3d.set_ylabel('p_between')
            ax_3d.set_zlabel('Avg Spikes (Hz)')
            ax_3d.set_zlim(0,max_rate)
            ax_3d.set_title(
                f'3D: Time vs p_between vs Avg Spikes\n'
                f'I0={I0_value}pA, freq={oscillation_frequency}Hz, STDP={"On" if use_stdp else "Off"}, '
                f'measure={measure_name}, p_within={p_within_str}, Time={current_time}ms',
                fontsize=14
            )
            fig_3d.colorbar(surf, shrink=0.5, aspect=5)

            fig_filename_3d = os.path.join(
                directory_path,
                f'3D_plot_I0_{I0_value}pA_freq_{oscillation_frequency}Hz_STDP_{"On" if use_stdp else "Off"}_'
                f'measure_{measure_name}_p_within_{p_within_str}_Time_{current_time}ms.png'
            )
            plt.savefig(fig_filename_3d)
            plt.close(fig_3d)


def plot_pinput_between_avg_spikes_with_std(
    spike_counts_second_cluster_for_input,
    spike_counts_second_cluster,
    I0_value,
    oscillation_frequency,
    use_stdp,
    current_time,
    time_window_size,
    max_rate,
    p_input_values,
    p_between_values,
    directory_path='results_ext_test1',
    measure_names=None,
):
    """
    Для каждой метрики (кроме 'random') строит 3D-график с наложением поверхности для 'random'.
    
    Структура хранения данных:
      spike_counts_second_cluster_for_input[I0_value][measure_name][p_within][p_input][p_between]
          = { 'mean_spikes': <среднее число спайков>,
              'avg_exc_synapses': <среднее число возбуждающих синапсов>,
              'avg_inh_synapses': <среднее число тормозных синапсов> }
      spike_counts_second_cluster[I0_value][measure_name][p_within][p_between] = список,
          где каждая запись — массив средних значений, полученных в различных тестах.
    
    По итогам работы функция сохраняет результаты в CSV‑файл с дополнительными столбцами,
    а также формирует 3D-графики для каждой пары (measure_name, p_within).
    """
    import csv
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Создание директории для сохранения результатов
    os.makedirs(directory_path, exist_ok=True)
    
    # Вывод отладочной информации о структурах данных
    print("spike_counts_second_cluster_for_input:", spike_counts_second_cluster_for_input)
    print("spike_counts_second_cluster:", spike_counts_second_cluster)
    
    # Формирование имени CSV-файла с учётом времени симуляции
    avg_csv_filename = os.path.join(directory_path, f"avg_tests_avg_spikes_{current_time}ms.csv")
    
    # Открытие CSV-файла для записи результатов
    with open(avg_csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Заголовок CSV-файла включает дополнительные столбцы для синаптических параметров
        writer.writerow([
            'I0_value', 'measure_name', 'p_within', 'p_input', 'p_between',
            'mean_spikes', 'std_spikes', 'avg_exc_synapses', 'avg_inh_synapses'
        ])
        
        # Проверка наличия данных для данного I0_value
        if I0_value not in spike_counts_second_cluster_for_input:
            print(f"Нет данных spike_counts_second_cluster_for_input для I0={I0_value}. Пропуск.")
            return
        
        # Определение списка метрик для анализа
        measure_names_in_data = list(spike_counts_second_cluster_for_input[I0_value].keys())
        if measure_names is None:
            measure_names = measure_names_in_data
        
        # Перебор метрик, за исключением 'random' (которая используется для контрольного сравнения)
        for measure_name in measure_names:
            if measure_name == 'random':
                continue
            if measure_name not in measure_names_in_data:
                print(f"measure_name={measure_name} не найден в данных. Пропуск.")
                continue
            if 'random' not in measure_names_in_data:
                print(f"Данные для 'random' не найдены. Пропуск объединённого графика для measure={measure_name}.")
                continue

            # Извлечение данных для текущей метрики и контрольной (random)
            data_measure = spike_counts_second_cluster_for_input[I0_value][measure_name]
            data_random = spike_counts_second_cluster_for_input[I0_value]['random']

            # Перебор всех значений p_within (строковая запись, например "0.15")
            for p_within_str in data_measure.keys():
                if p_within_str not in data_random:
                    print(f"Для p_within={p_within_str} нет данных в 'random'. Пропуск.")
                    continue

                dict_pinput_measure = data_measure[p_within_str]
                dict_pinput_random = data_random[p_within_str]

                # Отсортированные списки значений p_input (ключей словарей)
                sorted_p_inputs_measure = sorted(dict_pinput_measure.keys(), key=float)
                sorted_p_inputs_random = sorted(dict_pinput_random.keys(), key=float)
                common_p_inputs = sorted(set(sorted_p_inputs_measure).intersection(sorted_p_inputs_random), key=float)
                if not common_p_inputs:
                    print(f"Нет пересечения p_input для measure_name={measure_name}, p_within={p_within_str}. Пропуск.")
                    continue

                # Определение пересечения значений p_between
                first_p_input = common_p_inputs[0]
                p_between_list_measure = dict_pinput_measure[first_p_input].keys()
                p_between_list_random = dict_pinput_random[first_p_input].keys()
                common_p_between = sorted(set(p_between_list_measure).intersection(p_between_list_random), key=float)
                if not common_p_between:
                    print(f"Нет пересечения p_between для measure_name={measure_name}, p_within={p_within_str}. Пропуск.")
                    continue

                # Формирование осей для построения матриц
                p_input_list_float = [float(p) for p in common_p_inputs]
                p_between_list_float = [float(pb) for pb in common_p_between]
                Z_measure = np.zeros((len(p_input_list_float), len(p_between_list_float)))
                Z_random = np.zeros_like(Z_measure)
                Z_measure_std = np.zeros_like(Z_measure)
                Z_random_std = np.zeros_like(Z_measure)
                
                # Извлечение списка тестовых данных для вычисления стандартного отклонения
                cluster_data_measure = spike_counts_second_cluster.get(I0_value, {}).get(measure_name, {}).get(p_within_str, {})
                cluster_data_random = spike_counts_second_cluster.get(I0_value, {}).get('random', {}).get(p_within_str, {})

                # Перебор по значениям p_input и p_between для заполнения матриц и записи CSV
                for i, p_inp_str in enumerate(common_p_inputs):
                    p_btw_dict_measure = dict_pinput_measure[p_inp_str]
                    p_btw_dict_random  = dict_pinput_random[p_inp_str]
                    for j, p_btw in enumerate(p_between_list_float):
                        # Формирование строкового представления p_between
                        p_btw_str = f"{p_btw:.2f}"
                        
                        # Извлечение числовых значений среднего числа спайков
                        # Ожидается, что p_btw_dict_* хранит словарь с ключами 'mean_spikes', 'avg_exc_synapses', 'avg_inh_synapses'
                        data_entry_measure = p_btw_dict_measure.get(p_btw, {
                            'mean_spikes': 0.0, 'avg_exc_synapses': 0.0, 'avg_inh_synapses': 0.0
                        })
                        data_entry_random = p_btw_dict_random.get(p_btw, {
                            'mean_spikes': 0.0, 'avg_exc_synapses': 0.0, 'avg_inh_synapses': 0.0
                        })
                        mean_spikes_measure_val = float(data_entry_measure.get('mean_spikes', 0.0))
                        mean_spikes_random_val = float(data_entry_random.get('mean_spikes', 0.0))
                        
                        # Заполнение матриц для построения поверхностей
                        Z_measure[i, j] = mean_spikes_measure_val
                        Z_random[i, j] = mean_spikes_random_val

                        # Вычисление стандартного отклонения для measure (на основе данных тестовых повторов)
                        list_of_arrays_measure = cluster_data_measure.get(p_btw, [])
                        stdev_value_measure = 0.0
                        if list_of_arrays_measure:
                            num_pinputs = len(dict_pinput_measure.keys())
                            n_rep = len(list_of_arrays_measure) // num_pinputs if num_pinputs > 0 else 1
                            index_to_use = (i + 1) * n_rep - 1 if n_rep > 0 else 0
                            try:
                                trial_array = np.array(list_of_arrays_measure[index_to_use], dtype=float)
                                stdev_value_measure = np.std(trial_array)
                            except Exception as e:
                                print(e)
                        Z_measure_std[i, j] = stdev_value_measure

                        # Вычисление стандартного отклонения для контрольной выборки (random)
                        list_of_arrays_random = cluster_data_random.get(p_btw, [])
                        stdev_value_random = 0.0
                        if list_of_arrays_random:
                            num_pinputs = len(dict_pinput_random.keys())
                            n_rep = len(list_of_arrays_random) // num_pinputs if num_pinputs > 0 else 1
                            index_to_use = (i + 1) * n_rep - 1 if n_rep > 0 else 0
                            try:
                                trial_array = np.array(list_of_arrays_random[index_to_use], dtype=float)
                                stdev_value_random = np.std(trial_array)
                            except Exception as e:
                                print(e)
                        Z_random_std[i, j] = stdev_value_random

                        # Извлечение дополнительных синаптических параметров для measure
                        avg_exc_synapses_measure = float(data_entry_measure.get('avg_exc_synapses', 0.0))
                        avg_inh_synapses_measure = float(data_entry_measure.get('avg_inh_synapses', 0.0))
                        avg_exc_synapses_random = float(data_entry_random.get('avg_exc_synapses', 0.0))
                        avg_inh_synapses_random = float(data_entry_random.get('avg_inh_synapses', 0.0))
                        
                        # Запись строк в CSV для measure_name и для random
                        writer.writerow([
                            I0_value, measure_name, p_within_str, p_inp_str, p_btw_str,
                            mean_spikes_measure_val, stdev_value_measure,
                            avg_exc_synapses_measure, avg_inh_synapses_measure
                        ])
                        writer.writerow([
                            I0_value, 'random', p_within_str, p_inp_str, p_btw_str,
                            mean_spikes_random_val, stdev_value_random,
                            avg_exc_synapses_random, avg_inh_synapses_random
                        ])
                
                # Построение 3D-графика для текущей пары (measure_name, p_within)
                p_input_mesh, p_between_mesh = np.meshgrid(p_input_list_float, p_between_list_float, indexing='ij')
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title(
                    f'3D: p_input vs p_between vs avg_spikes\n'
                    f'I0={I0_value} pA, freq={oscillation_frequency} Hz, STDP={"On" if use_stdp else "Off"}\n'
                    f'p_within={p_within_str}, Time={current_time} ms, Bin={time_window_size} ms\n'
                    f'Metrics: {measure_name} and random',
                    fontsize=13
                )
                ax.set_zlim(0, max_rate)
                ax.set_xlabel('p_input')
                ax.set_ylabel('p_between')
                ax.set_zlabel('avg_spikes (Hz)')
                ax.set_xticks(p_input_values)
                ax.set_yticks(p_between_values)
                # Поверхность для основной метрики
                surf_measure = ax.plot_surface(
                    p_input_mesh, p_between_mesh, Z_measure,
                    cmap='magma', edgecolor='none', zorder=1, vmin=0, vmax=max_rate, alpha=0.7
                )
                # Поверхность для контрольной выборки ("random")
                surf_random = ax.plot_surface(
                    p_input_mesh, p_between_mesh, Z_random,
                    cmap='magma', edgecolor='none', zorder=1, vmin=0, vmax=max_rate, alpha=0.5
                )
                # Добавление вертикальных линий для error bars по основной метрике
                for irow in range(Z_measure.shape[0]):
                    for jcol in range(Z_measure.shape[1]):
                        x_val = p_input_mesh[irow, jcol]
                        y_val = p_between_mesh[irow, jcol]
                        z_val = Z_measure[irow, jcol]
                        err = Z_measure_std[irow, jcol]
                        ax.plot([x_val, x_val], [y_val, y_val],
                                [z_val - err, z_val + err], c='k', zorder=2, linewidth=1.5)
                # Error bars для контрольной выборки ("random")
                for irow in range(Z_random.shape[0]):
                    for jcol in range(Z_random.shape[1]):
                        x_val = p_input_mesh[irow, jcol]
                        y_val = p_between_mesh[irow, jcol]
                        z_val = Z_random[irow, jcol]
                        err = Z_random_std[irow, jcol]
                        ax.plot([x_val, x_val], [y_val, y_val],
                                [z_val - err, z_val + err], c='k', zorder=2, linewidth=1.5)
                cb1 = fig.colorbar(surf_measure, ax=ax, shrink=0.5, aspect=10)
                
                # Сохранение графика в указанный каталог
                filename = os.path.join(
                    directory_path,
                    f'3D_two_surfaces_{measure_name}_vs_random_I0_{I0_value}_p_within_{p_within_str}_Time_{current_time}ms.png'
                )
                plt.savefig(filename, dpi=150)
                plt.close(fig)
    
    print("Формирование объединённых графиков (основная метрика vs random) завершено.")



def plot_all_measures_vs_random_from_csv(
    csv_filename,
    p_input_values,
    p_between_values,
    max_rate,
    I0_value_filter=1000,
    output_pdf="results_ext_test1/3D_all_measures_vs_random.pdf",
    measure_names=None
):
    """
    Функция читает данные из CSV-файла, содержащего колонки:
      I0_value, measure_name, p_within, p_input, p_between, mean_spikes, std_spikes.
      
    Для 6 метрик (measure_name, кроме 'random') строится объединённый 3D-рисунок. Для каждой метрики 
    выбирается одно значение p_within (первое по сортировке), после чего на соответствующем сабплоте 
    отображаются две поверхности – одна для основной метрики, другая для контрольной (measure_name 'random').
      
    В каждой точке также отрисовываются вертикальные error bar, соответствующие значению std_spikes.
    Для всех сабграфиков используется единая цветовая шкала, а общий colormap отображается через колорбар справа.
    
    Параметры:
      csv_filename : str
          Путь к CSV-файлу с данными.
      max_rate : float
          Максимальное значение для оси Z и нормализации colormap.
      I0_value_filter : int или float, по умолчанию 1000
          Значение I0_value для фильтрации данных.
      output_pdf : str, по умолчанию "3D_all_measures_vs_random.pdf"
          Имя выходного PDF-файла.
      measure_names : list или None
          Список метрик для отображения. Если None, выбираются все метрики, кроме 'random'.
    """
    # Чтение CSV-файла и фильтрация по I0_value
    df = pd.read_csv(csv_filename)
    df = df[df["I0_value"] == I0_value_filter]
    
    # Получаем список всех уникальных метрик и исключаем 'random'
    all_measures = sorted(df["measure_name"].unique())
    available_measures = [m for m in all_measures if m != 'random']
    if measure_names is not None:
        available_measures = [m for m in measure_names if m in available_measures]
    
    # Если найдено не 6 метрик, берем первые 6
    if len(available_measures) != 6:
        print(f"Ожидается 6 метрик, а найдено {len(available_measures)}. Будут использованы первые 6.")
        available_measures = available_measures[:6]
    
    # Настройка фигуры с 6 сабплотами (2 строки x 3 столбца)
    fig = plt.figure(figsize=(18, 12))
    marker_size = 50  # размер маркеров для scatter
    cmap = plt.get_cmap('magma')
    norm = mpl.colors.Normalize(vmin=0, vmax=max_rate)
    
    # Перебор метрик для построения отдельных 3D-графиков
    for idx, measure in enumerate(available_measures):
        # Отбор данных для выбранной метрики и для 'random'
        df_measure = df[df["measure_name"] == measure]
        df_random = df[df["measure_name"] == "random"]
        
        # Выбор одного значения p_within (первое по сортировке)
        p_within_vals_measure = sorted(df_measure["p_within"].unique(), key=float)
        if not p_within_vals_measure:
            print(f"Нет значений p_within для {measure}. Пропуск.")
            continue
        p_within_val = p_within_vals_measure[0]
        
        # Фильтрация по выбранному p_within
        df_measure_pw = df_measure[df_measure["p_within"] == p_within_val]
        df_random_pw = df_random[df_random["p_within"] == p_within_val]
        
        # Определение общего множества значений p_input и p_between
        common_p_input = sorted(list(set(df_measure_pw["p_input"].unique()).intersection(
            set(df_random_pw["p_input"].unique()))), key=float)
        common_p_between = sorted(list(set(df_measure_pw["p_between"].unique()).intersection(
            set(df_random_pw["p_between"].unique()))), key=float)
        
        if not common_p_input or not common_p_between:
            print(f"Нет пересечения значений p_input или p_between для {measure} при p_within={p_within_val}. Пропуск.")
            continue
        
        # Используем pivot_table с агрегирующей функцией для устранения дублирования
        pivot_measure_mean = df_measure_pw.pivot_table(index="p_input", columns="p_between", values="mean_spikes", aggfunc=np.mean)
        pivot_random_mean = df_random_pw.pivot_table(index="p_input", columns="p_between", values="mean_spikes", aggfunc=np.mean)
        pivot_measure_std = df_measure_pw.pivot_table(index="p_input", columns="p_between", values="std_spikes", aggfunc=np.mean)
        pivot_random_std = df_random_pw.pivot_table(index="p_input", columns="p_between", values="std_spikes", aggfunc=np.mean)
        
        # Ограничиваем данные пересечением общих значений
        pivot_measure_mean = pivot_measure_mean.loc[common_p_input, common_p_between]
        pivot_random_mean = pivot_random_mean.loc[common_p_input, common_p_between]
        pivot_measure_std = pivot_measure_std.loc[common_p_input, common_p_between]
        pivot_random_std = pivot_random_std.loc[common_p_input, common_p_between]
        
        # Преобразуем списки ключей в массивы и создаем сетку для построения поверхности
        p_input_arr = np.array(common_p_input, dtype=float)
        p_between_arr = np.array(common_p_between, dtype=float)
        p_input_mesh, p_between_mesh = np.meshgrid(p_input_arr, p_between_arr, indexing='ij')
        
        # Извлекаем матрицы значений
        Z_measure = pivot_measure_mean.values.astype(float)
        Z_random = pivot_random_mean.values.astype(float)
        Z_measure_std = pivot_measure_std.values.astype(float)
        Z_random_std = pivot_random_std.values.astype(float)
        
        # Создание саблота для текущей метрики
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        ax.set_title(f"{measure} vs random, $p_{{within}}={p_within_val}$", fontsize=13)
        ax.set_zlim(0, max_rate)
        ax.set_xlabel(r'$p_{input}$', fontsize=14)
        ax.set_ylabel(r'$p_{between}$', fontsize=14)
        ax.set_zlabel(r'$\overline{spikes}$', fontsize=14)
        ax.set_xticks(p_input_values)   # или какие именно точки хотите
        ax.set_yticks(p_between_values)
        # Построение поверхности для measure
        surf_measure = ax.plot_surface(
            p_input_mesh, p_between_mesh, Z_measure,
            cmap=cmap, edgecolor='none', alpha=0.8, vmin=0, vmax=max_rate
        )
        # Построение поверхности для random
        surf_random = ax.plot_surface(
            p_input_mesh, p_between_mesh, Z_random,
            cmap=cmap, edgecolor='none', alpha=0.6, vmin=0, vmax=max_rate
        )
        
        # Отрисовка error bar для measure
        for i in range(Z_measure.shape[0]):
            for j in range(Z_measure.shape[1]):
                x_val = p_input_mesh[i, j]
                y_val = p_between_mesh[i, j]
                z_val = Z_measure[i, j]
                err = Z_measure_std[i, j]
                ax.plot([x_val, x_val], [y_val, y_val], [z_val - err, z_val + err],
                        color='k', linewidth=1.5)
        
        # Отрисовка error bar для random
        for i in range(Z_random.shape[0]):
            for j in range(Z_random.shape[1]):
                x_val = p_input_mesh[i, j]
                y_val = p_between_mesh[i, j]
                z_val = Z_random[i, j]
                err = Z_random_std[i, j]
                ax.plot([x_val, x_val], [y_val, y_val], [z_val - err, z_val + err],
                        color='k', linewidth=1.5)
    
    # Добавление общего колорбара справа от всех сабграфиков
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(r'$\overline{spikes}$', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_pdf)
    plt.show()
    
    print("Построение объединённого графика для 6 метрик vs random завершено.")


import random




def plot_granger(time_series_v, ax_granger, ax_dtf, ax_pdc):
    # Параметры мультитаперного спектрального анализа.
    # Увеличиваем длительность временного окна для более чёткого частотного разрешения.

    # n_time_samples = time_series_s.shape[1]
    # print('n_tine', n_time_samples)
    # t_full = np.arange(n_time_samples) / 10

    # fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    # # ax[0].plot(t_full, time_series[trial_idx, :, 0], 'o', label="x1")
    # ax[0].plot(t_full, time_series_s[trial_idx, :, 0], label="0-249")
    # ax[0].set_ylabel("Амплитуда")
    # ax[0].legend()

    # # ax[1].plot(t_full, time_series[trial_idx, :, 1], 'o', label="x2", color="orange")
    # ax[1].plot(t_full, time_series_s[trial_idx, :, 1], label="250-499", color="orange")
    # ax[1].set_xlabel("Время (сек)")
    # ax[1].set_ylabel("Амплитуда")
    # ax[1].legend()

    time_halfbandwidth_product = 5
    time_window_duration = 3
    time_window_step = 0.1

    print("Начало счета Multitaper")
    from spectral_connectivity import Multitaper, Connectivity
    print(time_series_v.shape)
    m = Multitaper(
        time_series_v,
        sampling_frequency=100,
        time_halfbandwidth_product=time_halfbandwidth_product,
        start_time=0,
        time_window_duration=time_window_duration,
        time_window_step=time_window_step,
    )
    # Рассчитываем объект Connectivity
    c = Connectivity.from_multitaper(m)

    # =============================================================================
    # 3. Расчет мер направленной связи
    # =============================================================================
    granger = c.pairwise_spectral_granger_prediction()
    dtf = c.directed_transfer_function()
    pdc = c.partial_directed_coherence()

    # 4.1. Спектральная Грейнджерова причинность
    ax_granger[0].pcolormesh(c.time, c.frequencies, granger[..., :, 0, 1].T, cmap="viridis", shading="auto")
    ax_granger[0].set_title("GC: x1 → x2")
    ax_granger[0].set_ylabel("Frequency (Hz)")
    ax_granger[1].pcolormesh(c.time, c.frequencies, granger[..., :, 1, 0].T, cmap="viridis", shading="auto")
    ax_granger[1].set_title("GC: x2 → x1")
    ax_granger[1].set_xlabel("Time (s)")
    ax_granger[1].set_ylabel("Frequency (Hz)")

    # 4.2. directed transfer function
    ax_dtf[0].pcolormesh(c.time, c.frequencies, dtf[..., :, 0, 1].T, cmap="viridis", shading="auto")
    ax_dtf[0].set_title("DTF: x1 → x2")
    ax_dtf[0].set_ylabel("Frequency (Hz)")
    ax_dtf[1].pcolormesh(c.time, c.frequencies, dtf[..., :, 1, 0].T, cmap="viridis", shading="auto")
    ax_dtf[1].set_title("DTF: x2 → x1")
    ax_dtf[1].set_xlabel("Time (s)")
    ax_dtf[1].set_ylabel("Frequency (Hz)")

    ax_pdc[0].pcolormesh(c.time, c.frequencies, pdc[..., :, 0, 1].T, cmap="viridis", shading="auto")
    ax_pdc[0].set_title("PDC: x1 → x2")
    ax_pdc[0].set_ylabel("Frequency (Hz)")
    ax_pdc[1].pcolormesh(c.time, c.frequencies, pdc[..., :, 1, 0].T, cmap="viridis", shading="auto")
    ax_pdc[1].set_title("PDC: x2 → x1")
    ax_pdc[1].set_xlabel("Time (s)")
    ax_pdc[1].set_ylabel("Frequency (Hz)")


def plot_connectivity(subplot_results):
    print(subplot_results)
    if subplot_results:
        import matplotlib.colors as mcolors
        n_measures = len(subplot_results)
        
        # Настройка структуры субплотов
        if n_measures > 4:
            nrows = 2
            ncols = math.ceil(n_measures / 2)
        else:
            nrows = 1
            ncols = n_measures
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 8 * nrows))
        
        # Приведение осей к плоскому массиву
        axes = np.array(axes).flatten() if n_measures > 1 else [axes]
        
        # Кастомная цветовая схема: 0-чёрный, 0.5-белый, 1-красный
        colors = ['black', 'white', 'red']
        cmap = mcolors.ListedColormap(colors)
        bounds = [0, 0.5, 1, 1.1]  # Границы для дискретных цветов
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        for ax, (measure, params) in zip(axes, subplot_results.items()):
            W, M, p_in_measured, p_out_measured, percent_central = params
            highlighted_neurons = np.unique(centrality)
            
            # Инициализация матрицы отображения
            n_neurons = W.shape[0]
            M_vis = np.zeros_like(W)
            
            # Все существующие связи (белый)
            M_vis[W != 0] = 0.5
            
            # Помечаем связи центральных узлов (красный)
            for neuron in highlighted_neurons:
                M_vis[neuron, W[neuron, :] != 0] = 1.0  # Исходящие
                M_vis[W[:, neuron] != 0, neuron] = 1.0  # Входящие

            # Визуализация матрицы
            ax.matshow(M_vis, cmap=cmap, norm=norm)
            
            # Настройка заголовка и осей
            ax.set_title(
                f'p_in: {p_in_measured:.3f}, p_out: {p_out_measured:.3f}\n'
                f'Central: {percent_central}%, Metric: {measure}',
                fontsize=10
            )
            ax.set_xticks([0, n_neurons//2, n_neurons-1])
            ax.set_yticks([0, n_neurons//2, n_neurons-1])
            ax.tick_params(axis='both', labelsize=8)

        # Сокрытие неиспользованных осей
        for ax in axes[len(subplot_results):]:
            ax.axis('off')
            
        # Сохранение рисунка
        plt.tight_layout()
        plt.savefig(os.path.join(directory_path, 
                    f"connectivity_pinput_{p_input:.2f}_pbetween_{p_between:.3f}.pdf"))
        plt.close()

# def plot_connectivity(subplot_results):
#     print(subplot_results)
#     # Если имеются результаты для заданных условий, строим общий график
#     if subplot_results:
#         n_measures = len(subplot_results)
        
#         # Определение количества строк и столбцов для субплотов
#         if n_measures > 4:
#             nrows = 2
#             ncols = math.ceil(n_measures / 2)
#         else:
#             nrows = 1
#             ncols = n_measures
        
#         fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 8 * nrows))
        
#         # Приведение axes к одномерному массиву для удобства перебора
#         if n_measures == 1:
#             axes = [axes]
#         else:
#             axes = np.array(axes).flatten()
        
#         # Проход по всем заданным мерам и построение соответствующих графиков
#         for ax, (measure, params) in zip(axes, subplot_results.items()):
#             W, M, p_in_measured, p_out_measured, percent_central = params
#             # Определение центральных нейронов (индексация от 0 до n_neurons-1)
#             highlighted_neurons = np.unique(centrality)
            
#             # Обновление значений для связей, затрагивающих центральные нейроны:
#             # Если нейрон является источником или получателем связи, и связь присутствует, присваивается значение 1.
#             for neuron in highlighted_neurons:
#                 M[neuron, :] = np.where(W[neuron, :] != 0, 1.0, M[neuron, :])
#                 M[:, neuron] = np.where(W[:, neuron] != 0, 1.0, M[:, neuron])
            
#             # Отображение матрицы с использованием градаций серого
#             ax.matshow(M, cmap='gray', vmin=0, vmax=1)
#             ax.set_title(
#                 f'\np_in_measured={p_in_measured:.3f},\n'
#                 f'p_out_measured={p_out_measured:.3f},\n'
#                 f'p_central={percent_central},\n'
#                 f'Metric: {measure}',
#                 fontsize=14
#             )
            
#             # Выбор делений осей (позиции) и корректировка меток (индексация от 1 до n_neurons)
#             ticks = [0, n_neurons // 2, n_neurons - 1]
#             ax.set_xticks(ticks)
#             ax.set_yticks(ticks)
#             x_labels = [f'{i+1}' for i in ticks]
#             y_labels = [f'{i+1}' for i in ticks]
#             ax.set_xticklabels(x_labels)
#             ax.set_yticklabels(y_labels)
#             ax.tick_params(axis='both', which='major', labelsize=10)
        
#         # Если имеются лишние оси (например, при нечётном числе мер), скрываем их
#         for ax in axes[len(subplot_results):]:
#             ax.axis('off')
        
#         # Формирование имени файла с учётом значений p_input и p_between
#         subplot_filename = os.path.join(
#             directory_path,
#             f"subplot_pinput_{p_input:.2f}_pbetween_{p_between:.3f}_last_test.pdf"
#         )
#         plt.tight_layout()
#         plt.savefig(subplot_filename)
#         plt.close(fig)


def sim(p_within, p_between, J, J2, refractory_period, sim_time, plotting_flags,
        rate_tick_step, t_range, rate_range, cluster_labels, cluster_sizes,
        I0_value, oscillation_frequency, use_stdp, time_window_size, measure_names, boost_factor_list, test_num, num_tests,
        C_total_prev=None, p_within_prev=None, p_between_prev=None,
        p_input=None, measure_name=None, measure_name_prev=None, centrality=None):
    """
    Выполняет один прогон симуляции при заданных параметрах.
    Если C_total_prev=None, генерируем матрицу "с нуля",
    иначе используем прошлую (чтобы копить STDP).
    """

    start_scope()

    start_time = time.time()

    if plotting_flags is None:
        plotting_flags = {}

    n_neurons = len(cluster_labels)
    target_cluster_index = 0
    proportion_high_centrality = p_input
    centrality_type = measure_name

    if measure_name in measure_names:
        boost_factor = boost_factor_list[measure_names.index(measure_name)]
    else:
        boost_factor = 0  # значение по умолчанию, если measure_name не найден
    
    print("Используемый boost_factor:", boost_factor)

    C_total, p_in_measured, p_out_measured = generate_sbm_with_high_centrality(
        n_neurons, cluster_sizes, p_within, p_between,
        target_cluster_index, proportion_high_centrality,
        centrality_type, boost_factor, 
    )    

    # -- Параметры LIF --
    N = n_neurons
    N1 = int(n_neurons/2)
    N2 = n_neurons
    N_E = int(N * 80 / 100)
    N_I = N - N_E
    R = 80 * Mohm
    C = 0.25 * nfarad
    tau = R*C # 20 ms
    max_rate = 1/tau 
    v_threshold = -50 * mV
    v_reset = -70 * mV
    v_rest = -65 * mV
    J = J * mV
    J2 = J2 * mV
    defaultclock.dt = 0.01 * second
    phi = 0
    f = 10 * Hz
    # -- Создаём группу нейронов --
    neurons = NeuronGroup(
        N,
        '''
        dv/dt = (v_rest - v + R*I_ext)/tau : volt
        I_ext = I0 * sin(2 * pi * f * t + phi) : amp
        I0 : amp
        ''',
        threshold="v > v_threshold",
        reset="v = v_reset",
        method="euler",
    )


    # Шаг 1. Вычисляем метрики и получаем список лучших узлов
    if p_input is None:
        p_input = 1.0

    if isinstance(C_total, np.ndarray):
        C_total_matrix = C_total
        C_total = nx.from_numpy_array(C_total)
    elif isinstance(C_total, nx.Graph):
        C_total = C_total
        C_total_matrix = nx.to_numpy_array(C_total)
    else:
        raise ValueError("data должен быть либо numpy-массивом, либо объектом networkx.Graph")


    centrality = print_centrality(C_total_matrix, N1, n_neurons, p_input, boost_factor, test_num, num_tests, measure_name=measure_name, direct_1_2=True)


    if isinstance(centrality, np.ndarray):
        centrality = centrality.tolist()
    centrality  = sorted(list(centrality))

    # # Назначение параметров модуляции для соответствующих временных интервалов:
    # if centrality:
    neurons.I0[centrality] = I0_value * pA

    percent_central = len(centrality) * 100 / N1
    
    print(measure_name)
    print(percent_central)
    # p_intra1 = p_within
    # p_intra2 = p_within
    # p_12     = p_between
    # p_21     = 0.02

    # # Веса соединения
    # w_intra1 = 1
    # w_intra2 = 1
    # w_12     = 1
    # w_21     = 1

    # n_half = n_neurons // 2


    input_rate = 1 * Hz
    input_group = PoissonGroup(n_neurons, rates=input_rate)
    syn_input = Synapses(input_group, neurons, on_pre='v_post += J')
    syn_input.connect()

    # S_intra1 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_intra1.connect(
    #     condition='i <= n_half and j <= n_half',
    #     p=p_intra1
    # )
    # S_intra1.w = w_intra1

    # # 2) Синапсы внутри 2-го кластера
    # S_intra2 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_intra2.connect(
    #     condition='i >= n_half and j >= n_half',
    #     p=p_intra2
    # )
    # S_intra2.w = w_intra2

    # # 3) Синапсы из 1-го кластера во 2-й
    # S_12 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_12.connect(
    #     condition='i < n_half and j >= n_half',
    #     p=p_12
    # )
    # S_12.w = w_12

    # # 4) Синапсы из 2-го кластера в 1-й
    # S_21 = Synapses(neurons, neurons, model='w : 1', on_pre='v_post += J2 * w')
    # S_21.connect(
    #     condition='i >= n_half and j < n_half',
    #     p=p_21
    # )
    # S_21.w = w_21


    # STDP или нет
    if use_stdp:
        tau_stdp = 20 * ms
        Aplus = 0.005
        Aminus = 0.005

        stdp_eqs = '''
        dApre/dt = -Apre / tau_stdp : 1 (event-driven)
        dApost/dt = -Apost / tau_stdp : 1 (event-driven)
        w : 1
        '''
        exc_synapses = Synapses(
            neurons, neurons,
            model=stdp_eqs,
            on_pre="""
                v_post += J * w
                Apre += Aplus
                w = clip(w + Apost, 0, 1)
            """,
            on_post="""
                Apost += Aminus
                w = clip(w + Apre, 0, 1)
            """,
        )
    else:
        exc_synapses = Synapses(neurons, neurons,
                                model="w : 1",
                                on_pre="v_post += J2 * w",
                                )

        
    inh_synapses = Synapses(neurons, neurons, 
                            model="w : 1", 
                            on_pre="v_post += -J2 * w", 
                            )
    

    # Генерация источников и целей
    N_E_2 = int(N_E / 2)
    N_I_2 = int(N_I / 2)
    N_2 = int(N / 2)
    rows = np.arange(0, N_E_2) 
    rows2 = np.arange(N_E_2, N_E_2 + N_I_2) 
    rows3 = np.arange(N_2, N_E_2 + N_2)
    rows4 = np.arange(N_E_2 + N_2, N)  
    mask = np.isin(np.arange(C_total_matrix.shape[0]), rows)  
    mask2 = np.isin(np.arange(C_total_matrix.shape[0]), rows2)  
    mask3 = np.isin(np.arange(C_total_matrix.shape[0]), rows3) 
    mask4 = np.isin(np.arange(C_total_matrix.shape[0]), rows4)  
    sources_exc1, targets_exc1 = np.where(C_total_matrix[mask, :] > 0)
    sources_inh1, targets_inh1 = np.where(C_total_matrix[mask2, :] > 0)
    sources_exc2, targets_exc2 = np.where(C_total_matrix[mask3, :] > 0)
    sources_inh2, targets_inh2 = np.where(C_total_matrix[mask4, :] > 0)
    sources_exc1 = rows[sources_exc1]  # Преобразует 0-4 в 20-24
    sources_inh1 = rows2[sources_inh1]  # Преобразует 0-4 в 20-24
    sources_exc2 = rows3[sources_exc2]  # Преобразует 0-4 в 20-24
    sources_inh2 = rows4[sources_inh2]  # Преобразует 0-4 в 20-24
    sources_exc = np.concatenate((sources_exc1, sources_exc2))
    targets_exc = np.concatenate((targets_exc1, targets_exc2))
    sources_inh = np.concatenate((sources_inh1, sources_inh2))
    targets_inh = np.concatenate((targets_inh1, targets_inh2))
    exc_synapses.connect(i=sources_exc, j=targets_exc)
    inh_synapses.connect(i=sources_inh, j=targets_inh)
    # exc_synapses.w = 1
    # inh_synapses.w = 1
    for idx in range(len(exc_synapses.i)):
        pre_neuron = exc_synapses.i[idx]
        post_neuron = exc_synapses.j[idx]
        if pre_neuron < N1 and post_neuron < n_neurons:
            exc_synapses.w[idx] = 1  # для синапсов внутри первого кластера
        else:
            exc_synapses.w[idx] = 1  # для остальных связей
    
    for idx in range(len(inh_synapses.i)):
        pre_neuron = inh_synapses.i[idx]
        post_neuron = inh_synapses.j[idx]
        if pre_neuron < N1 and post_neuron < n_neurons:
            inh_synapses.w[idx] = 1  # для синапсов внутри первого кластера
        else:
            inh_synapses.w[idx] = 1  # для остальных связей


    W = np.zeros((n_neurons, n_neurons))
    W[exc_synapses.i[:], exc_synapses.j[:]] = exc_synapses.w[:]
    W[inh_synapses.i[:], inh_synapses.j[:]] = inh_synapses.w[:]
    
    # Создание новой матрицы для отображения в градациях серого:
    # 0   -> отсутствие связи
    # 0.5 -> наличие связи
    # 1   -> наличие связи, если хотя бы один нейрон является центральным
    M = np.where(W != 0, 0.5, 0.0)


    # Мониторы  
    spike_monitor = SpikeMonitor(neurons)

    rate_monitor = None
    rate_monitor2 = None
    if plotting_flags.get('rates', False) or plotting_flags.get('psd', False) or plotting_flags.get('spectrogram', False):
        rate_monitor = PopulationRateMonitor(neurons[:int(n_neurons/2)])
    if plotting_flags.get('rates2', False) or plotting_flags.get('psd2', False) or plotting_flags.get('spectrogram', False):
        rate_monitor2 = PopulationRateMonitor(neurons[int(n_neurons/2):])

   
    trace = StateMonitor(neurons, 'v', record=True)

    print(f"Количество нейронов: {N}")
    # print(f"Количество возбуждающих синапсов: {exc_synapses.N}")
    # print(f"Количество тормозных синапсов: {inh_synapses.N}")
    # print(f"Количество возбуждающих синапсов: {len(exc_synapses.i)}")
    # print(f"Количество тормозных синапсов: {len(inh_synapses.i)}")

    avg_exc_synapses = len(exc_synapses.i)
    avg_inh_synapses = len(inh_synapses.i)
    # Запуск
    run(sim_time, profile=True)
    end_time = time.time()
    duration = end_time - start_time
    # print(f"Testing completed in {duration:.2f} seconds.")

    # Анализ спайков
    spike_times = spike_monitor.t / ms
    spike_indices = spike_monitor.i

    trace_times = trace.t / ms
    
    mask1 = spike_indices < N1
    mask2 = spike_indices >= N1
    spike_times1 = spike_times[mask1]
    spike_times2 = spike_times[mask2]

    x1 = trace.v[:n_neurons//2, :] / mV  # (форма: n_neurons//2, 1000)
    x2 = trace.v[n_neurons//2:, :] / mV  # (форма: n_neurons//2, 1000)

    trial0 = x1.T  # (1000, n_neurons//2)
    trial1 = x2.T  # (1000, n_neurons//2)

    time_series_v = np.stack((trial0, trial1), axis=-1)

    print("Форма 3D тензора:", time_series_v.shape)

    bins = np.arange(0, int(sim_time/ms) + 1, time_window_size)
    time_window_centers = (bins[:-1] + bins[1:]) / 2

    avg_neuron_spikes_cluster2_list = []
    start_cluster_neuron = n_neurons/2
    end_cluster_neuron = n_neurons
    for i in range(len(bins) - 1):
        start_t = bins[i]
        end_t = bins[i + 1]

        mask = (
            (spike_indices >= start_cluster_neuron) & (spike_indices < end_cluster_neuron) &
            (spike_times > start_t) & (spike_times < end_t)
        )
        filtered_spike_indices = spike_indices[mask]
        group2_spikes = len(filtered_spike_indices)
        avg_spikes = group2_spikes / (end_cluster_neuron - start_cluster_neuron)
        avg_neuron_spikes_cluster2_list.append(avg_spikes)

    # Построение графиков
    if plotting_flags.get('granger', False) and 'ax_granger' in plotting_flags:
        ax_granger = plotting_flags['ax_granger']
        ax_dtf = plotting_flags['ax_dtf']
        ax_pdc = plotting_flags['ax_pdc']
        plot_granger(time_series_v, ax_granger, ax_dtf, ax_pdc)

        
    if plotting_flags.get('spikes', False) and 'ax_spikes' in plotting_flags:
        ax_spikes = plotting_flags['ax_spikes']
        plot_spikes(ax_spikes, spike_times, spike_indices, time_window_size, t_range, sim_time,
                    oscillation_frequency, use_stdp, measure_name)

    if plotting_flags.get('rates', False) and 'ax_rates' in plotting_flags:
        ax_rates = plotting_flags['ax_rates']
        plot_rates(ax_rates, N1, N2, rate_monitor, t_range)

    if plotting_flags.get('rates2', False) and 'ax_rates2' in plotting_flags:
        ax_rates2 = plotting_flags['ax_rates2']
        plot_rates2(ax_rates2, N1, N2, rate_monitor2, t_range)

    if plotting_flags.get('psd', False) and 'ax_psd' in plotting_flags:
        ax_psd = plotting_flags['ax_psd']
        plot_psd(rate_monitor, N1, N2, ax_psd)

    if plotting_flags.get('psd2', False) and 'ax_psd2' in plotting_flags:
        ax_psd2 = plotting_flags['ax_psd2']
        plot_psd2(rate_monitor2, N1, N2, ax_psd2)

    if plotting_flags.get('spectrogram', False) and 'ax_spectrogram' in plotting_flags:
        ax_spectrogram = plotting_flags['ax_spectrogram']
        plot_spectrogram(rate_monitor, rate_monitor2, N1, N2, ax_spectrogram)


    return avg_neuron_spikes_cluster2_list, time_window_centers, C_total, spike_indices, centrality, p_in_measured, p_out_measured, max_rate, W, M, percent_central, avg_exc_synapses, avg_inh_synapses

# Флаги для построения графиков
do_plot_granger = True
do_plot_spikes = True
do_plot_rates = True
do_plot_rates2 = True
do_plot_psd = True
do_plot_psd2 = True
do_plot_spectrogram = True

spike_counts_second_cluster = {}
detailed_spike_data_for_3d = {}
spike_counts_second_cluster_for_input = {}
subplot_results = {}  # Ключ: measure_name, значение: кортеж (exc_synapses, inh_synapses, centrality, p_in_measured, p_out_measured, percent_central)


for current_time in simulation_times:
    sim_time = current_time * ms
    t_range = [0, current_time]

    # Подготовим структуры словарей
    for I0_value in I0_values:
        spike_counts_second_cluster[I0_value] = {}
        detailed_spike_data_for_3d[I0_value]  = {}
        spike_counts_second_cluster_for_input[I0_value] = {}

    # Циклы по частоте, I0, STDP
    for oscillation_frequency in oscillation_frequencies:
        for I0_value in I0_values:
            for use_stdp in use_stdp_values:
                print(f"\n### Запуск при I0={I0_value} пА, freq={oscillation_frequency} Гц, "
                        f"STDP={use_stdp}, Time={current_time} ms ###")

                # Создадим ключи верхнего уровня по measure_name И заодно по 'random'
                # (чтобы в дальнейшем вносить результаты для каждого measure_name отдельно)
                for measure_name in measure_names + ['random']:
                    spike_counts_second_cluster[I0_value].setdefault(measure_name, {})
                    detailed_spike_data_for_3d[I0_value].setdefault(measure_name, {})
                    spike_counts_second_cluster_for_input[I0_value].setdefault(measure_name, {})
                    # А внутри — под p_within
                    for p_within in p_within_values:
                        p_within_str = f"{p_within:.2f}"
                        spike_counts_second_cluster[I0_value][measure_name].setdefault(p_within_str, {})
                        detailed_spike_data_for_3d[I0_value][measure_name].setdefault(p_within_str, {})
                        spike_counts_second_cluster_for_input[I0_value][measure_name].setdefault(p_within_str, {})

                # --- Цикл по p_within ---
                for p_within in p_within_values:
                    p_within_str = f"{p_within:.2f}"

                    # --- Цикл по p_input ---
                    for p_input in p_input_values:
                        p_input = round(p_input, 2)
                        p_input_str = f"{p_input:.2f}"

                        # --- Цикл по measure_name ---
                        for measure_name in measure_names:
                            # Сначала запускаем симуляции для measure_name
                            centrality = None
                            measure_name_prev = None
                            C_total_prev = None
                            # Здесь будем накапливать GIF-кадры
                            images = []

                            # Список p_between
                            p_between_values = np.arange(0.01, p_within-0.01, 0.03)
                            # p_between_values = np.arange(0.05, p_within - 0.01, 0.05)
                            if len(p_between_values) == 0:
                                continue

                            for p_between in p_between_values:
                                p_between = round(p_between, 3)

                                # Создаём fig и gs, только если планируем что-то рисовать
                                fig = None
                                if any([do_plot_granger, do_plot_spikes, do_plot_rates, 
                                        do_plot_rates2, do_plot_psd, 
                                        do_plot_psd2, do_plot_spectrogram]):
                                    fig = plt.figure(figsize=(14, 12))
                                    # Настраиваем сетку подграфиков (пример на 3 строки x 6 столбцов)
                                    gs = fig.add_gridspec(ncols=6, nrows=3)

                                # Словарь с флагами для построения графиков
                                plotting_flags = {
                                    'granger': do_plot_granger,
                                    'spikes': do_plot_spikes,
                                    'rates': do_plot_rates,
                                    'rates2': do_plot_rates2,
                                    'psd': do_plot_psd,
                                    'psd2': do_plot_psd2,
                                    'spectrogram': do_plot_spectrogram,
                                }

                                # Если fig не None, создаём оси и пишем их в plotting_flags,
                                # чтобы внутри sim(...) можно было их получить
                                if fig is not None:
                                    

                                    if do_plot_spikes:
                                        ax_spikes = fig.add_subplot(gs[0, :])
                                        plotting_flags['ax_spikes'] = ax_spikes

                                    if do_plot_rates:
                                        ax_rates = fig.add_subplot(gs[1, 0])
                                        plotting_flags['ax_rates'] = ax_rates

                                    if do_plot_rates2:
                                        ax_rates2 = fig.add_subplot(gs[1, 1])
                                        plotting_flags['ax_rates2'] = ax_rates2

                                    if do_plot_psd:
                                        ax_psd = fig.add_subplot(gs[2, 0])
                                        plotting_flags['ax_psd'] = ax_psd

                                    if do_plot_psd2:
                                        ax_psd2 = fig.add_subplot(gs[2, 1])
                                        plotting_flags['ax_psd2'] = ax_psd2

                                    if do_plot_spectrogram:
                                        ax_spectrogram = fig.add_subplot(gs[2, 2])
                                        plotting_flags['ax_spectrogram'] = ax_spectrogram
                                        

                                    if do_plot_granger:
                                        ax_granger_1 = fig.add_subplot(gs[1, 3])
                                        ax_granger_2 = fig.add_subplot(gs[2, 3])
                                        ax_dtf_1 = fig.add_subplot(gs[1, 4])
                                        ax_dtf_2 = fig.add_subplot(gs[2, 4])
                                        ax_pdc_1 = fig.add_subplot(gs[1, 5])
                                        ax_pdc_2 = fig.add_subplot(gs[2, 5])
                                        # Сохраняем оси в словарь; внутри sim(...) вы будете делать:
                                        #   ax_granger = plotting_flags['ax_granger']
                                        #   ax_dtf     = plotting_flags['ax_dtf']
                                        #   ax_pdc     = plotting_flags['ax_pdc']
                                        plotting_flags['ax_granger'] = [ax_granger_1, ax_granger_2]
                                        plotting_flags['ax_dtf'] = [ax_dtf_1, ax_dtf_2]
                                        plotting_flags['ax_pdc'] = [ax_pdc_1, ax_pdc_2]

                                    # Присвоим заголовок для всей фигуры
                                    fig.suptitle(
                                        f'I0={I0_value} pA, p_input={p_input_str}, '
                                        f'p_within={p_within_str}, p_between={p_between}, '
                                        f'measure={measure_name}, Time={current_time} ms'
                                    )

                                # Список для средних спайков
                                avg_window_avg_neuron_spikes_cluster2_tests = []

                                for test_num in range(num_tests):
                                    print(measure_name)
                                    print(f'I0={I0_value} pA, p_within={p_within_str}, '
                                    f'p_input={p_input:.2f}, p_between={p_between}, '
                                    f'tест={test_num + 1}, Time={current_time} ms')
                                    # На последних тестах хотим видеть графики
                                    if test_num < num_tests - 1:
                                        # Отключаем рисование для экономии
                                        for key in [
                                            'granger','spikes','rates','rates2',
                                            'connectivity2','psd','psd2','spectrogram'
                                        ]:
                                            plotting_flags[key] = False
                                    else:
                                        # На последнем тесте всё включаем (или оставляем, как было)
                                        for key in [
                                            'granger','spikes','rates','rates2',
                                            'connectivity2','psd','psd2','spectrogram'
                                        ]:
                                            plotting_flags[key] = True

                                    # Запуск симуляции
                                    (avg_neuron_spikes_cluster2_list,
                                    time_window_centers,
                                    C_total,
                                    spike_indices,
                                    centrality,
                                    p_in_measured,
                                    p_out_measured, 
                                    max_rate, 
                                    W,
                                    M,
                                    percent_central,
                                    avg_exc_syn,
                                    avg_inh_syn
                                    ) = sim(
                                        p_within,
                                        p_between,
                                        J,
                                        J2,
                                        refractory_period,
                                        sim_time,
                                        plotting_flags,
                                        rate_tick_step,
                                        t_range,
                                        rate_range,
                                        cluster_labels,
                                        cluster_sizes,
                                        I0_value,
                                        oscillation_frequency,
                                        use_stdp,
                                        time_window_size,
                                        measure_names,
                                        boost_factor_list,
                                        test_num,
                                        num_tests,
                                        C_total_prev,
                                        p_within_prev=p_within,
                                        p_between_prev=p_between,
                                        p_input=p_input,
                                        measure_name=measure_name,
                                        measure_name_prev=measure_name_prev,
                                        centrality=centrality
                                    )
                                    # Усредняем по всем окнам
                                    if avg_neuron_spikes_cluster2_list is not None and len(avg_neuron_spikes_cluster2_list) > 0:
                                        avg_window_val = np.mean(avg_neuron_spikes_cluster2_list)
                                    else:
                                        avg_window_val = 0
                                    avg_window_avg_neuron_spikes_cluster2_tests.append(avg_window_val)

                                    # Если это последний тест И условия по p_input и p_between выполняются, сохраняем результат
                        
                                    if test_num == num_tests - 1 and (p_input == 0.2 and p_between == 0.13):
                                        # Сохраняем необходимые переменные для построения субплотов
                                        subplot_results[measure_name] = W, M, p_in_measured, p_out_measured, percent_central

                                    # Обновляем «предыдущие» данные (для STDP)
                                    C_total_prev = C_total.copy()
                                    measure_name_prev = measure_name

                                # Запись данных для 3D-графика (Time vs p_between vs AvgSpikes)
                                if time_window_centers is None:
                                    time_window_centers = np.array([])
                                if avg_neuron_spikes_cluster2_list is None:
                                    avg_neuron_spikes_cluster2_list = []

                                detailed_spike_data_for_3d[I0_value][measure_name][p_within_str][p_between] = {
                                    "time": time_window_centers.copy(),
                                    "spikes_list": avg_neuron_spikes_cluster2_list.copy()
                                }

                                # Считаем среднее по всем тестам
                                avg_window_avg_neuron_spikes_cluster2_tests = np.array(avg_window_avg_neuron_spikes_cluster2_tests)
                                mean_spikes = np.mean(avg_window_avg_neuron_spikes_cluster2_tests)

                                # Сохраняем в spike_counts_second_cluster
                                if p_between not in spike_counts_second_cluster[I0_value][measure_name][p_within_str]:
                                    spike_counts_second_cluster[I0_value][measure_name][p_within_str][p_between] = []
                                spike_counts_second_cluster[I0_value][measure_name][p_within_str][p_between].append(
                                    avg_window_avg_neuron_spikes_cluster2_tests
                                )
                                # Сохраняем в spike_counts_second_cluster_for_input
                                sc_input = spike_counts_second_cluster_for_input[I0_value][measure_name][p_within_str]
                                sc_input.setdefault(p_input_str, {})
                                sc_input[p_input_str][p_between] = {
                                    'mean_spikes': float(mean_spikes),
                                    'avg_exc_synapses': avg_exc_syn,   # добавляем значение возбуждающих синапсов
                                    'avg_inh_synapses': avg_inh_syn    # и значение тормозных синапсов
                                }
                                # Сохраняем кадр в GIF
                                if fig is not None:
                                    plt.tight_layout()
                                    buf = io.BytesIO()
                                    plt.savefig(buf, format='png')
                                    buf.seek(0)
                                    images.append(imageio.imread(buf))
                                    plt.close(fig)


                            # Конец цикла по p_between
                            if images:
                                gif_filename = (
                                    f'{directory_path}/gif_I0_{I0_value}freq_{oscillation_frequency}_'
                                    f'STDP_{"On" if use_stdp else "Off"}_p_within_{p_within_str}_'
                                    f'p_input_{p_input_str}_Time_{current_time}ms_{measure_name}.gif'
                                )
                                imageio.mimsave(gif_filename, images, duration=2000.0, loop=0)


                        # Конец цикла по measure_name, включая 'random'
                    # Конец цикла по p_input
                # Конец цикла по p_within

                # Построение старых 3D-графиков (Time vs p_between vs AvgSpikes) по каждому measure
                plot_3d_spike_data(
                    detailed_spike_data_for_3d,
                    measure_names + ['random'],  # включаем и random
                    p_within_values,
                    I0_value,
                    oscillation_frequency,
                    use_stdp,
                    current_time,
                    time_window_size,
                    max_rate_on_graph,
                    directory_path=directory_path,
                )
                
                # Построение 3D-графиков (p_input vs p_between vs avg_spikes) — две поверхности: measure & random
                plot_pinput_between_avg_spikes_with_std(
                    spike_counts_second_cluster_for_input,
                    spike_counts_second_cluster,
                    I0_value,
                    oscillation_frequency,
                    use_stdp,
                    current_time,
                    time_window_size,
                    max_rate_on_graph,
                    p_input_values, 
                    p_between_values,
                    directory_path=directory_path,
                    measure_names=measure_names + ['random'],
                )
                # Пример вызова функции
                plot_all_measures_vs_random_from_csv(
                    "results_ext_test1/avg_tests_avg_spikes_5000ms.csv",
                    p_input_values,
                    p_between_values,
                    max_rate=10,
                )
                plot_connectivity(subplot_results)


