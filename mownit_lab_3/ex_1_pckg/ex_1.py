import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import ex_1_pckg.cost_functions as cf


np.set_printoptions(precision=3, suppress=True)


def generate_start_edges(G):

    for i in range(0, len(G.nodes) - 1):
        G.add_edge(i, i + 1)
    G.add_edge(0, len(G.nodes) - 1)


def generate_graph_random(nodes_nr):

    G = nx.Graph()

    for i in range(nodes_nr):
        G.add_node(i, pos=(random.uniform(0, 35), random.uniform(0, 35)))

    generate_start_edges(G)

    return G


def generate_cluster_start_edges(G):

    taken = [random.randint(0, len(G.nodes) - 1)]

    while len(taken) is not len(G.nodes):

        x = random.randint(0, len(G.nodes) - 1)

        if x not in taken:
            G.add_edge(x, taken[-1])
            taken.append(x)

    G.add_edge(taken[0], taken[-1])


def generate_graph_clusters_4(cluster_nodes_nr):

    G = nx.Graph()

    for i in range(cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(0, 9), random.uniform(0, 9)))

    for i in range(cluster_nodes_nr, 2 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(26, 35), random.uniform(0, 9)))

    for i in range(2 * cluster_nodes_nr, 3 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(0, 9), random.uniform(26, 35)))

    for i in range(3 * cluster_nodes_nr, 4 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(26, 35), random.uniform(26, 35)))

    generate_cluster_start_edges(G)

    return G


def generate_graph_clusters_9(cluster_nodes_nr):

    G = nx.Graph()

    for i in range(cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(0, 3.5), random.uniform(0, 3.5)))

    for i in range(cluster_nodes_nr, 2 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(13.5, 17), random.uniform(0, 3.5)))

    for i in range(2 * cluster_nodes_nr, 3 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(0, 3.5), random.uniform(13.5, 17)))

    for i in range(3 * cluster_nodes_nr, 4 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(13.5, 17), random.uniform(13.5, 17)))

    for i in range(4 * cluster_nodes_nr, 5 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(27, 30.5), random.uniform(0, 3.5)))

    for i in range(5 * cluster_nodes_nr, 6 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(27, 30.5), random.uniform(13.5, 17)))

    for i in range(6 * cluster_nodes_nr, 7 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(27, 30.5), random.uniform(27, 30.5)))

    for i in range(7 * cluster_nodes_nr, 8 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(13.5, 17), random.uniform(27, 30.5)))

    for i in range(8 * cluster_nodes_nr, 9 * cluster_nodes_nr):
        G.add_node(i, pos=(random.uniform(0, 3.5), random.uniform(27, 30.5)))

    generate_cluster_start_edges(G)

    return G


def show_graph(G, colour):

    plt.axis('off')
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color=colour)

    # node_labels = {}
    # for number, node in enumerate(G.nodes()):
    #     node_labels[node] = number
    # nx.draw_networkx_labels(G, pos, node_labels, font_size=8)

    nx.draw_networkx_edges(G, pos, edge_color='grey')
    plt.show()
    plt.axis('on')


def count_euclid_distance(pos1, pos2):
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def make_euclid_matrix(G):

    result = np.zeros((len(G.nodes), len(G.nodes)))

    positions = nx.get_node_attributes(G, 'pos')

    n = 0
    for pos1 in positions:

        m = 0
        for pos2 in positions:
            if m > n:
                result[n][m] = count_euclid_distance(positions.get(pos1), positions.get(pos2))
            m += 1

        n += 1

    for n in range(len(G.nodes)):
        for m in range(n, len(G.nodes)):
            result[m][n] = result[n][m]

    return result


def count_energy(G, distances):

    sum = 0

    for edge in G.edges:
        sum += distances[edge[0], edge[1]]

    return sum


def generate_n_permutations_arbitrary(G, n):

    result = []

    for i in range(n):

        found = False

        while not found:

            index1 = random.randint(0, len(G.nodes) - 1)
            index2 = random.randint(0, len(G.nodes) - 1)

            while index1 == index2:
                index2 = random.randint(0, len(G.nodes) - 1)

            if (index1, index2) not in result:
                result.append((index1, index2))
                found = True

    return result


def generate_n_permutations_consecutive(G, n):

    # to działa nie za dobrze, chyba nie o to chodziło w consecutive

    result = []

    for i in range(n):

        found = False

        while not found:

            index1 = random.randint(0, len(G.nodes) - 1)

            if index1 is not len(G.nodes) - 1:
                index2 = index1 + 1
            else:
                index2 = 0

            if (index1, index2) not in result:
                result.append((index1, index2))
                found = True

    return result


def get_possibility_to_change(curr_energy, new_energy, temperature):

    if new_energy < curr_energy:
        return 1.0

    if temperature == 0:
        return 0.0

    return np.sin(np.exp((curr_energy - new_energy) / 4 / temperature) * np.pi / 2)


def choose_neighbour(G, permutations, curr_energy, temperature):

    neighbours = []

    for n, m in permutations:
        G.nodes[n]['pos'], G.nodes[m]['pos'] = G.nodes[m]['pos'], G.nodes[n]['pos']
        neighbours.append(((n, m), count_energy(G, make_euclid_matrix(G))))
        G.nodes[n]['pos'], G.nodes[m]['pos'] = G.nodes[m]['pos'], G.nodes[n]['pos']

    neighbours = sorted(neighbours, key=lambda t: t[1], reverse=True)

    for tup, energy in neighbours:
        if get_possibility_to_change(curr_energy, energy, temperature) > random.random():
            return tup, energy

    return (1, 1), curr_energy


def loop(G, perm_nr, iterations, T_function):

    best_G = G.copy()
    best_energy = count_energy(G, make_euclid_matrix(G))
    curr_energy = best_energy

    energies = [best_energy]
    T_list = []

    for i in range(iterations):

        T_list.append(T_function(i + 1))

        permutations = generate_n_permutations_arbitrary(G, perm_nr)

        (n, m), curr_energy = choose_neighbour(G, permutations, curr_energy, T_list[i])
        G.nodes[n]['pos'], G.nodes[m]['pos'] = G.nodes[m]['pos'], G.nodes[n]['pos']

        if curr_energy < best_energy:
            best_energy = curr_energy
            best_G = G.copy()

        energies.append(curr_energy)

    return best_G, energies, T_list


def loop_consecutve(G, perm_nr, iterations, T_function):

    # jeden z pomysłów co może oznaczać consecutive swap - w momencie w którym następuje wybór jakiejś permutacji
    # (nie wybrano aktualnego układu) zamrażam temperaturę aż do momentu pierwszego nie-wybrania żadnej permutacji.
    # Teoretycznie miałoby to "zachęcić" układ do wykonywania kolejnych następujących po sobie zamian, a zatem do tworzenia
    # na wykresie energii stoków idących w górę. W praktyce ma raczej znikomy wpływ na wynik dla wywołań majacych więcej niż
    # kikaset iteracji..

    best_G = G.copy()
    best_energy = count_energy(G, make_euclid_matrix(G))

    energies = [best_energy]
    T_list = []

    T_offset = 0

    for i in range(iterations):

        T_list.append(T_function(i + 1 + T_offset))

        permutations = generate_n_permutations_arbitrary(G, perm_nr)

        curr_energy = count_energy(G, make_euclid_matrix(G))
        (n, m), new_energy = choose_neighbour(G, permutations, curr_energy, T_list[i])

        if n is not m:
            T_offset -= 1
        else:
            T_offset = 0

        G.nodes[n]['pos'], G.nodes[m]['pos'] = G.nodes[m]['pos'], G.nodes[n]['pos']

        if new_energy < best_energy:
            best_energy = new_energy
            best_G = G.copy()

        energies.append(new_energy)

    return best_G, energies, T_list


# G = generate_graph_random(25)
# G = generate_graph_clusters_4(9)
G = generate_graph_clusters_9(5)
G2 = G.copy()

permutations = 4
iterations = 15000
starting_T = 0.95
T_function = cf.slow_linear(iterations, starting_T)

G, energies, T_list = loop(G, permutations, iterations, T_function)

print(count_energy(G2, make_euclid_matrix(G2)))
show_graph(G2, 'red')
print(count_energy(G, make_euclid_matrix(G)))
show_graph(G, 'orange')

x_axis = [i for i in range(len(energies))]
plt.plot(x_axis, energies)
plt.show()

# T_x_axis = [i for i in range(len(T_list))]
# plt.plot(T_x_axis, T_list)
# plt.show()

