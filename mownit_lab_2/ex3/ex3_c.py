import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import random


def set_up_graph_1_random(nodes_nr, edges_nr, min_weight, max_weight, E):

    found = False

    while not found:
        G = nx.gnm_random_graph(nodes_nr, edges_nr - 1)
        if nx.is_connected(G):
            found = True

    s, t = find_sem_edge(G, edges_nr - 1)
    G.add_edge(s, t)
    power = (s, t, E)

    edges_nr_new = len(G.edges)
    nodes_nr_new = len(G.nodes)

    for (n, m) in G.edges():
        G.edges[n, m]['weight'] = random.randint(min_weight, max_weight)
        G.edges[n, m]['direction'] = 1

    return G, power, edges_nr_new, nodes_nr_new


def set_up_graph_2_cubic(min_weight, max_weight, E):

    G = nx.cubical_graph()

    s, t = 0, 1
    G.add_edge(s, t)
    power = (s, t, E)

    edges_nr_new = len(G.edges)
    nodes_nr_new = len(G.nodes)

    for (n, m) in G.edges():
        G.edges[n, m]['weight'] = random.randint(min_weight, max_weight)
        G.edges[n, m]['direction'] = 1

    return G, power, edges_nr_new, nodes_nr_new


def set_up_graph_3_with_bridge(nodes_nr, edges_nr, min_weight, max_weight, E):

    found = False

    while not found:
        G1 = nx.gnm_random_graph(nodes_nr, edges_nr)
        if nx.is_connected(G1):
            found = True

    found = False

    while not found:
        G2 = nx.gnm_random_graph(nodes_nr, edges_nr)
        if nx.is_connected(G2):
            found = True

    G = nx.disjoint_union(G1, G2)

    G.add_edge(0, nodes_nr)

    edges_nr_new = len(G.edges)
    nodes_nr_new = len(G.nodes)


    edge_index = random.randint(0, edges_nr_new - 1)
    index = 0
    for edge in G.edges:
        if index is edge_index:
            (s, t) = edge
            break
        index += 1

    power = (s, t, E)

    for (n, m) in G.edges():
        G.edges[n, m]['weight'] = random.randint(min_weight, max_weight)
        G.edges[n, m]['direction'] = 1

    return G, power, edges_nr_new, nodes_nr_new


def set_up_graph_4_2d_net(x_axis, y_axis, min_weight, max_weight, E):

    found = False

    G = nx.empty_graph()

    for i in range(x_axis * y_axis):
        G.add_node(i)

    for i in range(1, x_axis * y_axis):

        if i % x_axis > 0:
            G.add_edge(i, i -1)
        if i % x_axis < x_axis - 1:
            G.add_edge(i, i + 1)
        if i > x_axis - 1:
            G.add_edge(i, i - x_axis)
        if i < (y_axis - 1) * x_axis:
            G.add_edge(i, i + x_axis)

    edges_nr_new = len(G.edges)
    nodes_nr_new = len(G.nodes)

    edge_index = random.randint(0, edges_nr_new - 1)
    index = 0
    for edge in G.edges:
        if index is edge_index:
            (s, t) = edge
            break
        index += 1

    power = (s, t, E)

    for (n, m) in G.edges():
        G.edges[n, m]['weight'] = random.randint(min_weight, max_weight)
        G.edges[n, m]['direction'] = 1

    return G, power, edges_nr_new, nodes_nr_new


# returns new edge to make sure there is a loop in graph, or if graph is complete returns edge (0, 1)
def find_sem_edge(G, edges_nr):

    if len(G.edges) >= sum(range(1, len(G.nodes))):
        return 0, 1

    for s in range(edges_nr):

        for t in range(s + 1, edges_nr):

            place_found = True
            for m, n in G.edges:

                if (s, t) == (m, n) or (t, s) == (m, n):
                    place_found = False
                    break

            if place_found:
                return s, t

    return -1, -1


def fill_nodes(matrix, G, filled):

    for node in G.nodes:

        e_iter = 0
        for edge in G.edges:

            if edge[0] == node:
                matrix[filled + node][e_iter] = -1.0

            if edge[1] == node:
                matrix[filled + node][e_iter] = 1.0

            e_iter += 1


def check_loop(loops, to_check):

    for i in loops:
        if set(to_check) == set(i):
            return False
    return True


def loop_recur(G, base_node, loops, current_loop, current_node, used_nodes):

    # if len(loops) >= len(G.edges) - len(G.nodes):
    #     return

    for node1, node2 in G.edges(current_node):

        if node2 in used_nodes:
            continue

        if len(current_loop) > 2 and node2 == base_node and check_loop(loops, current_loop):
            loops.append(current_loop.copy())

        if node2 in current_loop:
            continue

        current_loop.append(node2)
        loop_recur(G, base_node, loops, current_loop, node2, used_nodes)
        current_loop.remove(node2)


def find_loops(G):

    loops = []
    used_nodes = []

    for base_node in range(len(G.nodes)):
        loop_recur(G, base_node, loops, [base_node], base_node, used_nodes)
        used_nodes.append(base_node)
        # if len(loops) >= len(G.edges) - len(G.nodes):
        #     break

    return loops


def fill_loops(A_matrix, B_matrix, G, filled, power, loops):

    for loop in range(len(loops)):

        for i in range(len(loops[loop])):

            if i < len(loops[loop]) - 1:
                edge = (loops[loop][i], loops[loop][i + 1])
            else:
                edge = (loops[loop][i], loops[loop][0])

            index = 0
            for (n, m) in G.edges:

                in_loop = False
                if edge[0] == n and edge[1] == m:

                    A_matrix[filled][index] = 1 * G.edges[n, m]['weight']
                    in_loop = True

                if edge[0] == m and edge[1] == n:
                    A_matrix[filled][index] = (-1) * G.edges[n, m]['weight']
                    in_loop = True

                if in_loop and ((edge[0] == power[0] and edge[1] == power[1]) or (edge[0] == power[1] and edge[1] == power[0])):
                    B_matrix[filled] = (-1) * power[2]

                index += 1
        filled += 1

    return filled


def adjust_directions(G, result):

    index = 0
    for n, m in G.edges:

        if result[index] < 0:
            G.edges[n, m]['direction'] = -1

        index += 1

    G2 = nx.DiGraph()
    G2.add_nodes_from(G.nodes)

    index = 0
    for n, m in G.edges:

        if G.edges[n, m]['direction'] is 1:
            G2.add_edge(n, m, value=abs(result[index]), weight=G.edges[n, m]['weight'], colour=int(0))
        else:
            G2.add_edge(m, n, value=abs(result[index]), weight=G.edges[n, m]['weight'], colour=int(0))

        index += 1

    return G2


def get_colours(G):

    max_curr = 0

    for curr in [G[n][m]['value'] for n, m in G.edges()]:
        if curr > max_curr:
            max_curr = curr

    cmap = plt.cm.get_cmap('OrRd')  # coolwarm
    rgbw = cmap(np.linspace(0, 1, 100))

    for n, m in G.edges():
        G[n][m]['colour'] = rgbw[int(99 * G[n][m]['value'] / float(int(max_curr) + 1))]

    return [G[n][m]['colour'] for n, m in G.edges()], cmap, max_curr


def draw_wire(G, power, pos):
    # pos = nx.spring_layout(G2, iterations=100)
    labels = nx.get_edge_attributes(G2, 'weight')

    for edge in labels:
        if (edge[0] is power[0] and edge[1] is power[1]) or (edge[0] is power[1] and edge[1] is power[0]):
            labels[edge] = str(labels[edge]) + ", " + str(power[2]) + " V"
            break

    node_labels = {}
    for number, node in enumerate(G2.nodes()):
        node_labels[node] = number

    nx.draw_networkx_nodes(G2, pos, node_size=100, node_color='pink')
    nx.draw_networkx_labels(G2, pos, node_labels, font_size=7)

    # nx.draw_networkx_nodes(G2, pos, node_size=10, node_color='pink')

    colours, cmap, max_curr = get_colours(G2)

    bar_edges = nx.draw_networkx_edges(G2, pos, edgelist=G.edges(data=True), edge_color=colours)

    # nx.draw_networkx_edge_labels(G2, pos, edge_labels=labels, font_size=7)
    nx.draw_networkx_edge_labels(G2, pos, edge_labels=labels, font_size=6)

    pc = mpl.collections.PatchCollection(bar_edges, cmap=cmap)
    edge_colors = range(0, int(max_curr) + 2)
    pc.set_array(edge_colors)
    plt.colorbar(pc, shrink=0.8)

    ax = plt.gca()
    ax.set_axis_off()

    plt.show()


def check_nodes(G):

    nodes_valid = True

    for node in G.nodes:

        sum = 0

        for n, m in G.edges:

            if node is n:
                sum -= G[n][m]['value']

            if node is m:
                sum += G[n][m]['value']

        print('node %-5d-> %7.3f' % (node, sum))

        if abs(sum) > 1.0e-10:
            nodes_valid = False

    if nodes_valid:
        print("          -> Nodes valid. Kirchhoff's current law is met.")
    else:
        print("          -> Nodes are NOT valid. Kirchhoff's current law is NOT met.")


def check_loops(G, loops, power):

    loops_valid = True

    for loop in loops:

        sum = 0

        for i in range(len(loop)):

            if i < len(loop) - 1:
                edge = (loop[i], loop[i + 1])
            else:
                edge = (loop[i], loop[0])

            e_index = 0
            for (n, m) in G.edges:

                was_used = False

                if edge[0] == n and edge[1] == m:
                    sum += G.edges[n, m]['value'] * G.edges[n, m]['weight']
                    was_used = True
                    # print(str(edge) + " -> " + str(result[e_index]) + " * " + str(G.edges[n, m]['weight']))

                if edge[0] == m and edge[1] == n:
                    sum -= G.edges[n, m]['value'] * G.edges[n, m]['weight']
                    was_used = True

                if was_used and ((edge[0] == power[0] and edge[1] == power[1]) or (edge[0] == power[1] and edge[1] == power[0])):
                    sum += power[2]

                e_index += 1

        print('loop %-60s-> %7.3f' % (str(loop), sum))

        if abs(sum) > 1.0e-10:
            loops_valid = False

    if loops_valid:
        print('%65s<- ' % "Nodes valid. Kirchhoff\'s voltage law is met. ")
    else:
        print('%65s<- ' % "Nodes are NOT valid. Kirchhoff\'s voltage law is NOT met. ")


min_weight = 1
max_weight = 10
E = 100
# G = nx.read_edgelist('../graphs/facebook_combined.txt')
# power = (0, 1, E)

# G, power, edges_nr, nodes_nr = set_up_graph_1_random(8, 10, min_weight, max_weight, E)
# G, power, edges_nr, nodes_nr = set_up_graph_2_cubic(min_weight, max_weight, E)
# G, power, edges_nr, nodes_nr = set_up_graph_3_with_bridge(6, 9, min_weight, max_weight, E)
G, power, edges_nr, nodes_nr = set_up_graph_4_2d_net(4, 5, min_weight, max_weight, E)

pos = nx.spring_layout(G)

print(nx.info(G))
print("SEM: (" + str(power[0]) + ", " + str(power[1]) + "), E = " + str(power[2]) + " V\n")

loops = find_loops(G)

A = np.zeros((nodes_nr + len(loops), edges_nr))
B = np.zeros(nodes_nr + len(loops))

filled = fill_loops(A, B, G, 0, power, loops)
fill_nodes(A, G, filled)

AT = np.transpose(A)

A2 = np.matmul(AT, A)
B2 = np.matmul(AT, B)

np.set_printoptions(precision=3, suppress=True)

result = np.linalg.solve(A2, B2)

G2 = adjust_directions(G, result)

for n, m in G2.edges:
    print('%-8s: value -> %8.3f, weight -> %2d' % (str((n, m)), G2[n][m]['value'], G2[n][m]['weight']))
print()

draw_wire(G2, power, pos)

check_nodes(G2)
print()
check_loops(G2, loops, power)
