import time

import networkx as nx
from networkx import shortest_path
import numpy as np

from src.DiGraph import DiGraph
from src.GraphAlgo import GraphAlgo


def comp_run_time(file_name, key_comp_1, key_comp_2=None):
    graph_algo = GraphAlgo()
    graph_algo.load_from_json(file_name)
    start = time.time()
    all_comp = graph_algo.connected_components()
    end = time.time()
    our_time = end - start
    print("all_comp:", all_comp)
    print("all_comp time:", our_time)

    start = time.time()
    this_comp = graph_algo.connected_component(key_comp_1)
    end = time.time()
    comp1_time = end - start
    print("first comp:", this_comp)
    print("first comp time:", comp1_time)

    if key_comp_2 is not None:
        start = time.time()
        this_comp = graph_algo.connected_component(key_comp_2)
        end = time.time()
        comp2_time = end - start
        print("second comp:", this_comp)
        print("second comp time:", comp2_time)


def create_graph(file_name):
    graph_algo = GraphAlgo()
    graph_algo.load_from_json(file_name)
    print(1)
    # graph_algo.plot_graph()
    return graph_algo


def convert_our_graph_to_networkx_graph(our_graph: DiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(our_graph.nodes.keys())
    for edge, w in our_graph.edges.items():
        graph.add_weighted_edges_from([(edge[0], edge[1], w)])
    print(2)
    return graph


def compare_correct_shortest(our_graph: GraphAlgo, nx_graph: nx.DiGraph, src: int, dest: int) -> bool:
    try:
        our_ans = our_graph.shortest_path(src, dest)
        nx_length = nx.shortest_path_length(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
        nx_path = shortest_path(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
    except Exception as e:
        if our_ans == (float('inf'), []):
            return True
        else:
            return False

    if np.math.isclose(our_ans[0], nx_length):
        print("length is the same: ", nx_length)
        if nx_path == our_ans[1]:
            print("path is the same: ", nx_path)
            return True
    return False


def time_our_shortest(our_graph: GraphAlgo, nx_graph: nx.DiGraph, src: int, dest: int):
    start = time.time()
    our_length, our_path = our_graph.shortest_path(src, dest)
    end = time.time()
    our_time = end - start
    try:
        start = time.time()
        nx_path = shortest_path(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
    except Exception as e:
        pass
    end = time.time()
    nx_time = end - start
    return ("our time", our_time, "nx time", nx_time)


def check_same_ans_and_run_time(file_name: str, list):
    our_graph_algo = create_graph(file_name)
    nx_graph = convert_our_graph_to_networkx_graph(our_graph_algo.get_graph())
    for point in list:
        if compare_correct_shortest(our_graph_algo, nx_graph, point[0], point[1]):
            print(time_our_shortest(our_graph_algo, nx_graph, point[0], point[1]))


def shortest_check():
    print("garph G_10_80_0")
    check_same_ans_and_run_time("../Graphs_no_pos/G_10_80_0.json", [(4, 2), (1, 9)])

    print("garph G_100_800_0")
    check_same_ans_and_run_time("../Graphs_no_pos/G_100_800_0.json", [(1, 72), (98, 75)])

    print("garph G_1000_8000_0")
    check_same_ans_and_run_time("../Graphs_no_pos/G_1000_8000_0.json", [(10, 1000), (19, 631)])

    print("garph G_10000_80000_0")
    check_same_ans_and_run_time("../Graphs_no_pos/G_10000_80000_0.json", [(9, 8030), (151, 9087)])

    print("garph G_20000_160000_0")
    check_same_ans_and_run_time("../Graphs_no_pos/G_20000_160000_0.json", [(9, 18030), (151, 19087)])

    print("graph G_30000_240000_0")
    # (1, 100000), (43, 120000), (1001, 20000)
    check_same_ans_and_run_time("../Graphs_no_pos/G_30000_240000_0.json", [(17000, 29000)])


def strongly_comp_check():
    # print("graph G_10_80_0")
    # comp_run_time("../Graphs_no_pos/G_10_80_0.json", 0)
    #
    # print("graph G_100_800_0")
    # comp_run_time("../Graphs_no_pos/G_100_800_0.json", 0)
    #
    # print("graph G_1000_8000_0")
    # comp_run_time("../Graphs_no_pos/G_1000_8000_0.json", 0)

    print("graph G_10000_80000_0")
    comp_run_time("../Graphs_no_pos/G_10000_80000_0.json", 0, 238)

    print("graph G_20000_160000_0")
    comp_run_time("../Graphs_no_pos/G_20000_160000_0.json", 0, 1209)

    print("graph G_30000_240000_0")
    # # (1, 100000), (43, 120000), (1001, 20000)
    comp_run_time("../Graphs_no_pos/G_30000_240000_0.json", 0, 238)


if __name__ == '__main__':
    # shortest_check()
    strongly_comp_check()
