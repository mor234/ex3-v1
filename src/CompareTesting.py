import time

import networkx as nx
from networkx import shortest_path
import numpy as np

from src.DiGraph import DiGraph
from src.GraphAlgo import GraphAlgo


def create_graph():
    graph_algo = GraphAlgo()
    graph_algo.load_from_json("../data/A5")
    # graph_algo.plot_graph()
    return graph_algo


def convert_our_graph_to_networkx_graph(our_graph: DiGraph) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(our_graph.nodes.keys())
    for edge, w in our_graph.edges.items():
        graph.add_weighted_edges_from([(edge[0], edge[1], w)])

    print(graph.nodes)
    print(graph.edge_attr_dict_factory)
    return graph


def compare_correct_shortest(our_graph: GraphAlgo, nx_graph: nx.DiGraph, src: int, dest: int) -> bool:
    nx_length = nx.shortest_path_length(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
    nx_path = shortest_path(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
    our_length, our_path = our_graph.shortest_path(src, dest)
    # print("nx length ", nx_length)
    # print("our length ", our_length)
    # print("nx path ", nx_path)
    # print("our path", our_path)

    if np.math.isclose(our_length, nx_length):
        print("length is the same: ", nx_length)
    if nx_path == our_path:
        print("path is the same: ", nx_path)


def time_our_shortest(our_graph: GraphAlgo, nx_graph: nx.DiGraph, src: int, dest: int) :
    start = time.time()
    our_length, our_path = our_graph.shortest_path(src, dest)
    end = time.time()
    our_time = end - start
    start = time.time()
    nx_path = shortest_path(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
    end = time.time()
    nx_time = end - start
    return our_time, nx_time


if __name__ == '__main__':
    our_graph_algo = create_graph()
    nx_graph = convert_our_graph_to_networkx_graph(our_graph_algo.get_graph())
    compare_correct_shortest(our_graph_algo, nx_graph, 1, 7)
    print(time_our_shortest(our_graph_algo, nx_graph, 1, 7))
