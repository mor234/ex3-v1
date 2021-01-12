import time

import networkx as nx
import numpy as np
from networkx import shortest_path

from src.DiGraph import DiGraph
from src.GraphAlgo import GraphAlgo


def load_graph(file_name):
    """
    get file name containing a graph in json format,
    load it in to an object type GraphAlgo and return it.
    :param file_name: path of json file
    :return: GraphAlgo containing DiGraph of the json file graph
    """
    graph_algo = GraphAlgo()
    graph_algo.load_from_json(file_name)
    # graph_algo.plot_graph()
    return graph_algo


def convert_our_graph_to_networkx_graph(our_graph: DiGraph) -> nx.DiGraph:
    """
    receive graph in Type Digraph and convert to graph in type nx.DiGraph
    :param our_graph: graph in Type Digraph
    :return: graph in the type from netwarkx library
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(our_graph.nodes.keys())  # add the nodes
    for edge, w in our_graph.edges.items():
        # add the edges and their weights
        graph.add_weighted_edges_from([(edge[0], edge[1], w)])
    return graph


def time_our_shortest_compare(our_graph: GraphAlgo, nx_graph: nx.DiGraph, src: int, dest: int):
    """
    receive graph Type DiGraph and graph Type netwarkx.DiGraph .
    the method measure the time it take to calculate shortest_path between src and dest
    in each implementation,  checks if the function shortest_path between src and dest
    of each implementation return the same results.
    in other words: check our graph calculation is correct,
    and print the results
    :param our_graph:  graph Type DiGraph
    :param nx_graph:  graph Type netwarkx.DiGraph
    :param src: key of source node of the path
    :param dest: key of destination node of the path
    :return:
    """
    start = time.time()
    our_ans = our_graph.shortest_path(src, dest)
    end = time.time()
    our_time = end - start
    is_same_ans = False
    nx_length=-1;
    try:
        start = time.time()
        nx_path = shortest_path(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
        end = time.time()
        nx_length = nx.shortest_path_length(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')

    except Exception as e:
        print(e)
        end = time.time()
        # if the exception was caused dou to the fact their is no path between src and dest,
        # and keep result which also means it
        if our_ans == (float('inf'), []):
            is_same_ans = True

    nx_time = end - start
    if np.math.isclose(our_ans[0], nx_length):  # compare the length results
        print("length is the same: ", nx_length)
        is_same_ans = True  # if length is the same- it's the same. path could be different with the same length
        if nx_path == our_ans[1]:
            print("path is the same: ", nx_path)

    print("our time", our_time, "nx time", nx_time)
    print("the same as nx:", is_same_ans)


def check_same_ans_and_run_time(file_name: str, list):
    """
    receive file_name of json file and list of (src,dest)
    and calls to function to load the graph,
    to convert it to graph in type netwarkx.DiGraph,
    and to function that measure and print the time it took to preform each function
    and print whether both implementation returns the same result for the shortest path between src and dest
    :param file_name: path of json file containing a graph
    :param list: of (src,dest) to check the shortest path between
    :return:
    """
    our_graph_algo = load_graph(file_name)
    nx_graph = convert_our_graph_to_networkx_graph(our_graph_algo.get_graph())

    point = list[0]
    time_our_shortest_compare(our_graph_algo, nx_graph, point[0], point[1])
    point = list[1]
    time_our_shortest_compare(our_graph_algo, nx_graph, point[0], point[1])


def comp_run_time(file_name, key_comp_1, ):
    """
    The method load graph from json file, and calculate all strongly connected component in the graph, measure the
    time it takes and print the list of the connected components and the time.
    then, calculate separately up to 2 connected component for a given id of node
    (calculate which connected component it's part of)
    measure the time it takes and print the list of the connected component and the time.
    :param file_name: name of the file to load the graph from
    :param key_comp_1: the id of the first node to find the connectent component of.
    :return:
    """
    # load graphs
    our_graph_algo = load_graph(file_name)
    nx_graph = convert_our_graph_to_networkx_graph(our_graph_algo.get_graph())

    # nx method connected components time check
    start = time.time()
    ans_nx = nx.strongly_connected_components(nx_graph)
    end = time.time()
    nx_time = end - start
    print(list(ans_nx))
    print("nx_time", nx_time)

    # our method connected components time check
    start = time.time()
    all_comp = our_graph_algo.connected_components()
    end = time.time()
    our_time = end - start
    print("all_comp:", all_comp)
    print("all_comp time:", our_time)

    # correction check
    same = True
    for comp_nx in ans_nx:
        if not all_comp.__contains__(list(comp_nx)):
            same = False
            print("only in nx", comp_nx)
            break
    print("the same as nx: ", same)

    # our method connected component (id) time check
    start = time.time()
    print(key_comp_1)
    this_comp = our_graph_algo.connected_component(key_comp_1)
    end = time.time()
    comp1_time = end - start
    print("first comp:", this_comp)

    print("first comp time:", comp1_time)


def shortest_check():
    """
    this method call for each graph a function that measure performance of shortest path algorithm in this
    implementation and the performance of networkx implementation, performance measured in run time.

     :return:
     """
    print("graph G_10_80_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_10_80_1.json", [(4, 2), (1, 9)])

    print("graph G_100_800_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_100_800_1.json", [(1, 72), (98, 75)])

    print("graph G_1000_8000_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_1000_8000_1.json", [(10, 999), (19, 631)])

    print("graph G_10000_80000_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_10000_80000_1.json", [(9, 8030), (151, 9087)])

    print("graph G_20000_160000_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_20000_160000_1.json", [(9, 18030), (151, 19087)])

    print("graph G_30000_240000_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_30000_240000_1.json", [(1000, 22030), (151, 29087)])


def strongly_comp_check():
    """
    this method call for each graph a function that measure performance of connected_components() algorithm
    in this implementation and nx  implementation,performance measured in run time. also check if the result is the same
     :return:
    """

    print("graph G_10_80_1")
    comp_run_time("../Graphs_on_circle/G_10_80_1.json", 2)  # 0

    print("graph G_100_800_1")
    comp_run_time("../Graphs_on_circle/G_100_800_1.json", 3)  # 0

    print("graph G_1000_8000_1")
    comp_run_time("../Graphs_on_circle/G_1000_8000_1.json", 416)  # not working for single comp

    print("graph G_10000_80000_1")
    comp_run_time("../Graphs_on_circle/G_10000_80000_1.json", 1457)

    print("graph G_20000_160000_1")
    comp_run_time("../Graphs_on_circle/G_20000_160000_1.json", 6623)

    print("graph G_30000_240000_1")
    comp_run_time("../Graphs_on_circle/G_30000_240000_1.json", 22040)


if __name__ == '__main__':
    """this program calculate ,compare the results, and print run time of the algorithm shortest path in this 
    implantation and netwarkx implementation on several graphs.
    Also, calculate and print run time of the algorithm connected component on several graphs ,
    """
    shortest_check()
    strongly_comp_check()
