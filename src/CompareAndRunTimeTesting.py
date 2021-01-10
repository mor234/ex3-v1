import time

import networkx as nx
import numpy as np
from networkx import shortest_path

from src.DiGraph import DiGraph
from src.GraphAlgo import GraphAlgo
from src.JavaResults import JavaResults as j_res


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


def compare_correct_shortest(our_graph: GraphAlgo, nx_graph: nx.DiGraph, src: int, dest: int, java_ans) -> bool:
    """
    receive graph Type DiGraph and graph Type netwarkx.DiGraph.
    the method checks if the function shortest_path between src and dest
    of each implementation return the same results.
    in other words: check our graph calculation is correct
    :param our_graph:  graph Type DiGraph
    :param nx_graph:  graph Type netwarkx.DiGraph
    :param src: key of source node of the path
    :param dest: key of destination node of the path
    :return: True if returns the same results, False if not.
    """
    try:
        our_ans = None
        our_ans = our_graph.shortest_path(src, dest)
        nx_length = nx.shortest_path_length(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
        nx_path = shortest_path(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
    except Exception as e:
        print(e)
        # if the exception was caused dou to the fact their is no path between src and dest,
        # and return result which also means it
        if our_ans == (float('inf'), []):
            return True
        else:
            return False

    if np.math.isclose(our_ans[0], nx_length):  # compare the length results
        print("length is the same: ", nx_length)
        print (nx_path, "our:",our_ans[1],"java:", java_ans)
        if nx_path == our_ans[1] == java_ans:
            print("path is the same: ", nx_path)
            return True
    return False


def time_our_shortest(our_graph: GraphAlgo, nx_graph: nx.DiGraph, src: int, dest: int):
    """
    receive graph Type DiGraph and graph Type netwarkx.DiGraph.
    the method measure the time it take to calculate  shortest_path between src and dest
    in each implementation and returns it.
    :param our_graph:  graph Type DiGraph
    :param nx_graph:  graph Type netwarkx.DiGraph
    :param src: key of source node of the path
    :param dest: key of destination node of the path
    :return: the time it took to calculate in each implementation
    """
    start = time.time()
    our_graph.shortest_path(src, dest)
    end = time.time()
    our_time = end - start
    try:
        start = time.time()
        shortest_path(nx_graph, source=src, target=dest, weight='weight', method='dijkstra')
    except Exception as e:
        pass
    end = time.time()
    nx_time = end - start
    return ("our time", our_time, "nx time", nx_time)


def check_same_ans_and_run_time(file_name: str, list, java_ans1, java_ans2):
    """
    receive file_name of json file and list of (src,dest)
    and calls to function to load the graph,
    to convert it to graph in type netwarkx.DiGraph and if in both
    implementation returns the same result for the shortest path between src and dest,
    measure the time it take to couclate and print to the screen.

    :param file_name: path of json file containing a graph
    :param list: of (src,dest) to check the shortest path between
    :return:
    """
    our_graph_algo = load_graph(file_name)
    nx_graph = convert_our_graph_to_networkx_graph(our_graph_algo.get_graph())
    point = list[0]
    if compare_correct_shortest(our_graph_algo, nx_graph, point[0], point[1], java_ans1):
        print(time_our_shortest(our_graph_algo, nx_graph, point[0], point[1]))
    else:
        print("Error. did not return the same results")
    point = list[1]
    if compare_correct_shortest(our_graph_algo, nx_graph, point[0], point[1], java_ans2):
        print(time_our_shortest(our_graph_algo, nx_graph, point[0], point[1]))
    else:
        print("Error. did not return the same results")


def comp_run_time(file_name, key_comp_1, key_comp_2=None):
    """
    The method load graph from json file, and calculate all strongly connected component in the graph, measure the
    time it takes and print the list of the connected components and the time.
    then, calculate separately up to 2 connected component for a given id of node
    (calculate which connected component it's part of)
    measure the time it takes and print the list of the connected component and the time.
    :param file_name: name of the file to load the graph from
    :param key_comp_1: the id of the first node to find the connected
    component of
    :param key_comp_2: the id of the second node to find the connected component of
    :return:
    """
    graph_algo = load_graph(file_name)
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


def shortest_check():
    """
    this method call for each graph a function that measure performance of shortest path algorithm in this
    implementation and the performance of networkx implementation, performance measured in run time.

     :return:
    """
    print("graph G_10_80_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_10_80_1.json", [(4, 2), (1, 9)],
                                j_res.graph_G_10_80_1_path_4_2, j_res.graph_G_10_80_1_path_1_9)

    print("graph G_100_800_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_100_800_1.json", [(1, 72), (98, 75)],
                                j_res.graph_G_100_800_1_path_1_72, j_res.graph_G_100_800_1_path_98_75)
    #
    print("graph G_1000_8000_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_1000_8000_1.json", [(10, 999), (19, 631)],
                                j_res.graph_G_1000_8000_1_path_10_999, j_res.graph_G_1000_8000_1_path_19_631)

    print("graph G_10000_80000_1")
    check_same_ans_and_run_time("../Graphs_on_circle/G_10000_80000_1.json", [(9, 8030), (151, 9087)],
                                j_res.graph_G_10000_80000_1_path_9_8030, j_res.graph_G_10000_80000_1_path_151_9087)
    #
    # print("graph G_20000_160000_1")
    # check_same_ans_and_run_time("../Graphs_on_circle/G_20000_160000_1.json", [(9, 18030), (151, 19087)],
    #                             j_res.graph_G_20000_160000_1_path_9_18030, j_res.graph_G_20000_160000_1_path_151_19087)
    #
    # print("graph G_30000_240000_1")
    # check_same_ans_and_run_time("../Graphs_on_circle/G_30000_240000_1.json", [(1000, 22030), (151, 29087)],
    #                             j_res.graph_G_30000_240000_1_path_1000_22030,
    #                             j_res.graph_G_30000_240000_1_path_151_29087)
    #

def strongly_comp_check():
    """
    this method call for each graph a function that measure performance of connected_components() algorithm
    in this implementation ,performance measured in run time.
     :return:
    """

    print("graph G_10_80_1")
    comp_run_time("../Graphs_on_circle/G_10_80_1.json", 0)
    #,j_res.graph_G_10_80_1_all_comp,j_res.g)

    print("graph G_100_800_1")
    comp_run_time("../Graphs_on_circle/G_100_800_1.json", 0)

    print("graph G_1000_8000_1")
    comp_run_time("../Graphs_on_circle/G_1000_8000_1.json", 0)

    print("graph G_10000_80000_1")
    comp_run_time("../Graphs_on_circle/G_10000_80000_1.json", 0, 238)

    print("graph G_20000_160000_1")
    comp_run_time("../Graphs_on_circle/G_20000_160000_1.json", 1209)

    print("graph G_30000_240000_1")
    comp_run_time("../Graphs_on_circle/G_30000_240000_1.json", 0, 238)


if __name__ == '__main__':
    """this program calculate ,compare the results, and print run time of the algorithm shortest path in this 
    implantation and netwarkx implementation on several graphs.
    Also, calculate and print run time of the algorithm connected component on several graphs ,
    compare the results to results from running this algorithm in java, 
    """
    shortest_check()
    # strongly_comp_check()
