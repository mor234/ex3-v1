from typing import List

from DiGraph import DiGraph
from src.GraphAlgoInterface import GraphAlgoInterface
from src.GraphInterface import GraphInterface
import heapq


class GraphAlgo(GraphAlgoInterface):
    def __init__(self):
        self.graph = DiGraph()

    def get_graph(self) -> GraphInterface:
        """
        :return: the directed graph on which the algorithm works on.
        """
        return self.graph

    def load_from_json(self, file_name: str) -> bool:
        """
        Loads a graph from a json file.
        @param file_name: The path to the json file
        @returns True if the loading was successful, False o.w.
        """
        raise NotImplementedError

    def save_to_json(self, file_name: str) -> bool:
        """
        Saves the graph in JSON format to a file
        @param file_name: The path to the out file
        @return: True if the save was successful, False o.w.
        """
        raise NotImplementedError

    def init_is_part_of_scc(self):
        """

          :return:
        """
        for node in self.graph.nodes.values():
            node.is_part_of_scc = False;

    def init_tag_visited(self):
        """

        :return:
        """
        for node in self.graph.nodes.values():
            node.tag = float('inf')
            node.visited = False

    def shortest_path(self, id1: int, id2: int) -> (float, list):
        """
        Returns the shortest path from node id1 to node id2 using Dijkstra's Algorithm
        @param id1: The start node id
        @param id2: The end node id
        @return: The distance of the path, a list of the nodes ids that the path goes through


        Example:
#      >>> from GraphAlgo import GraphAlgo
#       >>> g_algo = GraphAlgo()
#        >>> g_algo.addNode(0)
#        >>> g_algo.addNode(1)
#        >>> g_algo.addNode(2)
#        >>> g_algo.addEdge(0,1,1)
#        >>> g_algo.addEdge(1,2,4)
#        >>> g_algo.shortestPath(0,1)
#        (1, [0, 1])
#        >>> g_algo.shortestPath(0,2)
#        (5, [0, 1, 2])

        Notes:
        If there is no path between id1 and id2, or one of them dose not exist the function returns (float('inf'),[])
        More info:
        https://en.wikipedia.org/wiki/Dijkstra's_algorithm
        """

        ans_dist = float('inf')
        ans_path = []
        if self.graph is None:
            return ans_dist, ans_path  # if the graph is None
        if self.graph.v_size() == 0:
            return ans_dist, ans_path  # if the graph is empty

        if id1 not in self.graph.nodes or id2 not in self.graph.nodes:
            return ans_dist, ans_path  # if the src node or dest node not in the graph

        self.init_tag_visited()  # initialize all the nodes tags to float('inf'), visited- to false
        distance_queue = [(node.tag, node) for node in self.graph.nodes]
        heapq.heapify(distance_queue)

        # find the dist using diextra algorithm
        while len(distance_queue) > 0:
            tuple_tag_node = heapq.heappop(distance_queue)  # get the node with the shortest distance
            curr_node = tuple_tag_node(1)
            curr_node.visited = True
            curr_dist = tuple_tag_node(0)
            if curr_node.id == id2:
                ans_dist = curr_dist  # maybe error, immutable
                break;

            nodes_from = self.graph.all_out_edges_of_node(curr_node.id)
            for dest_node_key in nodes_from:
                node = self.graph.nodes.get(dest_node_key)
                if not node.visited:
                    if curr_dist + nodes_from[dest_node_key] < node.tag:
                        # update tag
                        node.tag = curr_dist + nodes_from[dest_node_key]
                        distance_queue.remove(dest_node_key)
                        distance_queue.append((node.tag, node))

            # 1. Pop every item
            # while len(unvisited_queue):
            # heapq.heappop(unvisited_queue)

            heapq.heapify(distance_queue)

        # if there is no path to id2 from id1
        if ans_dist == float("inf"):
            return ans_dist, ans_path

        # find the path
        curr_node = self.graph.nodes.get(id2)
        curr_dist = curr_node.tag
        ans_path.insert(0, curr_node.id)
        while curr_node.id != id1:
            node = self.graph.nodes.get(node_key)
            nodes_to = self.graph.all_in_edges_of_node(curr_node.id)
            for node_key in nodes_to:
                if curr_dist - nodes_from[node_key] == node.tag:
                    curr_node = node
                    curr_dist = curr_node.tag
                    break

            ans_path.insert(0, curr_node.id)

        return ans_dist, ans_path

    def connected_component(self, id1: int) -> list:
        """
        Finds the Strongly Connected Component(SCC) that node id1 is a part of.
        @param id1: The node id
        @return: The list of nodes in the SCC

        Notes:
        If the graph is None or id1 is not in the graph, the function should return an empty list []
        """
        if self.graph is None:
            return []
        if id1 not in self.graph.nodes:
            return []

        self.init_tag_visited()

        curr_node = self.graph.nodes[id1]
        curr_node.tag = curr_node.id

        stack = []
        neighbors_list = self.graph.all_out_edges_of_node(curr_node.id)
        """


        """
        while True:
            for node in neighbors_list:
                if not node.visited:  # if the node still hasn't reached ll his neighbors
                    if node.tag is float("inf"):  # if not this is the first time arriving to the node
                        node.tag = curr_node.id  # sign which node it came from.
                        curr_node = node
                        neighbors_list = self.graph.all_out_edges_of_node(curr_node.id)

            # don't have any neighbors one got to.
            stack.append(curr_node)
            curr_node.visited = True
            if curr_node.id is id1:  # if all the neighbors of id1 are visited
                break
            curr_node = self.graph.nodes[curr_node.tag]  # go back

            self.init_tag_visited()
            curr_node = stack[-1]
            curr_node.tag = curr_node.id

        strongly_component = [curr_node]

        while True:
            neighbors_list = self.graph.all_in_edges_of_node(curr_node.id)
            for node in neighbors_list:
                if not node.visited:  # if the node still hasn't reached ll his neighbors
                    if node.tag is float("inf"):  # if not this is the first time arriving to the node
                        node.tag = curr_node.id  # sign which node it came from.
                        curr_node = node
                        strongly_component.append(curr_node)
                        curr_node.is_part_of_scc = True
                        neighbors_list = self.graph.all_in_edges_of_node(curr_node.id)

            # don't have any neighbors one got to.
            curr_node.visited = True
            if curr_node.id is id1:  # if all the neighbors of id1 are visited
                return strongly_component
            curr_node = self.graph.nodes[curr_node.tag]  # go back

    def connected_components(self) -> List[list]:
        """
        Finds all the Strongly Connected Component(SCC) in the graph.
        @return: The list all SCC

        Notes:
        If the graph is None the function should return an empty list []
        """
        if self.graph is None:
            return []
        self.init_is_part_of_scc()
        list_of_scc = []
        for node_key,node in self.graph.nodes:
            if not node.is_part_of_scc:
                list_of_scc.append(self.connected_component(node_key))

        return list_of_scc # what is List???

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """
        raise NotImplementedError
