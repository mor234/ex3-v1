import json
from typing import List
# import networkx as nx
from numpy import random
import matplotlib.pyplot as plt
import numpy as np

from DiGraph import DiGraph
from src.GraphAlgoInterface import GraphAlgoInterface
from src.GraphInterface import GraphInterface


class GraphAlgo(GraphAlgoInterface):
    def __init__(self, graph: DiGraph = None):
        self.graph = graph

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
        try:
            with open(file_name, 'r') as fs:  # excepton hendel
                j = json.load(fs)
                fs.close()
            g = DiGraph()
            for node in j['Nodes']:
                pos=None
                if 'pos' in node:
                    pos = node['pos'].split(',')
                    pos = (float(pos[0]), float(pos[1]), float(pos[2]))
                g.add_node(node['id'], pos)

            for edge in j['Edges']:
                g.add_edge(edge['src'], edge['dest'], edge['w'])

            self.graph = g
            return True
        except Exception as e:
            print(e)
            return False

    def save_to_json(self, file_name: str) -> bool:
        """
        Saves the graph in JSON format to a file
        @param file_name: The path to the out file
        @return: True if the save was successful, False o.w.
        """
        # create organize dictionary to save

        nodes_dict = []
        edges = []
        if self.graph is not None:
            for node in self.graph.nodes.values():
                single_node_dict = node.__dict__
                del single_node_dict["visited"]
                del single_node_dict["is_part_of_scc"]
                del single_node_dict["tag"]
                if single_node_dict["pos"] is not None:
                    single_node_dict["pos"] = ','.join(list(single_node_dict["pos"]))
                else:
                    del single_node_dict["pos"]
                nodes_dict.append(single_node_dict)

            class Edge():
                def __init__(self, src: int, dest: int, w: float):
                    self.src = src
                    self.dest = dest
                    self.w = w

            edges = [Edge(edge[0], edge[1], weight) for edge, weight in self.graph.edges.items()]
            # end if
        graph_to_save = {"Nodes": nodes_dict, "Edges": edges}

        try:
            with open(file_name, 'w') as f:
                json.dump(graph_to_save, default=lambda m: m.__dict__, fp=f, indent=4)
                f.close()
            return True
        except IOError as e:
            print(e)
            return False

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

        self.graph.nodes[id1].tag = 0
        distance_queue = [(node.tag, node.id) for node in self.graph.nodes.values()]
        distance_queue.sort(reverse=True)

        # find the dist using diextra algorithm
        while len(distance_queue) > 0:
            tag_id = distance_queue.pop()  # get the node with the shortest distance
            curr_node = self.graph.nodes[tag_id[1]]
            curr_node.visited = True
            curr_dist = tag_id[0]
            if curr_node.id == id2:
                ans_dist = curr_dist  # maybe error, immutable
                break;

            nodes_from = self.graph.all_out_edges_of_node(curr_node.id)
            for dest_node_key in nodes_from:
                node = self.graph.nodes[dest_node_key]
                if not node.visited:
                    if curr_dist + nodes_from[dest_node_key] < node.tag:
                        # update tag
                        tag_before_change = node.tag
                        node.tag = curr_dist + nodes_from[dest_node_key]
                        distance_queue.remove((tag_before_change, node.id))
                        distance_queue.append((node.tag, node.id))
            distance_queue.sort(reverse=True)

        # end of loop
        # if there is no path to id2 from id1
        if ans_dist == float("inf"):
            return ans_dist, ans_path

        # find the path
        curr_node = self.graph.nodes.get(id2)
        ans_path.insert(0, curr_node.id)  # add to the start of the list path
        while curr_node.id is not id1:
            nodes_to = self.graph.all_in_edges_of_node(curr_node.id)
            for neighbor_key in nodes_to:
                node=self.graph.nodes[neighbor_key]
                if np.math.isclose((curr_node.tag - nodes_to[neighbor_key]), node.tag):
                    curr_node = node
                    break
            ans_path.insert(0, curr_node.id)# add the node to the start of the list

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

        stack_nodes_from = []
        neighbors_list = self.graph.all_out_edges_of_node(curr_node.id)
        """
        """
        flag_visited_all_nei = True
        while True:
            for key_node in neighbors_list:
                node = self.graph.nodes[key_node]
                if not node.visited:  # if the node still hasn't reached ll his neighbors
                    if np.math.isclose(node.tag, float("inf")):  # if not this is the first time arriving to the node
                        node.tag = curr_node.id  # sign which node it came from.
                        curr_node = node
                        neighbors_list = self.graph.all_out_edges_of_node(curr_node.id)
                        flag_visited_all_nei = False
                        break


            # don't have any neighbors one got to.
            if flag_visited_all_nei:
                stack_nodes_from.append(curr_node.id)
                curr_node.visited = True
                if curr_node.id is id1:  # if all the neighbors of id1 are visited
                    break
                curr_node = self.graph.nodes[curr_node.tag]  # go back
                neighbors_list = self.graph.all_out_edges_of_node(curr_node.id)


            else:
                flag_visited_all_nei = True

        self.init_tag_visited()

        curr_node = self.graph.nodes[id1]
        neighbors_list = self.graph.all_in_edges_of_node(curr_node.id)
        curr_node.tag = curr_node.id


        stack_nodes_to = []

        flag_visited_all_nei = True

        while True:
            for key_node in neighbors_list:
                node = self.graph.nodes[key_node]
                if not node.visited:  # if the node still hasn't reached ll his neighbors
                    if np.math.isclose(node.tag,
                                       float("inf")):  # if not this is the first time arriving to the node
                        node.tag = curr_node.id  # sign which node it came from.
                        curr_node = node
                        neighbors_list = self.graph.all_in_edges_of_node(curr_node.id)
                        flag_visited_all_nei = False
                        break

            # don't have any neighbors one got to.
            if flag_visited_all_nei:
                stack_nodes_to.append(curr_node.id)
                curr_node.visited = True
                if curr_node.id is id1:  # if all the neighbors of id1 are visited
                    break
                curr_node = self.graph.nodes[curr_node.tag]  # go back
                neighbors_list = self.graph.all_in_edges_of_node(curr_node.id)


            else:
                flag_visited_all_nei = True

        ans=list(set(stack_nodes_from)&set(stack_nodes_to))
        for key_node in ans:
            self.graph.nodes[key_node].is_part_of_scc=True

        return ans

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
        for node_key, node in self.graph.nodes.items():
            if not node.is_part_of_scc:
                list_of_scc.append(self.connected_component(node_key))

        return list_of_scc  # what is List???

    def plot_graph(self) -> None:
        """
        Plots the graph.
        If the nodes have a position, the nodes will be placed there.
        Otherwise, they will be placed in a random but elegant manner.
        @return: None
        """

        # https://colab.research.google.com/github/makeabilitylab/signals/blob/master/Tutorials/IntroToMatplotlib.ipynb#scrollTo=DZBaqyORpHIS

        g = self.graph
        nodes = g.nodes

        index_for_node = {}
        # x = random.rand()
        x_nodes = []
        y_nodes = []
        i = 0
        for node in nodes.values():
            if node.pos is None:
                x_nodes.append(random.rand())
                y_nodes.append(random.rand())
            else:
                x_nodes.append(node.pos[0])
                y_nodes.append(node.pos[1])

            # label = "{:}".format(node.id)
            # ax = plt.subplots()
            # ax.text(x_nodes[i], y_nodes[i]+0.5,label)



            index_for_node[node.id] = i
            i += 1



        plt.xlabel("x ax is ")
        plt.ylabel("y ax is ")
        plt.title("The title of the graph")

        for (id1, id2), weight in g.edges.items():
            index_src = index_for_node[id1]
            index_dest = index_for_node[id2]
            # paint arrows
            plt.annotate('', xy=(x_nodes[index_src], y_nodes[index_src]), xycoords='data',
                         xytext=(x_nodes[index_dest], y_nodes[index_dest]), textcoords='data',
                         arrowprops=dict(facecolor='black', arrowstyle='<| -'))

        for node in self.graph.nodes.values():
            label = "{:}".format(node.id)
            i = index_for_node[node.id]

            plt.annotate(label,  # this is the text
                         xy=(x_nodes[i], y_nodes[i]),  # this is the point to label
                         xytext=(0, 7), ha='center',
                         textcoords="offset points",  # how to position the text
                         # ha='center',  # horizontal alignment can be left, right or center
                         color="lime")
        plt.scatter(x_nodes, y_nodes, label="vertx", color='red', s=100)

            #
            # plt.arrow(x_nodes[index_src], y_nodes[index_src], x_nodes[index_dest] - x_nodes[index_src],
            #           y_nodes[index_dest] - y_nodes[index_src]
            #           , width=0.0001, head_width=0.01, length_includes_head=True, color='yellow')


        #     middle_point_x = (x_nodes[index_dest] + x_nodes[index_src]) / 2
        #     middle_point_y = (y_nodes[index_dest] + y_nodes[index_src]) / 2
        #     print(middle_point_x, middle_point_y)
        #
        #     label = "{:.2f}".format(weight)
        #     # print(label)
        #     if(index_src>index_dest):
        #         print("f")
        #     text_params = {'ha': 'center', 'va': 'center', 'family': 'sans-serif',
        #                        'fontweight': 'bold'}
        #
        #     # plt.text(middle_point_x, middle_point_y,label, color='b', text_params)
        #
        #     # plt.annotate(text=label,  # this is the text
        #     #              xy=(middle_point_x, middle_point_y,),  # this is the point to label
        #     #              textcoords="offset points",  # how to position the text
        #     #              color="blue")
        #
        # Showing the graph
        plt.legend()
        plt.show()
