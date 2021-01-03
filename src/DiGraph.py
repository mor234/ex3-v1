from src.GraphInterface import GraphInterface
from src.NodeData import NodeData


class DiGraph(GraphInterface):
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.mc = 0

    def v_size(self) -> int:
        """
        Returns the number of vertices in this graph
        @return: The number of vertices in this graph
        """
        return len(self.nodes)

    def e_size(self) -> int:
        """
        Returns the number of edges in this graph
        @return: The number of edges in this graph
        """
        return len(self.edges)

    def get_all_v(self) -> dict:
        """return a dictionary of all the nodes in the Graph, each node is represented using a pair
            (node_id, node_data)
         """
        return self.nodes

    def all_in_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected to (into) node_id ,
        each node is represented using a pair (other_node_id, weight)
        """
        in_edges_of_node = {}
        for key, weight in self.edges.items():  # need to check
            if id1 in key:
                if key.index(id1) == 1:
                    in_edges_of_node[key[0]] = weight

        return in_edges_of_node


    def all_out_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected from node_id , each node is represented using a pair
        (other_node_id, weight)
        """
        out_edges_of_node = {}
        for key, weight in self.edges.items():  # need to check
            if id1 in key:
                if key.index(id1) == 0:
                    out_edges_of_node[key[1]] = weight

        return out_edges_of_node

    def get_mc(self) -> int:
        """
        Returns the current version of this graph,
        on every change in the graph state - the MC should be increased
        @return: The current version of this graph.
        """
        return self.mc

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        """
        Adds an edge to the graph.
        @param id1: The start node of the edge
        @param id2: The end node of the edge
        @param weight: The weight of the edge
        @return: True if the edge was added successfully, False o.w.

        Note: If the edge already exists or one of the nodes dose not exists the functions will do nothing
        """
        if weight < 0:
            return False  # weight not supose to be negative

        if id1 in self.nodes and id2 in self.nodes:
            if (id1, id2) in self.edges:
                return False
            else:
                self.mc += 1
                self.edges[(id1, id2)] = weight
                return True
        else:
            return False

    def add_node(self, node_id: int, pos: tuple = None) -> bool:
        """
        Adds a node to the graph.
        @param node_id: The node ID
        @param pos: The position of the node
        @return: True if the node was added successfully, False o.w.

        Note: if the node id already exists the node will not be added
        """
        if node_id in self.nodes:
            return False
        else:
            self.nodes[node_id] = NodeData(node_id, pos)
            self.mc += 1
            return True

    def remove_node(self, node_id: int) -> bool:
        """
        Removes a node from the graph.
        @param node_id: The node ID
        @return: True if the node was removed successfully, False o.w.

        Note: if the node id does not exists the function will do nothing
        """
        if node_id not in self.nodes:
            return False
        for key in list(self.edges.keys()):
            # delete all the edges the node is connected to
            if node_id in key:
                del self.edges[key]

        self.mc += 1
        del self.nodes[node_id]  # delete node
        return True

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        """
        Removes an edge from the graph.
        @param node_id1: The start node of the edge
        @param node_id2: The end node of the edge
        @return: True if the edge was removed successfully, False o.w.

        Note: If such an edge does not exists the function will do nothing
        """
        if (node_id1, node_id2) in self.edges:
            self.mc += 1
            del self.edges[(node_id1, node_id2)]
            return True
        return False  # if the edge doesn't exist
