from src.GraphInterface import GraphInterface
from src.NodeData import NodeData


class DiGraph(GraphInterface):
    def __init__(self):
        """
        constructor
        """
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
        return self.nodes[id1].in_edge_nodes

    def all_out_edges_of_node(self, id1: int) -> dict:
        """return a dictionary of all the nodes connected from node_id , each node is represented using a pair
        (other_node_id, weight)
        assumes id1 is in the graph
        """
        return self.nodes[id1].out_edge_nodes

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
                self.mc += 1  # update mc
                self.edges[(id1, id2)] = weight  # add to list of edges in the graph
                self.nodes[id1].out_edge_nodes[id2] = weight  # add the edge to dict in src node
                self.nodes[id2].in_edge_nodes[id1] = weight  # add the edge to dict in dest node

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
            self.mc += 1  # update mc
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
            del self.edges[(node_id1, node_id2)]  # delete from list of edges in the graph
            del self.nodes[node_id1].out_edge_nodes[node_id2]  # delete the edge from dict in src node
            del self.nodes[node_id2].in_edge_nodes[node_id1]  # delete the edge from dict in dest node

            return True
        return False  # if the edge doesn't exist

    def __repr__(self):
        ans = "Graph: | V |= " + str(self.v_size()) + ", | E |= " + str(self.e_size())
        return ans


