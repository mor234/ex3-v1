class NodeData:
    def __init__(self, node_id: int, pos: tuple = None):
        """
        contractor
        :param node_id: the id of the node
        :param pos: location of the node
        """
        self.id = node_id
        self.pos = pos
        self.out_edge_nodes = {}
        self.in_edge_nodes = {}

        self.visited = False  # used in algorithm
        self.is_part_of_scc = False;  # used in algorithm
        self.tag = float('inf')  # used in algorithm , the number of last id



    def __repr__(self):
        """

        :return: string representing the node, allows to print the node
        """
        ans = ""
        ans +=str(self.id)
        ans+= ": " + "|edges out| " + str(
            len(self.out_edge_nodes))
        ans += " |edges in| " + str(len(self.in_edge_nodes))
        return ans



