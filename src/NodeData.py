class NodeData:
    def __init__(self, node_id: int, pos: tuple = None):
        self.id = node_id
        self.pos = pos
        self.visited = False  # used in algorithm
        self.is_part_of_scc = False;  # used in algorithm
        self.tag = float('inf')  # used in algorithm , the number of last id

    def __lt__(self, other):
        """
        compare between 2 nodes according to their tag.
        used in shortest path function in GraphAlgo.
        :param other:
        :return:
        """
        return self.tag< other.tag
