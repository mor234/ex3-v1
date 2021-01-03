class NodeData:
    def __init__(self, node_id: int, pos: tuple = None):
        self.id = node_id
        self.pos = pos
        self.visited = False  # used in algorithm
        self.is_part_of_scc = False;  # used in algorithm
        self.tag = float('inf')  # used in algorithm , the number of last id
