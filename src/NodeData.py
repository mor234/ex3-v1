class NodeDate:
    def __init__(self, node_id: int, pos: tuple = None):
        self.visited =False
        self.tag =float('inf')
        self.id = node_id
        self.pos = pos
