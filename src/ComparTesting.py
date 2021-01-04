def plot_graph(self) -> None:
    """
    Plots the graph.
    If the nodes have a position, the nodes will be placed there.
    Otherwise, they will be placed in a random but elegant manner.
    @return: None
    """

    graph = nx.DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_node("D")
    graph.add_node("E")
    graph.add_node("F")
    graph.add_node("G")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "E")
    graph.add_edge("C", "F")
    graph.add_edge("D", "E")
    graph.add_edge("F", "G")

    print(graph.nodes())
    print(graph.edges())

    pos = nx.spring_layout(graph)

    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    nx.draw_networkx_edges(graph, pos, edge_color='r', arrows=True)

    plt.show()
