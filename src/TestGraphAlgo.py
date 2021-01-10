import unittest
from DiGraph import DiGraph
from GraphAlgo import GraphAlgo


class TestGraphAlgo(unittest.TestCase):

    def test_plot_graph(self):
        ga = GraphAlgo()
        g = DiGraph()
        g.add_node(1, (1, 7));
        g.add_node(2, (2, 7));
        g.add_node(3, (0, 7));

        g.add_edge(1, 2, 1.5)
        g.add_edge(2, 1, 3);
        g.add_edge(1, 3, 6);
        ga.graph = g

        ga.plot_graph()

        g.remove_edge(1, 3)
        g.add_edge(3, 1, 6)
        ga.plot_graph()

        ga.load_from_json("../data/A5")
        ga.plot_graph()

    def test_load_save_json(self):
        ga1 = GraphAlgo()
        # load return false correctly
        self.assertFalse(ga1.load_from_json("../data/A7"), "load from not existing file")
        self.assertFalse(ga1.load_from_json("../data/ss"), "loaded from file with no right json")

        # load graph
        self.assertTrue(ga1.load_from_json("../data/A0"))
        self.assertTrue(ga1.save_to_json("../data/try1"))
        ga2 = GraphAlgo()
        self.assertTrue(ga2.load_from_json("../data/try1"))
        self.assertTrue(ga2.get_graph().v_size()>0)


    def test_shortest_path(self):
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

        g = DiGraph()
        g.add_node(0)
        g.add_node(1)
        g.add_node(2)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 2, 4)

        g_algo = GraphAlgo(g)
        self.assertEqual( g_algo.shortest_path(0, 1),(1, [0, 1]))
        self.assertEqual( g_algo.shortest_path(0, 2),(5, [0, 1, 2]))

        path_check_graph = DiGraph()
        for i in range(7):
            path_check_graph.add_node(i)

        path_check_graph.add_edge(1, 2, 3)
        path_check_graph.add_edge(1, 4, 4)

        path_check_graph.add_edge(2, 4, 6)
        path_check_graph.add_edge(2, 5, 7)
        path_check_graph.add_edge(2, 3, 2)

        path_check_graph.add_edge(3, 5, 1)
        path_check_graph.add_edge(3, 6, 8)

        path_check_graph.add_edge(4, 5, 5)

        path_check_graph.add_edge(5, 6, 4)
        g_algo.graph=path_check_graph

        len,path= g_algo.shortest_path(1, 6);


        self.assertEqual(path,[1,2,3,5,6], "didn't give the correct path")
        self.assertEqual(len, 10)


    def test_connected_component_s(self):
        """
        Finds the Strongly Connected Component(SCC) that node id1 is a part of.
        @param id1: The node id
        @return: The list of nodes in the SCC

        Notes:
        If the graph is None or id1 is not in the graph, the function should return an empty list []
        """
        ga = GraphAlgo()
        g = DiGraph()
        g.add_node(1, (1, 7));
        g.add_node(2, (2, 7));
        g.add_node(3, (0, 7));

        g.add_edge(1, 2, 1.5)
        g.add_edge(2, 1, 3);
        g.add_edge(1, 3, 6);
        ga.graph = g
        self.assertEqual(ga.connected_component(1),[1,2])

        g = DiGraph()
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)
        g.add_node(4)
        g.add_node(5)
        g.add_node(6)
        g.add_edge(1, 2, 1.5)
        g.add_edge(2, 6, 3)
        g.add_edge(1, 3, 6)
        g.add_edge(3, 2, 1.5)
        g.add_edge(5, 4, 6)
        g.add_edge(4, 6, 1.5)
        g.add_edge(6, 5, 3)

        ga.graph = g

        self.assertEqual(ga.connected_component(1),[1])
        self.assertEqual(ga.connected_component(6),[4,5,6])
        self.assertEqual(ga.connected_components(),[[1],[2],[3],[4,5,6]])




if __name__ == '__main__':
    unittest.main()
