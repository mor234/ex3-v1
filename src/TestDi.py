import unittest

from DiGraph import DiGraph


class TestDiGraph(unittest.TestCase):

    def test_init_(self):
        pass

    def test_v_size(self):
        """
        Returns the number of vertices in this graph
        @return: The number of vertices in this graph
        """

    def test_e_size(self):
        """
        Returns the number of edges in this graph
        @return: The number of edges in this graph
        """

    def test_get_all_v(self):

        g = DiGraph()
        g.add_node(0, (0, 7));
        g.add_node(1, (1, 7));
        g.add_node(2, (2, 7));

        nodes = g.get_all_v();
        print(len(nodes))
        self.assertEqual(len(nodes), 3, "didn't get the right collection");

        index_counter = [0, 0, 0]

        for node in nodes.values():
            index_counter[node.id] += 1
            self.assertEqual((node.id, 7), node.pos, "not got the node correctly");

        self.assertEqual(index_counter[0], 1, "not found node 1");
        self.assertEqual(index_counter[1], 1, "not found node 2");
        self.assertEqual(index_counter[2], 1, "not found node 3");

    def test_all_in_out_edges_of_node(self):
        """return a dictionary of all the nodes connected to (into) node_id ,
        each node is represented using a pair (other_node_id, weight)
        """

        g = DiGraph()

        g.add_node(1, (1, 7));
        g.add_node(2, (2, 7));
        g.add_node(3, (0, 7));

        g.add_edge(1, 2, 1.5)
        g.add_edge(2, 1, 3);
        g.add_edge(1, 3, 6);

        edges_out_1 = g.all_out_edges_of_node(1)
        edges_in_1 = g.all_in_edges_of_node(1)
        edges_in_2 = g.all_in_edges_of_node(2)
        edges_out_3 = g.all_out_edges_of_node(3)

        self.assertEqual(len(edges_out_1), 2, "didn't get the right dictionary")
        self.assertEqual(edges_out_1[2], 1.5, "not found edge 1_2")

        self.assertEqual(len(edges_in_1), 1, "didn't get the right dictionary")
        self.assertEqual(edges_in_1[2], 3, "not found edge 2_1")

        self.assertEqual(len(edges_in_2), 1, "didn't get the right dictionary")
        self.assertEqual(edges_in_2[1], 1.5, "not found edge 1_2")

        self.assertEqual(len(edges_out_3), 0, "didn't get the right dictionary")

    def test_all_out_edges_of_node(self):
        """return a dictionary of all the nodes connected from node_id , each node is represented using a pair
        (other_node_id, weight)
        """

    def test_get_mc(self):
        """
        Returns the current version of this graph,
        on every change in the graph state - the MC should be increased
        @return: The current version of this graph.
        """

    def test_add_edge(self):
        g = DiGraph()

        g.add_node(0);
        g.add_node(1);
        g.add_node(2);

        self.assertTrue(g.add_edge(0, 1, 0.76), "1")
        self.assertFalse((1, 0) in g.edges, "opposite direction edge not suppose to be created")
        self.assertEquals(g.edges[0, 1], 0.76, "didn't create the edge")
        self.assertFalse(g.add_edge(0, 1, 8), "2")  # return: True if the edge was added successfully, False o.w.
        self.assertEquals(g.edges[0, 1], 0.76, "the edge didn't remain the same")

    def test_add_node(self):
        g = DiGraph()
        # add 4 new nodes
        self.assertTrue(g.add_node(0))
        self.assertTrue(g.add_node(1, (2, 5)))
        self.assertTrue(g.add_node(2))
        self.assertTrue(g.add_node(3))
        self.assertFalse(5 in g.nodes)
        self.assertTrue(1 in g.nodes)
        self.assertEqual(g.nodes[2].id, 2, "didn't get the right node")
        # not added already exist node
        self.assertFalse(g.add_node(1, (3, 7)), "didn't return false when added already exist node")
        self.assertEqual(g.nodes[1].pos, (2, 5), "added already exist node")

    def test_remove_node(self):
        """
        Removes a node from the graph.
        @param node_id: The node ID
        @return: True if the node was removed successfully, False o.w.

        Note: if the node id does not exists the function will do nothing
        """
        g = DiGraph()

        g.add_node(0);
        g.add_node(1);
        g.add_node(2);

        g.add_edge(1, 2, 1.5);
        g.add_edge(2, 1, 3);
        g.add_edge(1, 3, 6);

        mc = g.get_mc()
        self.assertTrue(g.remove_node(1))
        self.assertFalse(1 in g.nodes, "didn't delete node 1")
        self.assertFalse((1, 2) in g.edges, "didn't delete edge ");
        self.assertFalse((2, 1) in g.edges, "didn't delete edge in the oposite direction.");
        self.assertTrue(mc + 4 <= g.get_mc(), "didn't update mc correctly");

    def test_remove_edge(self):
        """
        Removes an edge from the graph.
        @param node_id1: The start node of the edge
        @param node_id2: The end node of the edge
        @return: True if the edge was removed successfully, False o.w.

        Note: If such an edge does not exists the function will do nothing
        """

    if __name__ == '__main__':
        unittest.main()
