import unittest

from src.DiGraph import DiGraph


class TestDiGraph(unittest.TestCase):

    def test_v_size(self):
        """
        check the function that returns the number of vertices in this graph
        """
        g = DiGraph()
        g.add_node(0);
        g.add_node(1);
        g.add_node(2);
        self.assertEqual(g.v_size(), 3, " node size is not correct")
        g.add_node(2);
        self.assertEqual(g.v_size(), 3, " node size is not correct after adding an existing node")
        g.add_edge(1, 2, 0.5)
        self.assertEqual(g.v_size(), 3, " node size is not suppose to change after adding edge")
        g.remove_node(0)
        self.assertEqual(g.v_size(), 2, " node size is did not change after removing node")

    def test_e_size(self):
        """
        check the function that returns the number of edges in this graph
        """
        g = DiGraph()
        g.add_node(0);
        g.add_node(1);
        g.add_node(2);
        g.add_edge(1, 2, 0.5)  # 1
        g.add_edge(1, 0, 0.5)  # 2
        g.add_edge(0, 1, 0.5)  # 3
        g.add_edge(0, 2, 0.5)  # 4
        g.add_edge(1, 0, 3)  # 4 ,not suppose to change

        self.assertEqual(g.e_size(), 4, " edge size is not correct")
        g.remove_edge(1, 0)
        self.assertEqual(g.e_size(), 3, " edge size is not correct after removing an edge")
        g.remove_node(1)
        self.assertEqual(g.e_size(), 1,
                         " edge size is not correct after removing a node. suppose to remove connected edges")

    def test_get_all_v(self):
        """
        check function that returns all vertx of the graph
        """

        g = DiGraph()
        g.add_node(0, (0, 7, 0));
        g.add_node(1, (1, 7, 0));
        g.add_node(2, (2, 7, 0));

        nodes = g.get_all_v();
        self.assertEqual(len(nodes), 3, "didn't get the right collection");

        index_counter = [0, 0, 0]

        for node in nodes.values():
            index_counter[node.id] += 1
            self.assertEqual((node.id, 7, 0), node.pos, "not got the node correctly");

        self.assertEqual(index_counter[0], 1, "not found node 1");
        self.assertEqual(index_counter[1], 1, "not found node 2");
        self.assertEqual(index_counter[2], 1, "not found node 3");

    def test_all_in_out_edges_of_node(self):
        """check function that returns a dictionary of all the nodes connected to (into) node_id ,
        each node is represented using a pair (other_node_id, weight)
        """

        g = DiGraph()

        g.add_node(1, (1, 7, 0));
        g.add_node(2, (2, 7, 0));
        g.add_node(3, (0, 7, 0));

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

        g = DiGraph()
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)
        g.add_node(4)
        g.add_edge(1, 4, 14)
        g.add_edge(1, 2, 12)
        g.add_edge(2, 1, 21)
        g.add_edge(1, 3, 13)
        g.add_edge(2, 3, 23)
        g.add_edge(3, 1, 31)
        self.assertEqual(g.all_out_edges_of_node(1), {4: 14, 2: 12, 3: 13})
        self.assertEqual(g.all_in_edges_of_node(1), {2: 21, 3: 31})

    def test_get_mc(self):
        """
        check mc change and not change correctly
        """
        g = DiGraph()
        mc_start = g.get_mc()
        g.add_node(1)
        g.add_node(2)
        g.add_node(3)
        mc_after_add_node = g.get_mc()
        self.assertTrue(mc_start + 3 <= mc_after_add_node, "not update mc correctly after add")
        g.add_node(3)
        self.assertEqual(g.get_mc(), mc_after_add_node, "mc not suppose to change when adding already existing node")
        g.add_edge(1, 2, 1.5)
        mc_after_add_edge = g.get_mc()
        self.assertTrue(mc_after_add_node < mc_after_add_edge, "not update mc correctly after add")
        g.add_edge(1, 2, 7)
        self.assertEqual(g.get_mc(), mc_after_add_edge, "mc not suppose to change when adding already existing edge")
        self.assertEqual(g.edges[(1, 2)], 1.5, "edge not suppose to change when adding already existing edge")
        g.remove_node(3)
        mc_rm_node = g.get_mc()
        self.assertTrue(mc_after_add_edge < mc_rm_node, "not update mc correctly after rmove edge")
        g.remove_edge(1, 2)
        mc_rm_edge = g.get_mc()
        self.assertTrue(mc_rm_node < mc_rm_edge, "not update mc correctly after rmove edge")
        g.all_out_edges_of_node(1)
        g.v_size()
        g.get_all_v()
        self.assertEqual(mc_rm_edge, g.get_mc(), "mc not suppose to change after get functions")

    def test_add_edge(self):
        g = DiGraph()

        g.add_node(0);
        g.add_node(1);
        g.add_node(2);

        self.assertTrue(g.add_edge(0, 1, 0.76), "1")
        self.assertFalse((1, 0) in g.edges, "opposite direction edge not suppose to be created")
        self.assertEqual(g.edges[0, 1], 0.76, "didn't create the edge")
        self.assertFalse(g.add_edge(0, 1, 8), "2")  # return: True if the edge was added successfully, False o.w.
        self.assertEqual(g.edges[0, 1], 0.76, "the edge didn't remain the same")

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
        self.assertFalse(g.add_node(1, (3, 7, 0)), "didn't return false when added already exist node")
        self.assertEqual(g.nodes[1].pos, (2, 5), "added already exist node")

    def test_remove_node(self):
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
        self.assertFalse((2, 1) in g.edges, "didn't delete edge in the opposite direction.");
        self.assertTrue(mc < g.get_mc(), "didn't update mc correctly");

    def test_remove_edge(self):
        g = DiGraph()

        g.add_node(0);
        g.add_node(1);
        g.add_node(2);

        g.add_edge(1, 2, 1.5);
        g.add_edge(2, 1, 3);
        g.add_edge(1, 3, 6);

        mc = g.get_mc()
        self.assertTrue(g.remove_edge(1, 2))
        self.assertFalse((1, 2) in g.edges, "didn't delete edge ");
        self.assertFalse((0, 1) in g.edges, "didn't return false for not existing edge.");
        self.assertTrue(mc + 1 <= g.get_mc(), "didn't update mc correctly");


if __name__ == '__main__':
    unittest.main()
