# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

from src.causal_graphs.pag import PAG


class GACTest(unittest.TestCase):

    def test_example_three(self):
        # Example 3 from Perkovic et al.
        g = GeneralGraph([GraphNode("X"), GraphNode("Y"), GraphNode("V1"), GraphNode("V2")])
        g.add_edge(Edge(g.get_node("V1"), g.get_node("X"), Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(g.get_node("V2"), g.get_node("X"), Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(g.get_node("V2"), g.get_node("Y"), Endpoint.CIRCLE, Endpoint.CIRCLE))
        g.add_edge(Edge(g.get_node("X"), g.get_node("Y"), Endpoint.CIRCLE, Endpoint.CIRCLE))
        pag = PAG(g)
        self.assertFalse(pag.is_amenable("X", "Y"))

    def test_example_four(self):
        # Example 4 from Perkovic et al.
        g = GeneralGraph([GraphNode("X"), GraphNode("Y"), GraphNode("V1"), GraphNode("V2"), GraphNode("V3"),
                          GraphNode("V4")])
        g.add_edge(Edge(g.get_node("V1"), g.get_node("X"), Endpoint.CIRCLE, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V2"), g.get_node("X"), Endpoint.CIRCLE, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("X"), g.get_node("Y"), Endpoint.TAIL, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("X"), g.get_node("V4"), Endpoint.TAIL, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V3"), g.get_node("X"), Endpoint.CIRCLE, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V3"), g.get_node("V4"), Endpoint.TAIL, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V3"), g.get_node("Y"), Endpoint.TAIL, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V4"), g.get_node("Y"), Endpoint.CIRCLE, Endpoint.ARROW))
        pag_one = PAG(g)
        self.assertTrue(pag_one.is_amenable("X", "Y"))

        g = GeneralGraph([GraphNode("X"), GraphNode("Y"), GraphNode("V1"), GraphNode("V2"), GraphNode("V3"),
                          GraphNode("V4")])
        g.add_edge(Edge(g.get_node("V1"), g.get_node("X"), Endpoint.CIRCLE, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V2"), g.get_node("X"), Endpoint.CIRCLE, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("X"), g.get_node("V4"), Endpoint.TAIL, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V3"), g.get_node("X"), Endpoint.ARROW, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V3"), g.get_node("V4"), Endpoint.ARROW, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V3"), g.get_node("Y"), Endpoint.TAIL, Endpoint.ARROW))
        g.add_edge(Edge(g.get_node("V4"), g.get_node("Y"), Endpoint.TAIL, Endpoint.ARROW))
        pag_two = PAG(g)
        self.assertTrue(pag_two.is_amenable("X", "Y"))

        self.assertTrue(pag_one.adjustment_valid({"V3"}, "X", "Y"))
        self.assertTrue(pag_one.adjustment_valid({"V3", "V1"}, "X", "Y"))
        self.assertTrue(pag_one.adjustment_valid({"V3", "V2"}, "X", "Y"))
        self.assertTrue(pag_one.adjustment_valid({"V3", "V2", "V1"}, "X", "Y"))
        self.assertTrue(pag_one.adjustment_valid(pag_one.get_adjustment_set("X", "Y"), "X", "Y"))

        self.assertFalse(pag_one.adjustment_valid({"V4"}, "X", "Y"))
        self.assertFalse(pag_one.adjustment_valid({"V1"}, "X", "Y"))

        self.assertFalse(pag_two.adjustment_valid({"V3"}, "X", "Y"))
        self.assertFalse(pag_two.adjustment_valid({"V3", "V1"}, "X", "Y"))
        self.assertFalse(pag_two.adjustment_valid({"V3", "V2"}, "X", "Y"))
        self.assertFalse(pag_two.adjustment_valid({"V3", "V2", "V1"}, "X", "Y"))
        self.assertFalse(pag_two.adjustment_valid({"V4"}, "X", "Y"))
        self.assertFalse(pag_two.adjustment_valid({"V1"}, "X", "Y"))

        self.assertFalse(pag_two.adjustment_valid(pag_two.get_adjustment_set("X", "Y"), "X", "Y"))


if __name__ == '__main__':
    unittest.main()
