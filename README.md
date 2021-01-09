# Directed Weighted Graph
![weight01](https://user-images.githubusercontent.com/74146562/104093717-f4c32800-5294-11eb-866b-8a922a9afa9d.gif)

## about our project:
In the project we presents an implementation of a directed weighted graph with different methods and algorithms using by Python.
the src contains different classes which represents the directional weighted graph and the methods you can do on it.

## the classes we use for this program:

## NodeData
This class is used for our algorithm in  DiGraph and GraphAlgo classes

## Digraph
This class implements a directed weighted graph.

## Methods

| Function  | Explanation |
| ------------- | ------------- |
|   v_size()   |  Return the amount of nodes in the graph.    |
|    e_size()  |    Return the amount of edges in the graph.  |
|   get_all_v   |   Return all the nodes in the graph.   |
|    all_in_edges_of_node  |   Return all the edges that enter into a one node.   |
|    all_out_edges_of_node | Return all the edges that get out from one node.    |
|   get_mc   | return the amount of changes the graph had  |
|    add_edge  |  Add  edges to the graph, a collection of all nodes.    |
|     add_node |   Add nodes to the graph, a collection edges from node.  |
|     remove_node |  Deletes a specific node from the graph   |
|    remove_edge  | Deletes an edge between two nodes OF the graph     |


## GraphAlgo
The class represents algorithms of a DiGraph class.

| Function  | Explanation |
| ------------- | ------------- |
|   get_graph   | Return the underlying graph of which the algorithm works    |
|    load_from_json  |   loads a graph from a file via Deserialization  |
|   save_to_json   |  saves a graph to a file via Serialization   |
|  shortest_path  | return the amount of changes the graph had  |
|    connected_component  | Finds the Strongly Connected Component  between given nodes, return the list of nodes in the Strongly Connected Component    |
|     connected_components |  returns a list of the shortest path  |
|    plot_graph |  Saves the graph into json format   |

