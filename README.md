# Directed Weighted Graph in python
![weight01](https://user-images.githubusercontent.com/74146562/104093717-f4c32800-5294-11eb-866b-8a922a9afa9d.gif)

## about our project:
this project created for an assignment for object orientep programming course in Ariel University
In the project we presents an implementation of a directed weighted graph with different methods and algorithms using by Python.
the src contains different classes which represents the directional weighted graph and the methods you can do on it.  There are three classes  in our project : the DiGraph and GraphAlgo and NodeData . 

## how to use:
write in the terminal:

>git clone https://github.com/mor234/ex3-v1.git   

install any library you are missing, and run any test you want. 

## the classes we use for this program:

## NodeData
This class is used for our algorithm in  DiGraph and GraphAlgo classes its represents a single node in graph, for every node have unique key ,such as : the neighbors, the node that direct to him, tag, info, and dictionay of all his neighbors.

## Digraph
This class implements a directed weighted graph.  here there is a lot of method such as  operate sum operations on the graph, removeNode,  AddNode ,  conncet between two nodes and put weight for the edge.


## His methods

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
The class represents algorithms of a DiGraph class. In this class we can find the shortestpath from src to other destination, found all the strongly component of the graph, find the Strongly Connected Component of a unique key, export Json file of the graph that contain all the edges and the nodes of the graph and read from Json file and load the class with him.

## His methods

| Function  | Explanation |
| ------------- | ------------- |
|   get_graph   | Return the underlying graph of which the algorithm works    |
|    load_from_json  |   loads a graph from a file via Deserialization  |
|   save_to_json   |  saves a graph to a file via Serialization   |
|  shortest_path  | return the amount of changes the graph had  |
|    connected_component  | Finds the Strongly Connected Component  between given nodes, return the list of nodes in the Strongly Connected Component    |
|     connected_components |  returns a list of the shortest path  |
|    plot_graph |  Saves the graph into json format   |

## The compare of our graphs:
in this comparison we take a several graphs and test which algorithm has better running time for same functions. we use in Java alghoritms from ex2 program and from our python algorithms and  NetworkX algorithms.

## the function that we are comparing :

### connected component: 
we only compare this function only between Java algotihms and Python algotihms because NetworkX doesn't use it.
this method set the Strongly Connected Component  of a specific node.
### connected components:
we are going to compare this function only between Java algotihms and Python algotihms because NetworkX doesn't use it. this method return all the connected components of the graph.

### shortest path:
returning the shortest path distance between two nodes in the graph and a list of nodes that represents the shortest path between this 2 nodes. 



