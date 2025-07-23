import networkx as nx
import copy
import numpy as np
import json

test_mode = False
network_config_path = "data/6-bus/network_config.json"

nT = 24  # number of time steps in a day
dT = 24 / nT  # time interval in hours



class DistributionSystemOperator:  # DSO: Distribution System Operator, responsible for encrypted information exchange among multiple DES units
    """
    DistributionSystemOperator
     Represent the distribution network as a directed graph for DES information exchange:
    (1) Nodes store DES data via the DesContainer class.
    (2) Edges hold encrypted interaction variables z between adjacent nodes.
    """
    def __init__(self, config_path=network_config_path):
        """
        Initialize the distribution network and attach a DES information container (DesContainer) to every node
        :param config_path: Path to network configuration file
        """
        with open(config_path, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)  # Read network configuration
        self.n_parent = self.graph.in_degree()  # Each node's in-degree equals the number of parent nodes
        self.n_child = self.graph.out_degree()  # Each node's out-degree equals the number of child nodes
        self.add_nodes()  # Initialize the DES information container for every node

    def add_nodes(self):
        """
        Attach a DES information container to every node in the graph
        :return: None
        """
        for node in self.graph.nodes:
            self.graph.add_node(node, des=DesContainer(_id=node, n_parent=self.n_parent[node], n_child=self.n_child[node]))  # Bind graph nodes to their corresponding DES information containers

    def update_edge_from_nodes(self, iter):
        """
        # Compute ciphertext of auxiliary variables
        :return: None
        """
        for u, adjacencies in self.graph.adjacency():
            for index, (v, data) in enumerate(adjacencies.items()):
                alpha = self.graph.nodes[u]['des'].alpha[f'child{index}']
                self.graph[u][v]['z_child'] = (self.graph.nodes[u]['des'].results_B['P_child'][index] * alpha +
                                               self.graph.nodes[v]['des'].results_A['P_parent'] * (1 - alpha))
                self.graph[u][v]['z_parent'] = (self.graph.nodes[u]['des'].results_A['P_child'][index] * alpha +
                                                self.graph.nodes[v]['des'].results_B['P_parent'] * (1 - alpha))
        return True

    def update_nodes_from_edges(self):
        """
        Update DES container information with ciphertext of auxiliary variables
        :return: None
        """
        for u, adjacencies in self.graph.adjacency():
            for index, (v, data) in enumerate(adjacencies.items()):
                z_parent = data['z_parent']
                z_child = data['z_child']
                self.graph.nodes[u]['des'].results_A['P_child'][index] = z_parent
                self.graph.nodes[v]['des'].results_A['P_parent'] = z_child
        return True

    def distribute_line_impedance_from_edges(self):
        """
        Distribution line impedance
        :return: None
        """
        for u, adjacencies in self.graph.adjacency():
            for index, (v, data) in enumerate(adjacencies.items()):
                self.graph.nodes[u]['des'].line_impedance[f'child{index}'] = data
                self.graph.nodes[v]['des'].line_impedance['parent'] = data
        return True

    def distribute_public_keys(self):
        """
        Distribute public keys based on topology information
        :return: None
        """
        for u, adjacencies in self.graph.adjacency():
            for index, (v, data) in enumerate(adjacencies.items()):
                self.graph.nodes[u]['des'].pub_keys_B[f'child{index}'] = self.graph.nodes[v]['des'].pub_keys_A['parent']
                self.graph.nodes[v]['des'].pub_keys_B['parent'] = self.graph.nodes[u]['des'].pub_keys_A[f'child{index}']
        return True

    def distribute_alpha_from_nodes(self):
        """
        Distribute alpha values to each node
        :return: None
        """
        for u, adjacencies in self.graph.adjacency():
            for index, (v, data) in enumerate(adjacencies.items()):
                h = self.graph.nodes[v]['height']
                alpha = 0.2 + (h / 22) * (0.5 - 0.2)
                alpha = 0.5
                if u == 0:
                    alpha = 0.5
                self.graph.nodes[u]['des'].alpha[f'child{index}'] = alpha
                self.graph.nodes[v]['des'].alpha['parent'] = alpha
        return True


# DES: Distributed Energy System information container, used by the power grid to collect data from each DES
class DesContainer:
    """
    DesContainer: Used to collect DES information (node data)
    """
    def __init__(self, _id: int = 0, n_parent: int = 1, n_child: int = 0):
        """
        Initialize DES information container
        :param _id: DES ID
        :param n_parent: Number of parent nodes
        :param n_child: Number of child nodes
        """
        self.id = _id
        self.n_parent = n_parent
        self.n_child = n_child
        self.voltage = np.zeros(nT)  # node voltage
        self.results_A = dict()  # Store ciphertext of optimization calculation results for each node
        self.results_B = dict()  # Store ciphertext of optimization calculation results for each node
        self.pub_keys_A = dict()  # Store the self generated public keys of each node
        self.pub_keys_B = dict()  # Store the public keys of neighbors of each node
        self.line_impedance = dict()
        self.alpha = dict()

    def update_x_from_des_A(self, results: dict):
        """
        Update the DES information container of the power grid based on the local optimization calculation results of the DES system
        :param results: The ciphertext of the optimization calculation results for each node
        :return:
        """
        self.results_A = copy.deepcopy(results)
        # self.parent_voltage['voltage'] = results['voltage']
        return True

    def update_x_from_des_B(self, results: dict):
        """
        Update the DES information container of the power grid based on the local optimization calculation results of the DES system
        :param results: The ciphertext of the optimization calculation results for each node
        :return:
        """
        self.results_B = copy.deepcopy(results)
        # self.parent_voltage=[0]
        return True

    def add_keys_A(self, keys_A: dict):
        """
        Add Key
        :param keys: Key, public key
        :return:
        """
        self.pub_keys_A = copy.deepcopy(keys_A)
        self.pub_keys_B = keys_A

        return True



if __name__ == '__main__':
    test_mode = True
    dso = DistributionSystemOperator()
