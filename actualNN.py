import numpy as np
from activation_function import sigmoid
import graph
import matplotlib.pyplot as plt
import pygad
import EA
# Define the neural network class
class ActualNeuralNetwork:
    def __init__(self, neural_graph, input_nodes, output_nodes) -> None:
        self.neural_graph = neural_graph
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.previous_activations = np.zeros(self.neural_graph.n)
        self.current_activations = np.zeros(self.neural_graph.n)
        self.weights = self.initialize_weights()
        self.max_decision_time = 100


    def initialize_weights(self):
        return np.random.rand(self.neural_graph.n, self.neural_graph.n) * self.neural_graph.adjacency_matrix
    
    def get_unfinished_threads(self):
        #return the lines of the current activations matrix where no output node has reached an activation of 1
        return np.where(np.sum(np.abs(self.current_activations[:, self.output_nodes]) >= 1, axis=1) == 0)[0]
        
    
    def forward_propagation(self, input_values):
        #input_values may be a matrix of shape (n, m) where n is the number of input nodes and m is the number of input values
        self.current_activations = np.zeros((input_values.shape[0],self.neural_graph.n))
        self.current_activations[:, self.input_nodes] = input_values
        self.output_activations = np.zeros((input_values.shape[0], len(self.output_nodes)))
        unfinished_threads = np.arange(input_values.shape[0])
        i = 0
        while i < self.max_decision_time:
            self.current_activations[unfinished_threads] = self.current_activations[unfinished_threads] + np.dot(self.current_activations[unfinished_threads],self.weights)
            unfinished_threads = self.get_unfinished_threads()
            if len(unfinished_threads) == 0:
                break
            i += 1
        output = np.square(self.current_activations[:, self.output_nodes])
        output = output/np.sum(output, axis=1)[:, np.newaxis]
        #softmax
        #softmax_output = np.exp(output) / np.sum(np.exp(output), axis=0)
        return output
    
            
    def display(self):
        # Display the graph with the weights as edge labels and the activations as node labels
        self.neural_graph.display(self.weights, self.current_activations, node_colors=['red' if i in self.input_nodes else 'blue' if i in self.output_nodes else 'green' for i in range(self.neural_graph.n)])
    
    
    def train(self, input_values, output_values, loss_function):
        num_gens = 0
        def fitness_function(solution):
            nonlocal num_gens
            num_gens += 1
            #set only the weights that are not null in the adjacency matrix
            self.weights = np.zeros((self.neural_graph.n, self.neural_graph.n))
            self.weights[self.neural_graph.adjacency_matrix != 0] = solution
            #compute the loss
            f = self.forward_propagation(input_values)
            loss = loss_function(f, output_values)
            print(f"\r num_gens: {num_gens} fitness: {1/loss}", f"loss: {loss}", end="")
            return 1/loss
        #initialize the genetic algorithm
        chromosome_size = np.count_nonzero(self.neural_graph.adjacency_matrix)
        ea_result = EA.simple_opoea(chromosome_size, 0.00005, 100000, fitness_function)
        #set the weights to the best solution
        self.weights = np.zeros((self.neural_graph.n, self.neural_graph.n))
        self.weights[self.neural_graph.adjacency_matrix != 0] = ea_result.best_solution
        #return the best solution
        return ea_result
    