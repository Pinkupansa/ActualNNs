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
        self.activations_history = []
        self.unfinished_threads_history = []

    def initialize_weights(self):
        return np.random.rand(self.neural_graph.n, self.neural_graph.n) * self.neural_graph.adjacency_matrix
    
    def get_unfinished_threads(self):
        #return the lines of the current activations matrix where no output node has reached an activation of 1
        return np.where(np.sum(np.abs(self.current_activations[:, self.output_nodes]) >= 1, axis=1) == 0)[0]
        
    
    def forward_propagation(self, input_values):
        #input_values may be a matrix of shape (n, m) where n is the number of input nodes and m is the number of input values
        self.activations_history = []
        self.unfinished_threads_history = []
        self.current_activations = np.zeros((input_values.shape[0],self.neural_graph.n))
        self.current_activations[:, self.input_nodes] = input_values
        self.activations_history.append(np.copy(self.current_activations))
        self.output_activations = np.zeros((input_values.shape[0], len(self.output_nodes)))
        unfinished_threads = np.arange(input_values.shape[0])
        
        self.unfinished_threads_history.append(unfinished_threads)
        i = 0
        while i < self.max_decision_time:
            self.current_activations[unfinished_threads] = self.current_activations[unfinished_threads] + np.dot(self.current_activations[unfinished_threads],self.weights)
            self.activations_history.append(np.copy(self.current_activations))
            unfinished_threads = self.get_unfinished_threads()
            self.unfinished_threads_history.append(unfinished_threads)
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
    
    #assuming mse
    def train_gradient_descent(self, input_values, output_values, learning_rate, num_iterations, loss_function):
        for i in range(num_iterations):
            f = self.forward_propagation(input_values)
            loss = loss_function(f, output_values)
            
            gradient = self.compute_mse_gradient(output_values)
            print(f"\r num_iterations: {i} fitness: {1/loss}", f"loss: {loss}", f"gradient: {np.linalg.norm(gradient)}", end="")
            #print(self.weights.shape)
            #print(gradient.shape)
            self.weights = self.weights - learning_rate * gradient
        return self.weights

    
    def compute_activation_gradients(self):
        d_activations = np.zeros((self.current_activations.shape[0], self.neural_graph.n, self.neural_graph.n, self.neural_graph.n))
        weightsplusidentity = self.weights + np.identity(self.neural_graph.n)
        ei_vectors = [np.zeros((self.neural_graph.n, 1)) for i in range(self.neural_graph.n)]
        for i in range(self.neural_graph.n):
            ei_vectors[i][i, :] = 1
        #todo : vectorize this
        for t in range(0, len(self.unfinished_threads_history)):
            for i in range(self.neural_graph.n):
                for j in range(self.neural_graph.n):
                    d_activations[self.unfinished_threads_history[t], i, j, :] = (weightsplusidentity @ d_activations[self.unfinished_threads_history[t], i, j, :].T + self.activations_history[t-1][self.unfinished_threads_history[t], j] * ei_vectors[i]).T
        return d_activations

    #assuming mse
    def compute_mse_gradient(self, true):
        d_activations = self.compute_activation_gradients()

        d_activations = d_activations[:, :, :, self.output_nodes]

        act = self.current_activations[:, self.output_nodes]
        act_2 = np.square(act)
        act_2_sum = np.sum(act_2, axis=1)
        weirdterm = np.sum(act * (true - act_2), axis=1)/act_2_sum
        #duplicate weirdterm as much as there are output nodes
        weirdterm = np.repeat(weirdterm[:, np.newaxis], len(self.output_nodes), axis=1)
        right_factor = act * (act + weirdterm - true)
        grad = d_activations @ right_factor[:, :].T

        return np.sum(grad, axis=0)[:,:,0]/len(true)