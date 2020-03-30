import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
import random
from math import sqrt, ceil, floor
from mpi4py import MPI
import pickle

comm = MPI.COMM_WORLD
pid = comm.Get_rank()

class GeneticAlgortithmNN():

    def __init__(self, population_size=2000, num_layers=0, num_nodes=[], policy_range=[-1, 1], fitness_function=None, selRate=0.1, mutRate=0.01, filename="temp.txt"):
        '''
        Constructor for initialising variables
        '''

        assert(num_layers == len(num_nodes))
        assert(len(policy_range) == 2)

        self.pop_size = population_size
        # int(ceil((1+sqrt(1+8*self.pop_size))/2.0))   # Choosing no. of best parents required for generating offsprings (inv. of nCr)
        self.par_size = int(self.pop_size*selRate)
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.fitness_func = fitness_function
        self.filename = filename
        self.population = []
        self.generation = 0
        self.policy_range = policy_range
        self.mutRate = mutRate
        self.all_fitness = []
        self.parents_score = 0.0
        # print(multiprocessing.cpu_count())
        '''
        Random initialisation of policy
        Policy : list of matrices 
        Population : list of policies
        '''
        for i in range(self.pop_size):
            policy = []
            for j in range(num_layers-1):
                # print(num_nodes[j+1])
                temp = np.random.uniform(size=(
                    num_nodes[j+1], num_nodes[j]), low=self.policy_range[0], high=self.policy_range[1])
                
                # temp += temp_policy[j]
                # temp = np.clip(temp,-1,1)
                policy.append(temp)

            self.population.append(policy)
        if pid == 0 :
            print("Class Initialised\n", self.par_size)
        # x = input()

    def calc_fitness(self):
        '''
        Function to calculate fitness values
        '''

        self.generation += 1
        if pid == 0 :
            print("Generation : {}".format(self.generation))
            print("\nCalculating Fitness\n")

        fitness_list = []
        policy_list = []
        for i in range(self.pop_size):
            policy = self.population[i]
            policy_list.append(policy)

        size = comm.Get_size()
        chunk_size = int(len(policy_list)/size)
        buffer = None
        if pid == 0:
            buffer = [[i*chunk_size,(i+1)*chunk_size] for i in range(size-1)]
            buffer.append([(size-1)*chunk_size,len(policy_list)])
            buffer = np.asmatrix(buffer)
        index = np.empty(2, dtype=np.int)
        comm.Scatter([buffer,MPI.INT], [index, MPI.INT], root=0)
        fitness = np.zeros(int(index[1]-index[0]))
        for i in range(index[0],index[1]) :
            fitness[i-index[0]] = self.fitness_func(policy_list[i])
        buffer = np.empty(self.pop_size)
        fitness = np.asarray(fitness)
        comm.Allgather([fitness,MPI.DOUBLE],[buffer, MPI.DOUBLE])
        fitness_list = []
        for i in range(self.pop_size):
            fitness_list.append((buffer[i], policy_list[i]))
            self.all_fitness.append(fitness_list[i])
        return fitness_list


    # def show(self):
    #     return
    #     '''
    #     Function to plot graph of fitness vs iteration
    #     '''
    #     plt.figure()
    #     plt.plot(self.all_fitness)
    #     plt.xlabel("generation")
    #     plt.ylabel("fitness")
    #     plt.show()

    #     time.sleep(1)

    #     fitness = self.fitness_func(self.best_policy, True)
    #     if pid == 0 :
    #         print("Best Fitness : {} \n {} \n".format(fitness, self.best_policy))

    #     time.sleep(1)

    def select_mating_pool(self, fitness_list):
        '''
        Select best parents based on fitness values
        '''
        parents = None
        # print("\nSelect Mating\n")
        if pid == 0 :
            fitness_list = sorted(fitness_list, key=lambda x: x[0], reverse=True)
            parents = []

            for i in range(self.par_size):
                parents.append(fitness_list[i])

            self.best_policy = fitness_list[0][1]
            self.best_fitness = fitness_list[0][0]
            self.parents_score = sum([x[0] for x in parents])
            self.write_to_file()
        
        comm.bcast(parents, root=0)

        # self.show()

        return parents

    def write_to_file(self):
        #filename_ = "{}/{}_{}.npz".format(self.filename,self.generation,self.best_fitness)
         with open(self.filename, "a") as output_file:
             output_file.write("\n Generation {} : \n Policy: {} \n Fitness: {}".format(
                 self.generation, self.best_policy, self.best_fitness))
        #np.savez_compressed(filename_,enumerate(self.best_policy))

    def cross_policy(self, policyA, policyB):
        '''
        Crossing over two policies
        '''

        assert(len(policyA) == len(policyB))
        childA = []
        childB = []

        '''
        Cutting all the matrices of a single policy at rows and cols of same ratio
        '''
        randR_ = np.random.rand()
        randC_ = np.random.rand()

        for i in range(len(policyA)):

            layerA = policyA[i]
            layerB = policyB[i]
            rows = layerA.shape[0]
            cols = layerA.shape[1]

            randR = int(randR_*(rows-1))  # np.random.randint(0, rows-1)
            randC = int(randC_*(cols-1))  # np.random.randint(0, cols-1)

            tempA = np.zeros((rows, cols))
            tempB = np.zeros((rows, cols))

            for j in range(rows):
                for k in range(cols):

                    if j < randR or (j == randR and k < randC):
                        tempA[j][k] = layerA[j][k]
                        tempB[j][k] = layerB[j][k]

                    else:
                        tempA[j][k] = layerB[j][k]
                        tempB[j][k] = layerA[j][k]
            childA.append(tempA)
            childB.append(tempB)

        return childA, childB

    def roulette_selection(self, pick, parents):
        current = 0
        for parent in parents:
            current += parent[0]
            if current > pick:
                return parent[1]

    def crossover(self, parents):

        # print("\nCrossing\n")
        offsprings = None
        # for i in range(0, len(parents)):
        #     for j in range(i, len(parents)):
        # '''
        # Generating list of 2 children by 2 parents
        # '''
        # child1, child2 = self.cross_policy(parents[i], parents[j])
        # offsprings.append(child1)
        # offsprings.append(child2)
        '''
        Selecting parents using roulette selection
        '''
        
        if pid == 0:
            offsprings = []
            for i in range(int(ceil(self.pop_size/2))):
                pick = random.uniform(0, self.parents_score)
                parent1 = self.roulette_selection(pick, parents)
                pick = random.uniform(0, self.parents_score)
                parent2 = self.roulette_selection(pick, parents)

                child1, child2 = self.cross_policy(parent1, parent2)
                offsprings.append(child1)
                offsprings.append(child2)
            
        comm.bcast(offsprings, root=0)
        return offsprings

    def mutation(self, offsprings):
        '''
        Mutating the offsprings
        '''
        # print("\nMutation\n")
        if pid == 0 :
            for i in range(len(offsprings)):
                for j in range(len(offsprings[i])):

                    random_weight = np.random.uniform(size=(
                        offsprings[i][j].shape[0], offsprings[i][j].shape[1]), low=self.policy_range[0], high=self.policy_range[1])
                    random_choice = np.random.choice([0, 1], size=offsprings[i][j].shape, p=[
                        1-self.mutRate, self.mutRate])
                    offsprings[i][j] += np.multiply(random_choice, random_weight)

                    offsprings[i][j] = np.clip(
                        offsprings[i][j], self.policy_range[0], self.policy_range[1], out=offsprings[i][j])
        comm.Bcast(offsprings, root=0)
        return offsprings

    def update_population(self, offsprings):

        # print("\nUpdate population\n")
        self.population = offsprings
