# -*- coding: utf-8 -*-
"""
SINGLE PRINGLES OR PEAS IN A POD?

Created on Sat Jul 15 21:28:30 2023

@author: seb-wilkes
"""
import numpy as np
from numpy.random import random # uniform random variable [0,1)

# Agent constants
SENSITIVITY_CONST = 0.42
NORM_CONST = np.exp(- (9 * SENSITIVITY_CONST))
RELATIONSHIP_SURVIVABILITY_CONST = np.sqrt(0.95)

# Run time parameters 
CLUSTER_SIZE_MEAN = 100

class agent:
    '''
    This class creates a myopic agent that seeks to find love 
    with a homogenous preference function
    '''
    def __init__(self, beauty_value, agent_id):
        self.beauty_value = beauty_value
        self.id = agent_id
        self.relationship_status = False
        
    def toggle_relationship_status(self):
        self.relationship_status = ~self.relationship_status
        
    def set_relationship_status(self, value_bool):
        self.relationship_status = value_bool
        
    def attraction_kernel(self, beauty_value_of_person):
        # This is the underlying model for how people interact with one another        
        prob = np.exp( (beauty_value_of_person - self.beauty_value) 
                      * SENSITIVITY_CONST ) 
        return prob * NORM_CONST
    
    def attraction_routine(self, beauty_value_subject):
        # Perform MC sampling
        random_number = random()
        if random_number < self.attraction_kernel(beauty_value_subject):
            return True
        else:
            return False
        
    def relationship_routine(self):
        # MC sample if relationship will survive Dt interval
        if random() > RELATIONSHIP_SURVIVABILITY_CONST:
            return False
        else:
            return True

class population:
    def __init__(self, random_func, size):
        '''
        random_func: is a python function, with no args, that produces 
        a beauty array that is distrubted with an appropriate PDF
        
        size: is the population size
        
        '''
            
        self.population = [agent(random_var(), i) 
                                    for i in range(size)]
        print("Population constructed")
        self.agent_ids = np.array([i for i in range(size)])
        # stores ids of agents not in relationships
        self.singles_register = np.full_like(self.population, 
                                             True).astype(bool)        
        self.relationship_register = [] # stores id tuples of relationships
        # Now create interaction matrix for all agents
        self.interaction_matrix = np.zeros((size,size))
        for i in range(size):
            for j in range(size):
                if i==j:
                    continue
                else:                    
                    self.interaction_matrix[i,j] = \
                        self.population[i].attraction_kernel(
                            self.population[j].beauty_value)
        print("Interaction matrix complete")
        self.couple_status = np.full_like(self.population, False)
        self.time_series_store = [] # where results over time are stored
     
        
     
    def permute_agents(self, indices):
        permuted_indices = np.random.permutation(indices)
        return permuted_indices
    
    def prepare_available_agents(self):
        available_agents = self.agent_ids[self.singles_register]
        return self.permute_agents(available_agents)
    
    def compatibility_stage_first_preference(self, shuffled_indices,
                                             compatibility_matrix):
        # this function means the agent will take the first partner that agrees
        _couple_tracker = np.full(len(shuffled_indices), False) # internal only
        for i in range(len(shuffled_indices)):
            # this loop implicitly gives the advantage to smaller i
            if _couple_tracker[i]:
                continue # already ineligible
            compatibility = compatibility_matrix[i]*compatibility_matrix[:,i]
            matches = np.argwhere(compatibility)
            number_of_matches = matches.size
            if number_of_matches == 0:
                continue # no love this time
            else:                
                # find first available partner
                potential_interests = _couple_tracker[matches]
                if np.all(potential_interests):
                    continue # no luck!
                new_partner_arg = matches[np.argwhere(
                    ~potential_interests)[0,0]][0]
            _couple_tracker[[i,new_partner_arg]] = True   
            new_pair = (shuffled_indices[i], shuffled_indices[new_partner_arg])
            self.relationship_register.append(new_pair)
            self.singles_register[[*new_pair]] = False
            
        return
    
    def check_relationship_outcomes(self):
        pop_indices = []; couples_no = len(self.relationship_register)
        for c_i, couple in enumerate(self.relationship_register[::-1]):
            sentiment_a = self.population[couple[0]].relationship_routine()
            sentiment_b = self.population[couple[1]].relationship_routine()
            if sentiment_a and sentiment_b:
                pass # happy days
            else:
                self.singles_register[[*couple]] = True
                # prepare to strike from register
                pop_indices.append(couples_no - c_i - 1)
        for index in pop_indices:
            self.relationship_register.pop(index) # free to mingle once more      
         
        return
        
    
    def batch_time_interval(self, shuffled_indices):        
        reduced_matrix = lambda M: M[shuffled_indices].T[shuffled_indices].T
        random_matrix = random((len(shuffled_indices),len(shuffled_indices)))
        mingle_stage = (reduced_matrix(self.interaction_matrix) 
                        - random_matrix) > 0
        self.compatibility_stage_first_preference(shuffled_indices, 
                                                  mingle_stage)
        
    def simple_batching_function(self, full_shuffled_indices):
        max_number_of_batches = len(full_shuffled_indices) // CLUSTER_SIZE_MEAN

        return [full_shuffled_indices[i*CLUSTER_SIZE_MEAN: 
                                      ((i+1)*CLUSTER_SIZE_MEAN) if i
                                      < max_number_of_batches else None]
                for i in range(max_number_of_batches + 1)]
    
    def full_time_interval(self):
        ''' This function moves the simulation forward a whole time interval '''
        shuffled_indices_list = self.simple_batching_function(
            self.prepare_available_agents())
        for mini_batch in shuffled_indices_list:
            self.batch_time_interval(mini_batch)
        # finish off interval by testing relationship status
        self.check_relationship_outcomes()
        # record any data
        for a in self.agent_ids:
            self.population[a].set_relationship_status(
                ~self.singles_register[a])
        self.time_series_store.append([len(self.relationship_register),~self.singles_register])
        return
