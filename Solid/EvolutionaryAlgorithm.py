from __future__ import division, print_function
from six import add_metaclass

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from random import random, shuffle
from numpy import argmax
from numpy.random import choice

from logging import getLogger
logger = getLogger(__name__)

@add_metaclass(ABCMeta)
class EvolutionaryAlgorithm:
    """
    Conducts evolutionary algorithm
    """

    def __init__(self, crossover_rate, mutation_rate, max_steps, max_fitness=None):
        """

        :param crossover_rate: probability of crossover
        :param mutation_rate: probability of mutation
        :param max_steps: maximum steps to run genetic algorithm for
        :param max_fitness: fitness value to stop algorithm once reached
        """
        if isinstance(crossover_rate, float) and 0 <= crossover_rate <= 1:
            self.crossover_rate = crossover_rate
        else:
            raise ValueError('Crossover rate must be a float between 0 and 1')

        if isinstance(mutation_rate, float) and 0 <= mutation_rate <= 1:
            self.mutation_rate = mutation_rate
        else:
            raise ValueError('Mutation rate must be a float between 0 and 1')

        if isinstance(max_steps, int) and max_steps > 0:
            self.max_steps = max_steps
        else:
            raise ValueError('Maximum steps must be a positive integer')

        if max_fitness is None or isinstance(max_fitness, (int, float)):
            self.max_fitness = max_fitness
        else:
            raise ValueError('Maximum fitness must be a numeric type')

    def status(self):
        return ('{self.__class__.__name__}:\n'
                'step:         {self.cur_steps}\n'
                'best fitness: {self.best_fitness}\n'
                'best member:  {self.best_member}').format(self=self)

    def __repr__(self):
        return ('{self.__class__.__name__}('
                '{self.crossover_rate!r}, '
                '{self.mutation_rate!r}, '
                '{self.max_steps!r}, '
                '{self.max_fitness!r})').format(self=self)

    def _clear(self):
        """
        Resets the variables that are altered on a per-run basis of the algorithm

        :return: None
        """
        self.cur_steps = 0
        self.population = None
        self.fitnesses = None
        self.best_member = None
        self.best_fitness = None

    @abstractmethod
    def _initial_population(self):
        """
        Generates initial population

        :return: list of members of population
        """
        pass

    @abstractmethod
    def _fitness(self, member):
        """
        Evaluates fitness of a given member

        :param member: a member
        :return: fitness of member
        """
        pass

    def _populate_fitness(self):
        """
        Calculates fitness of all members of current population

        :return: None
        """
        self.fitnesses = [self._fitness(x) for x in self.population]

    def _most_fit(self):
        """
        Finds most fit member of current population

        :return: most fit member and most fit member's fitness
        """
        best = argmax(self.fitnesses)
        return self.population[best], self.fitnesses[best]

    def _select_n(self, n):
        """
        Probabilistically selects n members from current population using
        roulette-wheel selection

        :param n: number of members to select
        :return: n members
        """
        total_fitness = sum(self.fitnesses)
        if total_fitness == 0:
            return self.population[:n]
        probs = [x/total_fitness for x in self.fitnesses]
        return list(choice(self.population, size=n, p=probs))

    @abstractmethod
    def _crossover(self, parent1, parent2):
        """
        Creates new member of population by combining two parent members

        :param parent1: a member
        :param parent2: a member
        :return: member made by combining elements of both parents
        """
        pass

    @abstractmethod
    def _mutate(self, member):
        """
        Randomly mutates a member

        :param member: a member
        :return: mutated member
        """
        pass

    def run(self):
        """
        Conducts evolutionary algorithm

        :return: best state and best objective function value
        """
        self._clear()
        self.population = self._initial_population()
        self._populate_fitness()
        self.best_member, self.best_fitness = self._most_fit()
        num_copy = max(int((1 - self.crossover_rate) * len(self.population)), 2)
        num_crossover = len(self.population) - num_copy
        for i in range(self.max_steps):
            self.cur_steps = i

            if (i + 1) % 100 == 0:
                logger.info(self.status())

            self.population = self._select_n(num_copy)
            self._populate_fitness()

            parents = self._select_n(2)
            for _ in range(num_crossover):
                self.population.append(self._crossover(*parents))

            self.population = [self._mutate(x) for x in self.population]
            self._populate_fitness()

            best_member, best_fitness = self._most_fit()
            if best_fitness > self.best_fitness:
                self.best_fitness = best_fitness
                self.best_member = deepcopy(best_member)

            if self.max_fitness is not None and self.best_fitness >= self.max_fitness:
                logger.info("TERMINATING - REACHED MAXIMUM FITNESS")
                return self.best_member, self.best_fitness
        logger.info("TERMINATING - REACHED MAXIMUM STEPS")
        return self.best_member, self.best_fitness
