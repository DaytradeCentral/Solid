from random import choice, randrange, random
from string import ascii_lowercase
from Solid.EvolutionaryAlgorithm import EvolutionaryAlgorithm
from unittest import TestCase, main

class Algorithm(EvolutionaryAlgorithm):
    """
    Tries to get a randomly-generated string to match some word
    """
    def __init__(self, word, *args, **kwargs):
        self.word = word
        super(Algorithm, self).__init__(*args, **kwargs)

    def _initial_population(self):
        return [''.join(choice(ascii_lowercase) for _ in self.word)
                for _ in range(50)]

    def _fitness(self, member):
        return sum(x == y for x, y in zip(self.word, member))

    def _crossover(self, parent1, parent2):
        index = randrange(len(self.word))
        return parent1[:index] + parent2[index:]

    def _mutate(self, member):
        if self.mutation_rate >= random():
            index = randrange(len(self.word))
            return member[:index] + choice(ascii_lowercase) + member[index+1:]
        return member

class TestEvolutionaryAlgorithm(TestCase):
    def test_algorithm(self):
        word = 'clout'
        algorithm = Algorithm(word, .5, .7, 500, max_fitness=None)
        algorithm.run()
        self.assertEqual(algorithm.best_member, word)

if __name__ == '__main__':
    from logging import basicConfig, INFO
    basicConfig(level=INFO)
    main()
