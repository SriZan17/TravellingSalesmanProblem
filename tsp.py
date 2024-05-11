# Purpose: To solve the Travelling Salesman Problem using Genetic Algorithm

from __future__ import annotations
from typing import Tuple, List, Any
from chromosome import Chromosome
from genetic_algorithm import GeneticAlgorithm
from random import sample


def main():
    CITIES = [(0, 0), (1, 5), (5, 2), (2, 3), (6, 4), (7, 7), (8, 6), (9, 7), (10, 8)]
    POPULATION_SIZE: int = 100
    THRESHOLD: float = 0.90
    MAX_GENERATIONS: int = 100
    MUTATION_CHANCE: float = 0.2
    CROSSOVER_CHANCE: float = 0.7
    # NUMBER_OF_CITIES: int = 20
    # initial_population: List[TSP] = [
    # TSP.random_instance(NUMBER_OF_CITIES) for _ in range(POPULATION_SIZE)
    # ]

    initial_population: List[TSP] = [
        TSP.random_instance(CITIES) for _ in range(POPULATION_SIZE)
    ]
    ga: GeneticAlgorithm[TSP] = GeneticAlgorithm(
        initial_population=initial_population,
        threshold=THRESHOLD,
        max_generations=MAX_GENERATIONS,
        mutation_chance=MUTATION_CHANCE,
        crossover_chance=CROSSOVER_CHANCE,
    )
    result: TSP = ga.run()
    print(result)


class TSP(Chromosome):
    def __init__(self, lst: List[Any]) -> None:
        self.lst: List[Any] = lst

    @property
    def distance(self) -> int:
        return sum(
            (
                (self.lst[i][0] - self.lst[i + 1][0]) ** 2
                + (self.lst[i][1] - self.lst[i + 1][1]) ** 2
            )
            ** 0.5
            for i in range(len(self.lst) - 1)
        )

    def fitness(self) -> float:
        return 1 / (self.distance + 1)

    @classmethod
    def random_instance(cls, cities) -> TSP:
        return TSP([cities[0]] + sample(cities[1:], k=len(cities) - 1) + [cities[0]])

    def crossover(self, other: TSP) -> Tuple[TSP, TSP]:
        idx1, idx2 = sorted(sample(range(1, len(self.lst) - 1), k=2))
        middle1 = self.lst[idx1:idx2]
        middle2 = other.lst[idx1:idx2]

        def get_child(middle, parent):
            not_in_middle = [item for item in parent.lst if item not in middle]
            return TSP(not_in_middle[:idx1] + middle + not_in_middle[idx1:])

        return get_child(middle1, other), get_child(middle2, self)

    def mutate(self) -> None:  # swap two locations
        idx1, idx2 = sample(range(1, len(self.lst) - 1), k=2)
        self.lst[idx1], self.lst[idx2] = self.lst[idx2], self.lst[idx1]

    def __str__(self) -> str:
        return f"Route: {self.lst} Distance: {self.distance}"


if __name__ == "__main__":
    main()
