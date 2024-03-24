##Workshop II - Swarm Intelligence and Sinergy: Ant Colony for the Traveling Salesman Problem
Code: 
import numpy as np
def generate_cities(number_cities: int) -> np.ndarray:

    """
    it generates a list cities and return it
    """

    cities = np.random.rand(number_cities, 3)
    return cities

def calculate_distance(point_1: np.array, point_2: np.array) -> float:

    """
      calculate the distance between two points using the euclidean formula and return the distance
    """

    return np.sqrt(np.sum((point_2 - point_1)**2))

En esta parte del código se definen las funciones para generar las ciudades de manera aleatoria con coordenadas (x,y,z), y la función para calcular la distancia entre dos ciudades, funciones que luego utilizaremos en la colonia de hormigas.
def ant_colony_optimization(cities, n_ants, n_iterations, alpha, beta, evaporation_rate, Q):

    """
    This function solves the Traveling Salesman Problem using Ant Colony Optimization.

    Parameters:
    - cities (list): List of cities.
    - n_ants (int): Number of ants.
    - n_iterations (int): Number of iterations.
    - alpha (float): It determines how much the ants are influenced by the pheromone trails left by other ants.
    - beta (float):  It determines how much the ants are influenced by the distance to the next city
    - evaporation_rate (float): Evaporation rate.
    - Q (float): It determines the intensity of the pheromone trail left behind by an ant.
    """
    # get the size of the array cities and define the pheromone array with ones using np.ones of the same size
    number_cities =  cities.shape[0]
    pheromone = np.ones((number_cities, number_cities))

    # initialize output metrics
    best_path = None
    best_path_length = np.inf

    # per each iteration the ants will build a path
    for iteration in range(n_iterations):
        paths = []  # store the paths of each ant
        path_lengths = []

        for ant in range(n_ants):
            visited = [False] * number_cities

            # you could start from any city, but let's start from a random one
            current_city = np.random.randint(number_cities)
            visited[current_city] = True
            path = [current_city]
            path_length = 0

            while False in visited:  # while there are unvisited cities
                unvisited = np.where(np.logical_not(visited))[0]
                probabilities = np.zeros(len(unvisited))

                # based on pheromone, distance and alpha and beta parameters, define the preference
                # for an ant to move to a city

                #define the equation for the ants, using alpha, beta, pheromones array, and the function calculate_distance
                for i, unvisited_city in enumerate(unvisited):
                    probabilities[i] = (pheromone[current_city, unvisited_city] ** alpha) * \
                                        ((1 / -(calculate_distance(cities[current_city], cities[unvisited_city])) ** beta))

                # normalize probabilities, it means, the sum of all probabilities is 1
                # HERE add normalization for calculated probabilities
                probabilities /= np.sum(probabilities)

                next_city = np.random.choice(unvisited, p=probabilities)
                path.append(next_city)
                # increase the cost of move through the path
                path_length += calculate_distance(
                    cities[current_city], cities[next_city]
                )
                visited[next_city] = True
                # move to the next city, for the next iteration
                current_city = next_city

            paths.append(path)
            path_lengths.append(path_length)

            # update with current best path, this is a minimization problem
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        # remove a bit of pheromone of all map, it's a way to avoid local minima
        pheromone *= evaporation_rate

        # current ant must add pheromone to the path it has walked
        for path, path_length in zip(paths, path_lengths):
            for i in range(number_cities - 1):
                pheromone[path[i], path[i + 1]] += Q / path_length
            pheromone[path[-1], path[0]] += Q / path_length
    return best_path, best_path_length


