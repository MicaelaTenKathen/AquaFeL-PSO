from Benchmark.benchmark_functions import Benchmark_function
from Environment.map import Map

array, resolution = Map(1, 100, 150).black_white()
_z = Benchmark_function("e", array, resolution, 100, 150, w_ostacles=False, obstacles_on=False, randomize_shekel=False, sensor="", no_maxima=10,
                        load_from_db=False, file=0).create_map()