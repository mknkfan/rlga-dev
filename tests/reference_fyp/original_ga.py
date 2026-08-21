from core.datastruct import Point, Machine
from feasibility.collision_detector import CollisionDetector
import numpy as np
import random
from typing import List, Tuple, Optional, Dict
import time
import copy

class GeneticAlgorithm:
    def __init__(self, machines: List[Machine], sequence: List[int], 
                 robot_position: Point, workspace_bounds: Tuple[float, float, float, float]):
        self.machines = machines
        self.sequence = sequence
        self.robot_position = robot_position
        self.workspace_bounds = workspace_bounds  # (min_x, max_x, min_y, max_y)
        
        # GA Parameters
        self.population_size = 200
        self.generations = 300
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.elitism_rate = 0.1
        
        # Performance tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def decode_chromosome(self, chromosome: np.ndarray) -> List[Machine]:
        """Decode chromosome into machine configurations"""
        machines = copy.deepcopy(self.machines)
        for i, machine in enumerate(machines):
            idx = i * 3
            machine.position.x = chromosome[idx]
            machine.position.y = chromosome[idx + 1]

            # Restrict rotation to nearest 0,90,180,270
            raw_rot = chromosome[idx + 2] % 360
            machine.rotation = round(raw_rot / 90) * 90
        return machines
    
    def calculate_total_distance(self, machines: List[Machine]) -> float:
        """Calculate total robot travel distance"""
        positions = [self.robot_position]
        
        for machine_id in self.sequence:
            machine = next(m for m in machines if m.id == machine_id)
            access_point = machine.get_access_point_world()
            positions.append(access_point)
        
        # Return to center
        positions.append(self.robot_position)
        
        total_distance = 0
        for i in range(len(positions) - 1):
            total_distance += positions[i].distance_to(positions[i + 1])
        
        return total_distance
    
    def check_collisions(self, machines: List[Machine]) -> bool:
        """Check for collisions between machines"""
        for i in range(len(machines)):
            for j in range(i + 1, len(machines)):
                poly1 = machines[i].get_corners()
                poly2 = machines[j].get_corners()
                if CollisionDetector.polygons_intersect(poly1, poly2):
                    return True
        return False
    
    def check_workspace_bounds(self, machines: List[Machine]) -> bool:
        """Check if machines are within workspace bounds"""
        min_x, max_x, min_y, max_y = self.workspace_bounds
        
        for machine in machines:
            corners = machine.get_corners()
            for corner in corners:
                if not (min_x <= corner.x <= max_x and min_y <= corner.y <= max_y):
                    return False
        return True
    
    def fitness_function(self, chromosome: np.ndarray) -> float:
        machines = self.decode_chromosome(chromosome)
        
        # Constraint: no machine may overlap center (0,0)
        for machine in machines:
            if CollisionDetector.point_in_polygon(Point(0,0), machine.get_corners()):
                return float('inf')
        
        # Check collisions and bounds
        if self.check_collisions(machines) or not self.check_workspace_bounds(machines):
            return float('inf')
        
        # Normal fitness evaluation
        distance = self.calculate_total_distance(machines)
        compactness_penalty = self.calculate_compactness_penalty(machines)
        accessibility_bonus = self.calculate_accessibility_bonus(machines)
        
        # return distance
        return distance + compactness_penalty - accessibility_bonus
    
    def calculate_compactness_penalty(self, machines: List[Machine]) -> float:
        """Penalty for spread out layouts"""
        if not machines:
            return 0
        
        x_coords = [m.position.x for m in machines]
        y_coords = [m.position.y for m in machines]
        
        x_span = max(x_coords) - min(x_coords)
        y_span = max(y_coords) - min(y_coords)
        
        return (x_span + y_span) * 0.1
    
    def calculate_accessibility_bonus(self, machines: List[Machine]) -> float:
        """Bonus for easily accessible layouts"""
        total_clearance = 0
        for machine in machines:
            access_point = machine.get_access_point_world()
            # Check clearance around access point
            clearance = self.calculate_clearance_around_point(access_point, machines, machine)
            total_clearance += clearance
        
        return total_clearance * 0.05
    
    def calculate_clearance_around_point(self, point: Point, machines: List[Machine], 
                                       exclude_machine: Machine) -> float:
        """Calculate clearance around a point"""
        min_distance = float('inf')
        for machine in machines:
            if machine.id == exclude_machine.id:
                continue
            
            corners = machine.get_corners()
            for corner in corners:
                distance = point.distance_to(corner)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 10.0
    
    def create_initial_population(self) -> List[np.ndarray]:
        """Create initial population with diverse layouts"""
        population = []
        min_x, max_x, min_y, max_y = self.workspace_bounds
        
        for _ in range(self.population_size):
            chromosome = []
            
            # Distribute machines across quadrants
            quadrants = [(min_x, 0, min_y, 0), (0, max_x, min_y, 0),
                        (min_x, 0, 0, max_y), (0, max_x, 0, max_y)]
            
            for i, machine in enumerate(self.machines):
                if i < len(quadrants):
                    qx1, qx2, qy1, qy2 = quadrants[i]
                    x = random.uniform(qx1 + machine.width/2, qx2 - machine.width/2)
                    y = random.uniform(qy1 + machine.height/2, qy2 - machine.height/2)
                else:
                    x = random.uniform(min_x + machine.width/2, max_x - machine.width/2)
                    y = random.uniform(min_y + machine.height/2, max_y - machine.height/2)
                
                rotation = random.uniform(0, 360)
                chromosome.extend([x, y, rotation])
            
            population.append(np.array(chromosome))
        
        return population
    
    def tournament_selection(self, population: List[np.ndarray], 
                           fitness_scores: List[float], k: int = 3) -> np.ndarray:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), k)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_index = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_index].copy()
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Uniform crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        return child1, child2
    
    def apply_bounds(self, chromosome: np.ndarray) -> np.ndarray:
        """Apply workspace bounds to chromosome"""
        min_x, max_x, min_y, max_y = self.workspace_bounds
        
        for i in range(0, len(chromosome), 3):
            machine_idx = i // 3
            machine_width = self.machines[machine_idx].width
            machine_height = self.machines[machine_idx].height
            
            chromosome[i] = np.clip(chromosome[i], 
                                  min_x + machine_width/2, 
                                  max_x - machine_width/2)
            chromosome[i+1] = np.clip(chromosome[i+1], 
                                    min_y + machine_height/2, 
                                    max_y - machine_height/2)
            chromosome[i+2] = chromosome[i+2] % 360
        
        return chromosome
    
    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        mutated = chromosome.copy()
        min_x, max_x, min_y, max_y = self.workspace_bounds
        
        for i in range(0, len(chromosome), 3):
            if random.random() < self.mutation_rate:
                # Position mutations
                mutated[i] += random.gauss(0, 2.0)  # x
                mutated[i+1] += random.gauss(0, 2.0)  # y
                
                # Rotation mutations (force multiples of 90)
                rot_options = [0, 90, 180, 270]
                mutated[i+2] = random.choice(rot_options)
                
                # Bounds
                machine_idx = i // 3
                w, h = self.machines[machine_idx].width, self.machines[machine_idx].height
                mutated[i] = np.clip(mutated[i], min_x + w/2, max_x - w/2)
                mutated[i+1] = np.clip(mutated[i+1], min_y + h/2, max_y - h/2)

        return self.apply_bounds(mutated)
    
    def adaptive_parameters(self, generation: int, diversity: float):
        """Adapt GA parameters based on progress"""
        # Increase mutation rate if diversity is low
        if diversity < 0.1:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.1)
        else:
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
        
        # Decrease crossover rate in later generations
        self.crossover_rate = 0.9 - 0.3 * (generation / self.generations)
    
    def calculate_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 1.0
        
        total_distance = 0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(population[i] - population[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0
    
    def local_search(self, chromosome: np.ndarray, max_iterations: int = 10) -> np.ndarray:
        """Hill climbing local search"""
        current = chromosome.copy()
        current_fitness = self.fitness_function(current)
        
        for _ in range(max_iterations):
            # Generate neighbor by small perturbation
            neighbor = current.copy()
            idx = random.randint(0, len(neighbor) - 1)
            
            if idx % 3 == 2:  # Rotation
                neighbor[idx] += random.gauss(0, 5.0)
                neighbor[idx] = neighbor[idx] % 360
            else:  # Position
                neighbor[idx] += random.gauss(0, 0.5)
            
            neighbor_fitness = self.fitness_function(neighbor)
            
            if neighbor_fitness < current_fitness:
                current = neighbor
                current_fitness = neighbor_fitness
        
        return current
    
    def extract_features(self, chromosome: np.ndarray) -> np.ndarray:
        machines = self.decode_chromosome(chromosome)
        
        features = []
        
        # Machine positions (normalized)
        min_x, max_x, min_y, max_y = self.workspace_bounds
        for machine in machines:
            features.extend([
                (machine.position.x - min_x) / (max_x - min_x),
                (machine.position.y - min_y) / (max_y - min_y),
                machine.rotation / 360.0
            ])
        
        # Distance features
        distances = []
        for i in range(len(self.sequence) - 1):
            m1 = next(m for m in machines if m.id == self.sequence[i])
            m2 = next(m for m in machines if m.id == self.sequence[i+1])
            dist = m1.get_access_point_world().distance_to(m2.get_access_point_world())
            distances.append(dist)
        
        features.extend([
            np.mean(distances),
            np.std(distances),
            np.max(distances),
            np.min(distances)
        ])
        
        # Pad or truncate to fixed size
        target_size = 50
        if len(features) > target_size:
            features = features[:target_size]
        elif len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        
        return np.array(features)
    
    def optimize(self, initial_population: Optional[List[np.ndarray]] = None) -> Tuple[List[Machine], float, Dict]:

        print("Starting Genetic Algorithm Optimization")
        
        # Initialize population (shared or new)
        if initial_population is None:
            population = self.create_initial_population()
        else:
            population = [chrom.copy() for chrom in initial_population]
            if len(population) != self.population_size:
                raise ValueError(
                    f"Initial population size {len(population)} != GA population_size {self.population_size}"
                )
        
        decoded_evolution = []
        best_solution = None
        best_fitness = float('inf')
        
        start_time = time.time()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for chromosome in population:
                fitness = self.fitness_function(chromosome)
                fitness_scores.append(fitness)
                
            
            # Track best solution
            min_fitness_idx = np.argmin(fitness_scores)
            if fitness_scores[min_fitness_idx] < best_fitness:
                best_fitness = fitness_scores[min_fitness_idx]
                best_solution = population[min_fitness_idx].copy()

            snapshot_interval = 20
            if snapshot_interval and (generation % snapshot_interval == 0):
                snap_machines = self.decode_chromosome(population[min_fitness_idx])
                decoded_evolution.append(copy.deepcopy(snap_machines))

            # Statistics
            avg_fitness = np.mean([f for f in fitness_scores if f != float('inf')])
            diversity = self.calculate_diversity(population)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Adaptive parameters
            # self.adaptive_parameters(generation, diversity)
            
            # Print progress
            if generation % 20 == 0:
                print(f"Generation {generation}: Best={best_fitness:.2f}, "
                      f"Avg={avg_fitness:.2f}, Diversity={diversity:.3f}")
            
            # Create new population
            new_population = []
            
            # Elitism
            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness_scores)[:elite_count]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # Local search on promising solutions
                if random.random() < 0.1:  # 10% chance
                    child1 = self.local_search(child1, max_iterations=5)
                
                new_population.extend([child1, child2])
            
            # Ensure exact population size
            population = new_population[:self.population_size]
        
        end_time = time.time()

        # Prepare results
        final_machines = self.decode_chromosome(best_solution)

        results = {
            'total_distance': self.calculate_total_distance(final_machines),
            'generations': self.generations,
            'execution_time': end_time - start_time,
            "decoded_evolution": decoded_evolution,
            "snapshot_interval": snapshot_interval,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'final_fitness': best_fitness,
        }
        
        return final_machines, best_fitness, results
