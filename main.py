import pygame
import neat
import numpy as np
import math
import os
import pickle
import random


# REALISTIC physics constants
GRAVITY = 9.81  # Earth gravity
M = 1.0  # Cart mass (kg) - lighter for better control
m1 = 0.1  # First pendulum mass (kg) - lighter
m2 = 0.1  # Second pendulum mass (kg) - lighter
L1 = 0.5  # First pendulum length (m)
L2 = 0.5  # Second pendulum length (m)

# REALISTIC actuator limits
MAX_FORCE = 20.0  # N - reduced for stability
MOTOR_TIME_CONSTANT = 0.01  # Motor response time (s)
FORCE_NOISE_STD = 0.1  # Force noise standard deviation

# REALISTIC sensor parameters
ENCODER_RESOLUTION = 0.001  # Encoder resolution (rad)
POSITION_NOISE_STD = 0.0005  # Position sensor noise (m)

# OPTIMAL FRICTION - Key change for stability!
CART_FRICTION = 0.8  # Cart friction coefficient - helps prevent sliding
PENDULUM_FRICTION = 0.02  # Light pendulum friction - helps dampen oscillations

# Visuals
WIDTH, HEIGHT = 1200, 800
SCALE = 300  # Scale for visualization
TRACK_LIMIT = 2.0  # Track length (m)
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
LIME = (0, 255, 0)

# Simulation parameters
DT = 0.01  # Control timestep (100Hz)
EVALUATION_TIME = 30.0

# STRICT upright balance criteria
UPRIGHT_ANGLE_TOLERANCE = 0.1  # radians (about 6 degrees)
UPRIGHT_VELOCITY_TOLERANCE = 0.5  # rad/s
CART_POSITION_TOLERANCE = 0.3  # meters from center


CONFIG_TEXT = """
[NEAT]
fitness_criterion     = max
fitness_threshold     = 500000.0
pop_size              = 200
reset_on_extinction   = True

[DefaultGenome]
num_inputs            = 10
num_outputs           = 1
num_hidden            = 0
feed_forward          = True
enabled_default       = True
enabled_mutate_rate   = 0.1

activation_default    = tanh
activation_mutate_rate= 0.1
activation_options    = tanh sigmoid relu

aggregation_default   = sum
aggregation_mutate_rate = 0.0
aggregation_options   = sum

bias_init_mean        = 0.0
bias_init_stdev       = 1.0
bias_max_value        = 30.0
bias_min_value        = -30.0
bias_mutate_power     = 0.5
bias_mutate_rate      = 0.7
bias_replace_rate     = 0.1

response_init_mean    = 1.0
response_init_stdev   = 0.0
response_max_value    = 30.0
response_min_value    = -30.0
response_mutate_power = 0.2
response_mutate_rate  = 0.1
response_replace_rate = 0.0

weight_init_mean      = 0.0
weight_init_stdev     = 1.0
weight_max_value      = 30
weight_min_value      = -30
weight_mutate_power   = 0.5
weight_mutate_rate    = 0.8
weight_replace_rate   = 0.1

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

initial_connection    = full_direct
node_add_prob         = 0.2
node_delete_prob      = 0.2
conn_add_prob         = 0.5
conn_delete_prob      = 0.5

[DefaultReproduction]
elitism               = 10
survival_threshold    = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func  = max
max_stagnation        = 20
species_elitism       = 4
"""

class UprightDoublePendulumEnv:
    def __init__(self):
        self.reset()
        self.actual_force = 0.0
        self.commanded_force = 0.0
        
    def reset(self):
        """Reset to hanging down position (bottom)"""
        # Start hanging down (pi = hanging down, 0 = upright)
        self.x = random.uniform(-0.05, 0.05)
        self.dx = random.uniform(-0.1, 0.1)
        self.theta1 = math.pi + random.uniform(-0.2, 0.2)  # Hanging down
        self.dtheta1 = random.uniform(-0.2, 0.2)
        self.theta2 = math.pi + random.uniform(-0.2, 0.2)  # Hanging down
        self.dtheta2 = random.uniform(-0.2, 0.2)
        
        self.t = 0
        self.perfect_balance_time = 0
        self.max_perfect_balance_time = 0
        self.total_energy_used = 0
        self.actual_force = 0.0
        self.commanded_force = 0.0
        
        return self.get_state()

    def get_state(self):
        """Get sensor readings with realistic noise - enhanced state space"""
        # Add sensor noise
        x_noisy = self.x + random.gauss(0, POSITION_NOISE_STD)
        theta1_noisy = self.theta1 + random.gauss(0, ENCODER_RESOLUTION)
        theta2_noisy = self.theta2 + random.gauss(0, ENCODER_RESOLUTION)
        
        # Calculate angle differences for straight line detection
        angle_diff = theta1_noisy - theta2_noisy
        angle_diff_norm = math.atan2(math.sin(angle_diff), math.cos(angle_diff))
        
        return [
            np.clip(x_noisy / TRACK_LIMIT, -1, 1),
            np.clip(self.dx / 5.0, -1, 1),
            math.sin(theta1_noisy),
            math.cos(theta1_noisy),
            np.clip(self.dtheta1 / 10.0, -1, 1),
            math.sin(theta2_noisy),
            math.cos(theta2_noisy),
            np.clip(self.dtheta2 / 10.0, -1, 1),
            math.sin(angle_diff_norm),  # Angle difference between pendulums
            math.cos(angle_diff_norm)   # For straight line alignment
        ]

    def step(self, force_command):
        """Physics simulation step"""
        # Saturate and apply motor dynamics
        force_command = np.clip(force_command, -1, 1) * MAX_FORCE
        
        # Simple motor lag
        self.commanded_force += (force_command - self.commanded_force) * DT / MOTOR_TIME_CONSTANT
        self.actual_force = self.commanded_force + random.gauss(0, FORCE_NOISE_STD)
        
        # Track energy usage
        self.total_energy_used += abs(self.actual_force) * DT
        
        # Physics integration using Runge-Kutta 4th order
        self._integrate_physics()
        
        # Update time
        self.t += DT
        
        # Track perfect balance performance
        if self._is_perfectly_balanced():
            self.perfect_balance_time += DT
            self.max_perfect_balance_time = max(self.max_perfect_balance_time, self.perfect_balance_time)
        else:
            self.perfect_balance_time = 0
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        done = self._check_termination()
        
        return self.get_state(), reward, done
    
    def _integrate_physics(self):
        """Runge-Kutta 4th order integration"""
        state = [self.x, self.dx, self.theta1, self.dtheta1, self.theta2, self.dtheta2]
        
        # RK4 integration
        k1 = self._derivatives(state)
        k1 = [k * DT for k in k1]
        
        state2 = [state[i] + k1[i] * 0.5 for i in range(6)]
        k2 = self._derivatives(state2)
        k2 = [k * DT for k in k2]
        
        state3 = [state[i] + k2[i] * 0.5 for i in range(6)]
        k3 = self._derivatives(state3)
        k3 = [k * DT for k in k3]
        
        state4 = [state[i] + k3[i] for i in range(6)]
        k4 = self._derivatives(state4)
        k4 = [k * DT for k in k4]
        
        # Update state
        for i in range(6):
            increment = (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6
            state[i] += increment
        
        self.x, self.dx, self.theta1, self.dtheta1, self.theta2, self.dtheta2 = state
        
        # Handle cart limits
        if abs(self.x) > TRACK_LIMIT:
            self.x = np.clip(self.x, -TRACK_LIMIT, TRACK_LIMIT)
            self.dx *= -0.5  # Damped collision

    def _derivatives(self, state):
        """Calculate system derivatives WITH OPTIMAL FRICTION"""
        x, dx, theta1, dtheta1, theta2, dtheta2 = state
        
        # Trigonometric values
        s1, c1 = math.sin(theta1), math.cos(theta1)
        s2, c2 = math.sin(theta2), math.cos(theta2)
        s12 = math.sin(theta1 - theta2)
        c12 = math.cos(theta1 - theta2)
        
        # OPTIMAL FRICTION - This is the key change!
        # Cart friction: opposes cart motion
        friction_cart = -CART_FRICTION * dx * M * GRAVITY
        
        # Pendulum friction: opposes angular motion
        friction_pend1 = -PENDULUM_FRICTION * dtheta1
        friction_pend2 = -PENDULUM_FRICTION * dtheta2
        
        # Mass matrix elements
        M11 = M + m1 + m2
        M12 = (m1 + m2) * L1 * c1
        M13 = m2 * L2 * c2
        M22 = (m1 + m2) * L1 * L1
        M23 = m2 * L1 * L2 * c12
        M33 = m2 * L2 * L2
        
        # Right-hand side
        RHS1 = (self.actual_force + friction_cart - 
                (m1 + m2) * L1 * s1 * dtheta1 * dtheta1 - 
                m2 * L2 * s2 * dtheta2 * dtheta2)
        
        RHS2 = (friction_pend1 - (m1 + m2) * GRAVITY * L1 * s1 + 
                m2 * L1 * L2 * s12 * dtheta2 * dtheta2)
        
        RHS3 = (friction_pend2 - m2 * GRAVITY * L2 * s2 - 
                m2 * L1 * L2 * s12 * dtheta1 * dtheta1)
        
        # Solve system of equations
        det = M11 * (M22 * M33 - M23 * M23) - M12 * (M12 * M33 - M13 * M23) + M13 * (M12 * M23 - M13 * M22)
        
        if abs(det) < 1e-10:
            return [0, 0, 0, 0, 0, 0]
        
        # Calculate accelerations using Cramer's rule
        ddx = ((M22 * M33 - M23 * M23) * RHS1 - 
               (M12 * M33 - M13 * M23) * RHS2 + 
               (M12 * M23 - M13 * M22) * RHS3) / det
        
        ddtheta1 = (-(M12 * M33 - M13 * M23) * RHS1 + 
                    (M11 * M33 - M13 * M13) * RHS2 - 
                    (M11 * M23 - M12 * M13) * RHS3) / det
        
        ddtheta2 = ((M12 * M23 - M13 * M22) * RHS1 - 
                    (M11 * M23 - M12 * M13) * RHS2 + 
                    (M11 * M22 - M12 * M12) * RHS3) / det
        
        return [dx, ddx, dtheta1, ddtheta1, dtheta2, ddtheta2]

    def _is_perfectly_balanced(self):
        """Check if both pendulums are perfectly upright and aligned"""
        # Normalize angles to [-pi, pi]
        theta1_norm = math.atan2(math.sin(self.theta1), math.cos(self.theta1))
        theta2_norm = math.atan2(math.sin(self.theta2), math.cos(self.theta2))
        
        # Both pendulums must be very close to upright (0 radians)
        upright1 = abs(theta1_norm) < UPRIGHT_ANGLE_TOLERANCE
        upright2 = abs(theta2_norm) < UPRIGHT_ANGLE_TOLERANCE
        
        # Both pendulums must be nearly stationary
        stable1 = abs(self.dtheta1) < UPRIGHT_VELOCITY_TOLERANCE
        stable2 = abs(self.dtheta2) < UPRIGHT_VELOCITY_TOLERANCE
        
        # Cart must be near center
        centered = abs(self.x) < CART_POSITION_TOLERANCE
        
        # Angles must be nearly identical (forming straight line)
        aligned = abs(theta1_norm - theta2_norm) < UPRIGHT_ANGLE_TOLERANCE
        
        return upright1 and upright2 and stable1 and stable2 and centered and aligned

    def _calculate_reward(self):
        """Reward system for swinging up from bottom to upright"""
        reward = 0
        
        # Normalize angles
        theta1_norm = math.atan2(math.sin(self.theta1), math.cos(self.theta1))
        theta2_norm = math.atan2(math.sin(self.theta2), math.cos(self.theta2))
        
        # MASSIVE reward for perfect upright balance
        if self._is_perfectly_balanced():
            reward += 1000 + self.perfect_balance_time * 500
            
            # Extra bonus for maintaining perfect balance
            if self.perfect_balance_time > 1.0:
                reward += 2000
            if self.perfect_balance_time > 5.0:
                reward += 5000
        
        # Reward for being close to upright (0 radians)
        upright_reward1 = 200 * math.exp(-10 * abs(theta1_norm))
        upright_reward2 = 200 * math.exp(-10 * abs(theta2_norm))
        reward += upright_reward1 + upright_reward2
        
        # Reward for angle alignment (straight line)
        angle_diff = abs(theta1_norm - theta2_norm)
        alignment_reward = 100 * math.exp(-20 * angle_diff)
        reward += alignment_reward
        
        # Reward for cart centering
        centering_reward = 50 * math.exp(-5 * abs(self.x))
        reward += centering_reward
        
        # Reward for low velocities (stability)
        velocity_penalty = abs(self.dtheta1) + abs(self.dtheta2) + abs(self.dx)
        reward -= velocity_penalty * 10
        
        # Penalty for energy usage (encourage efficiency)
        reward -= 0.1 * abs(self.actual_force)
        
        # Base survival reward
        reward += 5
        
        return reward

    def _check_termination(self):
        """Check termination conditions"""
        # Cart hits track limits
        if abs(self.x) > TRACK_LIMIT:
            return True
        
        # Excessive velocities
        if abs(self.dx) > 15 or abs(self.dtheta1) > 30 or abs(self.dtheta2) > 30:
            return True
        
        # Time limit
        if self.t > EVALUATION_TIME:
            return True
        
        return False

    def draw(self, screen):
        """Enhanced visualization showing swing up from bottom"""
        screen.fill(WHITE)
        
        # Draw track
        track_y = HEIGHT // 2 + 100
        track_start = WIDTH // 2 - int(TRACK_LIMIT * SCALE)
        track_end = WIDTH // 2 + int(TRACK_LIMIT * SCALE)
        pygame.draw.line(screen, BLACK, (track_start, track_y), (track_end, track_y), 10)
        
        # Draw track limits
        pygame.draw.line(screen, RED, (track_start, track_y-15), (track_start, track_y+15), 6)
        pygame.draw.line(screen, RED, (track_end, track_y-15), (track_end, track_y+15), 6)
        
        # Draw center line
        pygame.draw.line(screen, GRAY, (WIDTH//2, track_y-5), (WIDTH//2, track_y+5), 2)
        
        # Draw UPRIGHT reference line (the target)
        ref_x = WIDTH // 2
        ref_y_top = track_y - int((L1 + L2) * SCALE)
        pygame.draw.line(screen, GRAY, (ref_x, track_y), (ref_x, ref_y_top), 5)
        
        # Draw cart
        cart_x = WIDTH // 2 + int(self.x * SCALE)
        cart_y = track_y
        cart_color = LIME if self._is_perfectly_balanced() else BLUE
        pygame.draw.rect(screen, cart_color, (cart_x - 25, cart_y - 20, 50, 40))
        pygame.draw.rect(screen, BLACK, (cart_x - 25, cart_y - 20, 50, 40), 3)
        
        # Calculate pendulum positions
        x1 = cart_x + int(L1 * SCALE * math.sin(self.theta1))
        y1 = cart_y - int(L1 * SCALE * math.cos(self.theta1))  # Negative for upward
        x2 = x1 + int(L2 * SCALE * math.sin(self.theta2))
        y2 = y1 - int(L2 * SCALE * math.cos(self.theta2))  # Negative for upward
        
        # Draw pendulum links with color coding based on how close to upright
        def get_pendulum_color(theta):
            theta_norm = math.atan2(math.sin(theta), math.cos(theta))
            if abs(theta_norm) < UPRIGHT_ANGLE_TOLERANCE:
                return LIME
            elif abs(theta_norm) < 0.3:
                return GREEN
            elif abs(theta_norm) < 0.6:
                return YELLOW
            else:
                return RED
        
        pend1_color = get_pendulum_color(self.theta1)
        pend2_color = get_pendulum_color(self.theta2)
        
        # Draw pendulum links
        pygame.draw.line(screen, pend1_color, (cart_x, cart_y), (x1, y1), 10)
        pygame.draw.line(screen, pend2_color, (x1, y1), (x2, y2), 10)
        
        # Draw pendulum masses
        pygame.draw.circle(screen, pend1_color, (x1, y1), 18)
        pygame.draw.circle(screen, BLACK, (x1, y1), 18, 3)
        pygame.draw.circle(screen, pend2_color, (x2, y2), 15)
        pygame.draw.circle(screen, BLACK, (x2, y2), 15, 3)
        
        # Draw status
        font = pygame.font.Font(None, 36)
        if self._is_perfectly_balanced():
            status_text = "🎯 PERFECT UPRIGHT BALANCE!"
            status_color = LIME
        else:
            theta1_norm = math.atan2(math.sin(self.theta1), math.cos(self.theta1))
            theta2_norm = math.atan2(math.sin(self.theta2), math.cos(self.theta2))
            if abs(theta1_norm) > 2.5 and abs(theta2_norm) > 2.5:
                status_text = "⬇️ SWINGING UP FROM BOTTOM"
                status_color = ORANGE
            else:
                status_text = "⬆️ APPROACHING UPRIGHT"
                status_color = YELLOW
        
        status_surface = font.render(status_text, True, status_color)
        screen.blit(status_surface, (WIDTH//2 - 200, 50))
        
        # Draw force indicator
        force_scale = 3
        force_end = cart_x + int(self.actual_force * force_scale)
        if abs(self.actual_force) > 0.5:
            color = RED if abs(self.actual_force) > MAX_FORCE * 0.7 else YELLOW
            pygame.draw.line(screen, color, (cart_x, cart_y - 50), (force_end, cart_y - 50), 6)
            pygame.draw.polygon(screen, color, [(force_end, cart_y - 50), 
                                              (force_end - 10 * np.sign(self.actual_force), cart_y - 55),
                                              (force_end - 10 * np.sign(self.actual_force), cart_y - 45)])
        
        pygame.display.flip()

def eval_genomes(genomes, config):
    """Evaluate genomes for swing-up task"""
    results = []
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        total_fitness = 0
        num_tests = 5
        
        for test in range(num_tests):
            env = UprightDoublePendulumEnv()
            episode_reward = 0
            
            while True:
                state = env.get_state()
                
                try:
                    output = net.activate(state)
                    force_command = output[0]
                    
                    _, reward, done = env.step(force_command)
                    episode_reward += reward
                    
                    if done:
                        break
                        
                except Exception as e:
                    episode_reward -= 2000
                    break
            
            total_fitness += episode_reward
        
        genome.fitness = total_fitness / num_tests
        results.append((genome_id, genome.fitness, env.max_perfect_balance_time))
    
    # Sort by fitness
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print top performers
    print(f"\n🏆 TOP 5 PERFORMERS:")
    for i, (gid, fitness, balance_time) in enumerate(results[:5]):
        print(f"  {i+1}. Genome {gid}: Fitness={fitness:.1f}, Perfect Balance={balance_time:.2f}s")

def train_neat():
    """Train NEAT controller for swing-up task"""
    # Create config file
    with open("config-upright", "w") as f:
        f.write(CONFIG_TEXT)
    
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "config-upright"
    )
    
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    
    print("🎯 Training SWING-UP Double Pendulum Controller WITH OPTIMAL FRICTION")
    print("=" * 70)
    print("🎯 GOAL: Swing BOTH pendulums from BOTTOM to UPRIGHT!")
    print("📏 Start: θ1 = θ2 = π (hanging down)")
    print("📏 Target: θ1 = θ2 = 0 (straight up)")
    print("🔧 OPTIMAL FRICTION:")
    print(f"   • Cart friction: {CART_FRICTION} (prevents sliding)")
    print(f"   • Pendulum friction: {PENDULUM_FRICTION} (dampens oscillations)")
    print("🎯 FORCING 250 GENERATIONS - NO EARLY STOPPING!")
    print("=" * 70)
    
    # Force exactly 250 generations by setting very high fitness threshold
    best_genome = None
    best_fitness = float('-inf')
    
    for generation in range(10):
        print(f"\n🔄 Generation {generation + 1}/250")
        
        # Evaluate all genomes
        genomes = list(pop.population.items())
        eval_genomes(genomes, config)
        
        # Find current best
        current_best = max(pop.population.values(), key=lambda g: g.fitness)
        if current_best.fitness > best_fitness:
            best_fitness = current_best.fitness
            best_genome = current_best
            print(f"🏆 NEW BEST FITNESS: {best_fitness:.1f}")
        
        # Reproduce for next generation (except on last generation)
        if generation < 10:
            pop.population = pop.reproduction.reproduce(config, pop.species, 
                                                      config.pop_size, generation)
            pop.species.speciate(config, pop.population, generation)
    
    # Save best controller
    with open("swing_up_pendulum_friction.pkl", "wb") as f:
        pickle.dump(best_genome, f)
    
    print(f"\n✅ Training complete! Best fitness after 250 generations: {best_fitness:.1f}")
    return best_genome

def draw_network(screen, net, config, genome, position=(900, 100), node_radius=10):
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    connections = genome.connections
    x0, y0 = position
    input_nodes = []
    hidden_nodes = []
    output_nodes = []

    # Layout inputs
    for i, nid in enumerate(input_keys):
        x = x0
        y = y0 + i * 60
        input_nodes.append((nid, (x, y)))
        pygame.draw.circle(screen, (0, 0, 255), (x, y), node_radius)

    # Layout outputs
    for i, nid in enumerate(output_keys):
        x = x0 + 300
        y = y0 + i * 60
        output_nodes.append((nid, (x, y)))
        pygame.draw.circle(screen, (0, 255, 0), (x, y), node_radius)

    # Layout hidden nodes
    hidden_ids = [nid for nid in genome.nodes if nid not in input_keys and nid not in output_keys]
    for i, nid in enumerate(hidden_ids):
        x = x0 + 150
        y = y0 + i * 60
        hidden_nodes.append((nid, (x, y)))
        pygame.draw.circle(screen, (255, 255, 0), (x, y), node_radius)

    # Combine all positions
    node_pos = dict(input_nodes + hidden_nodes + output_nodes)

    # Draw connections
    for (input_id, output_id), conn in connections.items():
        if not conn.enabled:
            continue
        start = node_pos.get(input_id)
        end = node_pos.get(output_id)
        color = (255, 0, 0) if conn.weight < 0 else (0, 0, 0)
        if start and end:
            pygame.draw.line(screen, color, start, end, 2)


def visualize_best():
    """Visualize the trained swing-up controller"""
    if not os.path.exists("swing_up_pendulum_friction.pkl"):
        print("❌ No trained controller found. Run training first!")
        return

    # Load config
    with open("config-upright", "w") as f:
        f.write(CONFIG_TEXT)

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "config-upright"
    )

    # Load best controller
    with open("swing_up_pendulum_friction.pkl", "rb") as f:
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Swing-Up Double Pendulum with Friction - Fitness: {genome.fitness:.1f}")
    clock = pygame.time.Clock()

    env = UprightDoublePendulumEnv()
    running = True
    paused = False

    print(f"🎯 Swing-Up Visualization with Optimal Friction (Fitness: {genome.fitness:.1f})")
    print("Controls: ESC=quit, R=reset, SPACE=pause")
    print("GOAL: Swing both pendulums from BOTTOM to UPRIGHT!")
    print(f"Cart friction: {CART_FRICTION}, Pendulum friction: {PENDULUM_FRICTION}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            state = env.get_state()
            try:
                output = net.activate(state)
                force_command = output[0]
                _, reward, done = env.step(force_command)

                if done:
                    print(f"Episode ended - Perfect balance time: {env.max_perfect_balance_time:.2f}s")
                    env.reset()

            except Exception as e:
                print(f"Error: {e}")
                env.reset()

        env.draw(screen)
        draw_network(screen, net, config, genome)
  # ✅ Draw the neural network on screen

        # Display detailed stats
        font = pygame.font.Font(None, 24)
        theta1_norm = math.atan2(math.sin(env.theta1), math.cos(env.theta1))
        theta2_norm = math.atan2(math.sin(env.theta2), math.cos(env.theta2))

        stats = [
            f"Time: {env.t:.1f}s",
            f"Perfect Balance: {env.perfect_balance_time:.2f}s (Best: {env.max_perfect_balance_time:.2f}s)",
            f"Cart Position: {env.x:.3f}m",
            f"Pendulum 1 Angle: {math.degrees(theta1_norm):+.1f}° (target: 0.0°)",
            f"Pendulum 2 Angle: {math.degrees(theta2_norm):+.1f}° (target: 0.0°)",
            f"Angular Vel 1: {env.dtheta1:+.2f}rad/s",
            f"Angular Vel 2: {env.dtheta2:+.2f}rad/s",
            f"Force: {env.actual_force:+.1f}N",
            f"Status: {'🎯 PERFECT!' if env._is_perfectly_balanced() else '🔄 SWINGING'}",
            "⏸️ PAUSED" if paused else "",
            f"🔧 Cart Friction: {CART_FRICTION}, Pendulum Friction: {PENDULUM_FRICTION}"
        ]

        for i, stat in enumerate(stats):
            if stat:
                if "PERFECT" in stat:
                    color = LIME
                elif "PAUSED" in stat:
                    color = RED
                elif "Friction" in stat:
                    color = BLUE
                else:
                    color = BLACK

                text = font.render(stat, True, color)
                screen.blit(text, (10, 100 + i * 22))

        pygame.display.flip()
        clock.tick(FPS)

        
        # Display detailed stats
        font = pygame.font.Font(None, 24)
        theta1_norm = math.atan2(math.sin(env.theta1), math.cos(env.theta1))
        theta2_norm = math.atan2(math.sin(env.theta2), math.cos(env.theta2))
        
        stats = [
            f"Time: {env.t:.1f}s",
            f"Perfect Balance: {env.perfect_balance_time:.2f}s (Best: {env.max_perfect_balance_time:.2f}s)",
            f"Cart Position: {env.x:.3f}m",
            f"Pendulum 1 Angle: {math.degrees(theta1_norm):+.1f}° (target: 0.0°)",
            f"Pendulum 2 Angle: {math.degrees(theta2_norm):+.1f}° (target: 0.0°)",
            f"Angular Vel 1: {env.dtheta1:+.2f}rad/s",
            f"Angular Vel 2: {env.dtheta2:+.2f}rad/s",
            f"Force: {env.actual_force:+.1f}N",
            f"Status: {'🎯 PERFECT!' if env._is_perfectly_balanced() else '🔄 SWINGING'}",
            "⏸️ PAUSED" if paused else "",
            f"🔧 Cart Friction: {CART_FRICTION}, Pendulum Friction: {PENDULUM_FRICTION}"
        ]
        
        for i, stat in enumerate(stats):
            if stat:
                if "PERFECT" in stat:
                    color = LIME
                elif "PAUSED" in stat:
                    color = RED
                elif "Friction" in stat:
                    color = BLUE
                else:
                    color = BLACK
                
                text = font.render(stat, True, color)
                screen.blit(text, (10, 100 + i * 22))
        
        clock.tick(FPS)


if __name__ == "__main__":
    print("🎯 PERFECT UPRIGHT Double Pendulum Challenge")
    print("=" * 60)
    print("🎯 GOAL: Keep BOTH pendulums in PERFECT vertical alignment!")
    print("📐 Target: θ1 = θ2 = 0° (straight up line)")
    print("🔧 Features:")
    print("  • Starts near upright position")
    print("  • Strict tolerance for perfect balance")
    print("  • Harsh selection - only best survive")
    print("  • Enhanced state space for alignment detection")
    print("  • Massive rewards for perfect upright balance")
    print("  • Real-time perfect balance tracking")
    print("=" * 60)
    
    choice = input("\nEnter 'train', 'visualize', or 'both': ").lower()
    
    if choice in ['train', 'both']:
        print("\n🏋️ Starting aggressive training...")
        train_neat()
    
    if choice in ['visualize', 'both']:
        print("\n🎮 Starting perfect balance visualization...")
        visualize_best()
    
    print("\n✅ Complete!")