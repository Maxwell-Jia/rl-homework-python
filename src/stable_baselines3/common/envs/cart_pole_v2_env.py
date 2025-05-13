import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.linalg  # For matrix inverse and expm
# import matplotlib.pyplot as plt # Add matplotlib for plotting # REMOVED
import pygame # ADDED

# --- Constants ---
MODEL_STATE_DIM = 8  # Dimension of the underlying physics model's state vector
MODEL_ACTION_DIM = 2 # Dimension of the underlying physics model's action vector
OBSERVATION_DIM = 6  # Dimension of the observation vector provided to the agent
ACTION_ENV_DIM = 1   # Dimension of the action vector received from the agent

# MAX_ACTION was 1000.0, let's rename for clarity if it's common acceleration
MAX_COMMON_WHEEL_ACCEL = 1000.0
MAX_ENV_ACTION = 1.0  # For normalized action space for the agent
MAX_EPISODE_STEPS = 1000

# Suggested bounds for state variables (can be tuned)
MAX_WHEEL_VEL = 50.0  # rad/s
MAX_BODY_VEL = 50.0   # rad/s
MAX_PENDULUM_VEL = 50.0 # rad/s

# Indices for the 8D internal state vector self.state:
# 0: theta_l (left wheel angle)
# 1: theta_r (right wheel angle)
# 2: theta_1 (body angle relative to vertical)
# 3: theta_2 (pendulum angle relative to vertical)
# 4: theta_l_dot (left wheel angular velocity)
# 5: theta_r_dot (right wheel angular velocity)
# 6: theta_1_dot (body angular velocity)
# 7: theta_2_dot (pendulum angular velocity)

# Default physical parameters from MATLAB code
DEFAULT_PARAMS = {
    "m_1": 0.9,
    "m_2": 0.1,
    "r": 0.0335,
    "L_1": 0.126,
    "L_2": 0.390,
    "g": 9.8,
    "Ts": 0.01  # Sampling time
}

# Reward weights removed as per CartPole style reward
# DEFAULT_REWARD_WEIGHTS = {
#     'theta_1': 5.0,
#     'theta_2': 10.0,
#     'theta_1_dot': 0.2,
#     'theta_2_dot': 0.2,
#     'control': 0.01
# }

class CartPoleV2Env(gym.Env):
    """
    Custom Gymnasium environment for the inverted pendulum on a self-balancing robot.
    Calculates discrete-time system matrices G and H internally based on physical parameters,
    replicating the logic from the provided MATLAB script.
    Uses CartPole-style reward: +1 for every step not terminated.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50} # UPDATED

    def __init__(self, params=None, render_mode=None):
        super().__init__()

        self.params = params if params is not None else DEFAULT_PARAMS
        # reward_weights removed
        # self.reward_weights = reward_weights if reward_weights is not None else DEFAULT_REWARD_WEIGHTS

        # Calculate intermediate parameters
        self.params['l_1'] = self.params['L_1'] / 2
        self.params['l_2'] = self.params['L_2'] / 2
        self.params['I_1'] = (1/12) * self.params['m_1'] * self.params['L_1']**2
        self.params['I_2'] = (1/12) * self.params['m_2'] * self.params['L_2']**2

        # --- Calculate G and H matrices ---
        self.G, self.H = self._calculate_discrete_matrices()

        # --- Define Spaces ---
        # Observation space (6D):
        # [theta_w, theta_1, theta_2, theta_w_dot, theta_1_dot, theta_2_dot]
        obs_low_bounds = np.array([
            -np.inf,                # theta_w (wheel angle)
            -np.pi / 2,             # theta_1 (body angle)
            -np.pi / 2,             # theta_2 (pendulum angle)
            -MAX_WHEEL_VEL,         # theta_w_dot
            -MAX_BODY_VEL,          # theta_1_dot
            -MAX_PENDULUM_VEL       # theta_2_dot
        ], dtype=np.float32)
        obs_high_bounds = np.array([
            np.inf,                 # theta_w
            np.pi / 2,              # theta_1
            np.pi / 2,              # theta_2
            MAX_WHEEL_VEL,          # theta_w_dot
            MAX_BODY_VEL,           # theta_1_dot
            MAX_PENDULUM_VEL        # theta_2_dot
        ], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low_bounds, obs_high_bounds, shape=(OBSERVATION_DIM,), dtype=np.float32)

        # Action space (1D): common wheel acceleration, normalized for the agent
        self.action_space = spaces.Box(-MAX_ENV_ACTION, MAX_ENV_ACTION, shape=(ACTION_ENV_DIM,), dtype=np.float32)

        # --- Environment State ---
        self.state = None # This will be the 8D model state
        self.current_step = 0
        self._max_episode_steps = MAX_EPISODE_STEPS

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # --- Pygame Rendering Attributes ---
        self.screen = None
        self.clock = None
        self.isopen = True # For human mode window state
        self._pygame_initialized = False # Track if pygame.init() has been called

        if self.render_mode in ["human", "rgb_array"]:
            # Common visual parameters
            self.screen_width = 800  # pixels
            self.screen_height = 600 # pixels
            # world_width_visible defines how many meters of cart travel are shown across screen_width
            self.world_width_visible = 3.0 # meters
            self.scale = self.screen_width / self.world_width_visible

            # Visual properties (in pixels for direct rendering)
            # self.cartwidth_px = 60 # REMOVED
            # self.cartheight_px = 40 # REMOVED
            # self.polewidth_px = 8 # REMOVED
            # self.pole_render_length_px = 80 # REMOVED

            self.wheel_radius_px = 15
            self.body_pole_render_length_px = 80
            self.body_pole_width_px = 12
            self.pendulum_pole_render_length_px = 60
            self.pendulum_pole_width_px = 8

            if not pygame.get_init():
                pygame.init()
                self._pygame_initialized = True

            if self.render_mode == "rgb_array":
                # Create an offscreen surface for rgb_array rendering if not in human mode
                # This surface is used if human mode isn't also active.
                self._rgb_surface = pygame.Surface((self.screen_width, self.screen_height))
        # Removed matplotlib plotting attributes and _setup_render_plot call
        # self.plot_fig = None
        # self.plot_axs = None
        # self.plot_lines = None
        # self.state_history = [] # REMOVED
        # self.step_history = [] # REMOVED
        # if self.render_mode == "human":
        #    self._setup_render_plot() # REMOVED

    def _calculate_discrete_matrices(self):
        """Replicates the MATLAB calculation to get discrete G and H."""
        m_1, m_2 = self.params['m_1'], self.params['m_2']
        r = self.params['r']
        L_1, l_1 = self.params['L_1'], self.params['l_1']
        L_2, l_2 = self.params['L_2'], self.params['l_2']
        g = self.params['g']
        I_1, I_2 = self.params['I_1'], self.params['I_2']
        Ts = self.params['Ts']

        # Construct p matrix
        p = np.zeros((4, 4), dtype=np.float32)
        p[0, 0] = 1
        p[1, 1] = 1
        p[2, 0] = (r / 2) * (m_1 * l_1 + m_2 * L_1)
        p[2, 1] = p[2, 0]
        p[2, 2] = m_1 * l_1**2 + m_2 * L_1**2 + I_1
        p[2, 3] = m_2 * L_1 * l_2
        p[3, 0] = (r / 2) * m_2 * l_2
        p[3, 1] = p[3, 0]
        p[3, 2] = m_2 * L_1 * l_2
        p[3, 3] = m_2 * l_2**2 + I_2

        # Construct q matrix (related to continuous dynamics)
        q = np.zeros((4, 10), dtype=np.float32)
        q[0, 8] = 1
        q[1, 9] = 1
        q[2, 2] = (m_1 * l_1 + m_2 * L_1) * g
        q[3, 3] = m_2 * g * l_2

        # Calculate intermediate temp matrix
        try:
            p_inv = scipy.linalg.inv(p)
        except np.linalg.LinAlgError:
            print("Error: p matrix is singular, cannot invert.")
            raise
        temp = p_inv @ q

        # Construct continuous-time A matrix
        A = np.zeros((8, 8), dtype=np.float32)
        A[0:4, 4:8] = np.eye(4)
        A[4:8, 0:8] = temp[:, 0:8]

        # Construct continuous-time B matrix
        B = np.zeros((8, 2), dtype=np.float32)
        B[4:8, :] = temp[:, 8:10]

        # Perform continuous-to-discrete conversion (c2d using matrix exponential)
        # Based on zero-order hold method
        n_states = A.shape[0]
        n_inputs = B.shape[1]

        # Build augmented matrix [[A, B], [0, 0]]
        augmented_matrix = np.zeros((n_states + n_inputs, n_states + n_inputs), dtype=np.float32)
        augmented_matrix[:n_states, :n_states] = A
        augmented_matrix[:n_states, n_states:] = B
        # Lower block remains zero

        # Compute matrix exponential
        M = scipy.linalg.expm(augmented_matrix * Ts)

        # Extract G and H
        G = M[:n_states, :n_states]
        H = M[:n_states, n_states:]

        return G.astype(np.float32), H.astype(np.float32)

    def _get_obs(self):
        # Converts the 8D internal state to a 6D observation for the agent.
        # Assumes self.state[0] (theta_l) and self.state[1] (theta_r) are kept consistent,
        # and self.state[4] (theta_l_dot) and self.state[5] (theta_r_dot) are kept consistent.
        # We pick the first one of each pair for the common wheel values.
        # obs: [theta_w, theta_1, theta_2, theta_w_dot, theta_1_dot, theta_2_dot]
        if self.state is None:
            # This can happen if _get_obs is called before reset (e.g. by a wrapper)
            # Return a zero observation matching the space
            return np.zeros(OBSERVATION_DIM, dtype=np.float32)

        return np.array([
            self.state[0],  # theta_w (from theta_l)
            self.state[2],  # theta_1 (body angle)
            self.state[3],  # theta_2 (pendulum angle)
            self.state[4],  # theta_w_dot (from theta_l_dot)
            self.state[6],  # theta_1_dot (body_dot)
            self.state[7]   # theta_2_dot (pendulum_dot)
        ], dtype=np.float32)

    def _get_info(self):
        # Include G and H in info if needed by the agent/controller later
        # return {"G": self.G_matrix, "H": self.H_matrix}
        return {}

    def _calculate_reward(self, prev_state_8d, model_action_2d, terminated):
        """ Calculates reward to encourage stable balance and minimize oscillations.
        -10 if terminated due to falling.
        Otherwise, +1 alive_bonus minus penalties for angle/velocity deviations and control effort.
        """
        if terminated:
            return -10.0
        else:
            alive_bonus = 1.0

            # Penalties are based on the current state (self.state) after the action
            current_body_angle = self.state[2]        # theta_1 from state x[k+1]
            current_pendulum_angle = self.state[3]    # theta_2 from state x[k+1]
            current_body_velocity = self.state[6]     # theta_1_dot from state x[k+1]
            current_pendulum_velocity = self.state[7] # theta_2_dot from state x[k+1]

            # Control effort from the applied model action u[k]
            # model_action_2d[0] is the common wheel acceleration, which can be up to MAX_COMMON_WHEEL_ACCEL (e.g., 1000.0)
            control_effort_raw = model_action_2d[0]

            # --- Tunable weights for penalties ---
            # You will likely need to adjust these weights based on experimentation.
            w_body_angle = 0.2        # Penalty for body angle deviation
            w_pendulum_angle = 0.2    # Penalty for pendulum angle deviation (often more critical)
            w_body_velocity = 0.01     # Penalty for body angular velocity
            w_pendulum_velocity = 0.01 # Penalty for pendulum angular velocity
            # MAX_COMMON_WHEEL_ACCEL is 1000. So (control_effort_raw)^2 can be 1e6.
            # This weight needs to be very small if penalizing raw acceleration.
            w_control = 0.000001      # Penalty for large control actions (raw wheel acceleration)
            # Consider normalizing control_effort if raw values are too large, or use a smaller weight.

            reward = alive_bonus \
                     - w_body_angle * current_body_angle**2 \
                     - w_pendulum_angle * current_pendulum_angle**2 \
                     - w_body_velocity * current_body_velocity**2 \
                     - w_pendulum_velocity * current_pendulum_velocity**2 \
                     - w_control * control_effort_raw**2

            return float(reward)

    def reset(self, seed=None, options=None, initial_state=None):
        super().reset(seed=seed)

        # Allow setting a specific initial 8D state, otherwise use random start
        if initial_state is not None:
             self.state = np.array(initial_state, dtype=np.float32)
             if self.state.shape != (MODEL_STATE_DIM,): # Check against model state dim
                 raise ValueError(f"Provided initial_state must have shape ({MODEL_STATE_DIM},)")
             # Ensure wheel consistency if user provides initial state
             # (Could add a check or force consistency here if desired)
             # e.g., self.state[1] = self.state[0]; self.state[5] = self.state[4]
        else:
            # Initialize state near upright equilibrium, small random angles, zero velocities
            # Ensure theta_l = theta_r and theta_l_dot = theta_r_dot = 0
            # wheel_angle_init = np.float32(self.np_random.uniform(low=-0.1, high=0.1))
            # body_angle_init = np.float32(self.np_random.uniform(low=-0.05, high=0.05))
            # pendulum_angle_init = np.float32(self.np_random.uniform(low=-0.05, high=0.05))

            # Increased initialization ranges for more challenging starts
            wheel_angle_init = np.float32(self.np_random.uniform(low=-0.5, high=0.5))
            body_angle_init = np.float32(self.np_random.uniform(low=-0.5, high=0.5))
            pendulum_angle_init = np.float32(self.np_random.uniform(low=-0.5, high=0.5))

            initial_angles = np.array([
                wheel_angle_init,    # theta_l
                wheel_angle_init,    # theta_r (consistent)
                body_angle_init,     # theta_1
                pendulum_angle_init  # theta_2
            ], dtype=np.float32)
            
            # All initial velocities are zero
            initial_velocities = np.zeros(4, dtype=np.float32)
            self.state = np.concatenate([initial_angles, initial_velocities])

        # Reset step counter
        self.current_step = 0
        # Removed state_history and step_history for plotting
        # self.state_history = []
        # self.step_history = []

        # Removed matplotlib plot reset logic
        # if self.render_mode == "human" and self.plot_lines:
        #    ...

        # Return initial observation and info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render() # Render initial state for human mode

        return observation, info

    def step(self, action):
        # Ensure action is numpy array and clip (action is 1D from agent)
        action_1d = np.asarray(action, dtype=np.float32).reshape((ACTION_ENV_DIM,)) # Ensure shape
        # Agent action is in [-MAX_ENV_ACTION, MAX_ENV_ACTION] (e.g., [-1, 1] after clipping)
        clipped_agent_action_1d = np.clip(action_1d, self.action_space.low, self.action_space.high)

        # Scale the normalized agent action to the physical model's expected range
        physical_action_value = clipped_agent_action_1d[0] * MAX_COMMON_WHEEL_ACCEL

        # Convert 1D physical action to 2D model action [u_common, u_common]
        model_action_2d = np.array([physical_action_value, physical_action_value], dtype=np.float32)

        if self.state is None:
            raise ValueError("Environment must be reset before calling step.")

        # Store state before transition (8D model state)
        prev_state = self.state.copy() # Use copy to avoid modification issues

        # Apply discrete dynamics: x[k+1] = G*x[k] + H*u[k]
        # self.state is 8D, model_action_2d is 2D
        self.state = self.G @ prev_state + self.H @ model_action_2d

        self.current_step += 1

        # Check termination based on the *new* 8D state
        # Indices 2 and 3 correspond to theta_1 (body) and theta_2 (pendulum)
        terminated = bool(
            abs(self.state[2]) > np.pi / 2 or # Body angle theta_1 too large
            abs(self.state[3]) > np.pi / 2    # Pendulum angle theta_2 too large
        )

        # Calculate reward (CartPole style: +1 for this step)
        # The 'action' argument to _calculate_reward should be the action applied to the model
        reward = self._calculate_reward(prev_state, model_action_2d, terminated)

        # Check truncation: episode length exceeded?
        truncated = bool(self.current_step >= self._max_episode_steps)

        observation = self._get_obs() # Get 6D observation
        info = self._get_info()

        if self.render_mode == "human":
            self.render() # Call new pygame render

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id if self.spec else "UnknownEnv"}", render_mode="human")'
            )
            return

        # Initialize Pygame screen/surface for drawing
        draw_surface = None
        current_screen_width = self.screen_width
        current_screen_height = self.screen_height

        if self.render_mode == "human":
            if not self.isopen:  # Window was closed by user
                return

            if self.screen is None:  # Initialize display for human mode
                if not pygame.get_init(): # Ensure pygame is initialized
                    pygame.init()
                    self._pygame_initialized = True
                pygame.display.init() # Initialize display module specifically
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("CartPoleV2")
                if self.clock is None:
                    self.clock = pygame.time.Clock()
            draw_surface = self.screen

            for event in pygame.event.get():  # Human mode handles events
                if event.type == pygame.QUIT:
                    pygame.display.quit()  # Quit display part
                    self.isopen = False
                    self.screen = None  # Mark as closed, allows re-opening
                    # Note: pygame.quit() (full uninitialization) is handled in self.close()
                    return
        
        elif self.render_mode == "rgb_array":
            if self.screen is not None: # Human mode might be active, use its screen content
                draw_surface = self.screen
            else: # Human mode not active. Use the pre-initialized _rgb_surface.
                if not hasattr(self, '_rgb_surface') or self._rgb_surface is None:
                    # This case should ideally be covered by __init__ ensuring _rgb_surface exists
                    if not pygame.get_init(): pygame.init(); self._pygame_initialized = True
                    self._rgb_surface = pygame.Surface((self.screen_width, self.screen_height))
                draw_surface = self._rgb_surface
        
        if draw_surface is None: # Should not happen if render_mode is valid
            return

        # --- Common Drawing Logic ---
        # Colors
        white = (255, 255, 255)
        # cart_color = (150, 150, 150)  # Grey # REMOVED
        # pole_color = (100, 50, 50)    # Brownish # REMOVED
        track_color = (70, 70, 70)
        wheel_color = (50, 50, 50)      # Dark grey for wheel
        body_pole_color = (80, 130, 180) # Bluish for body
        pendulum_pole_color = (180, 80, 80) # Reddish for pendulum

        draw_surface.fill(white)

        if self.state is None: # If state is None (e.g. pre-reset), draw blank
            if self.render_mode == "human" and self.screen: # Ensure screen exists for human mode
                pygame.display.flip()
                if self.clock: self.clock.tick(self.metadata["render_fps"])
            elif self.render_mode == "rgb_array":
                return np.transpose(pygame.surfarray.array3d(draw_surface), axes=(1, 0, 2))
            return

        # --- Calculate positions from self.state ---
        # Cart's horizontal position in meters (from wheel angle)
        # self.state[0] is theta_l (left wheel angle), self.params['r'] is wheel radius
        cart_x_meters = self.state[0] * self.params['r']
        
        # Convert to screen coordinates for the wheel center
        wheel_center_x_screen = cart_x_meters * self.scale + current_screen_width / 2.0
        # Position wheel near the bottom, accounting for its radius
        wheel_center_y_screen = current_screen_height * 0.85 - self.wheel_radius_px 

        # Body pole angle (theta_1)
        body_angle_rad = self.state[2]
        # Pendulum pole angle (theta_2)
        pendulum_angle_rad = self.state[3]

        # --- Draw Track ---
        track_y = wheel_center_y_screen + self.wheel_radius_px # Track below the wheel
        pygame.draw.line(draw_surface, track_color, (0, int(track_y)), (current_screen_width, int(track_y)), 2)

        # --- Draw Wheel (as a circle) ---
        pygame.draw.circle(
            draw_surface, 
            wheel_color, 
            (int(wheel_center_x_screen), int(wheel_center_y_screen)), 
            self.wheel_radius_px
        )

        # --- Draw Body Pole (Pole 1) ---
        # Starts from the center of the wheel
        body_pole_bottom_x = wheel_center_x_screen
        body_pole_bottom_y = wheel_center_y_screen

        body_pole_top_x = body_pole_bottom_x + self.body_pole_render_length_px * np.sin(body_angle_rad)
        body_pole_top_y = body_pole_bottom_y - self.body_pole_render_length_px * np.cos(body_angle_rad)

        pygame.draw.line(
            draw_surface,
            body_pole_color,
            (int(body_pole_bottom_x), int(body_pole_bottom_y)),
            (int(body_pole_top_x), int(body_pole_top_y)),
            self.body_pole_width_px
        )

        # --- Draw Pendulum Pole (Pole 2) ---
        # Starts from the top of the body pole
        pendulum_pole_bottom_x = body_pole_top_x
        pendulum_pole_bottom_y = body_pole_top_y

        pendulum_pole_top_x = pendulum_pole_bottom_x + self.pendulum_pole_render_length_px * np.sin(pendulum_angle_rad)
        pendulum_pole_top_y = pendulum_pole_bottom_y - self.pendulum_pole_render_length_px * np.cos(pendulum_angle_rad)

        pygame.draw.line(
            draw_surface,
            pendulum_pole_color,
            (int(pendulum_pole_bottom_x), int(pendulum_pole_bottom_y)),
            (int(pendulum_pole_top_x), int(pendulum_pole_top_y)),
            self.pendulum_pole_width_px
        )

        if self.render_mode == "human":
            if self.screen: # Ensure screen is still valid (e.g. not closed during event loop)
                 pygame.display.flip()
            if self.clock: self.clock.tick(self.metadata["render_fps"])
            return # For human mode, returns None
        
        elif self.render_mode == "rgb_array":
            # Pygame surfarray is W, H, C. SB3 expects H, W, C.
            return np.transpose(pygame.surfarray.array3d(draw_surface), axes=(1, 0, 2))

    # _render_frame method REMOVED (its role is now part of render())

    def close(self):
        # Clean up Pygame resources
        if self.screen is not None: # If human mode display was initialized
            pygame.display.quit()
            self.screen = None
        
        # self._rgb_surface is a Surface, will be garbage collected.
        # No specific close needed other than pygame.quit().

        if self._pygame_initialized: # If pygame.init() was called by this env instance
            pygame.quit()
            self._pygame_initialized = False
        
        self.isopen = True # Reset for potential next use/re-initialization
        if self.clock is not None:
            self.clock = None
        # Removed matplotlib cleanup logic
        # if self.render_mode == "human" and self.plot_fig is not None:
        #    ...