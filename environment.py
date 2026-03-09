import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficIntersectionEnv(gym.Env):
    """Custom Environment for a 4-way traffic intersection."""
    metadata = {"render_modes": ["human", "console"]}

    def __init__(self):
        super(TrafficIntersectionEnv, self).__init__()
        
        # State: [North_count, South_count, East_count, West_count, EV_Flag]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32), 
            high=np.array([300, 300, 300, 300, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: Green light duration for each phase [Phase_NS, Phase_EW, Phase_N_Turn, Phase_S_Turn]
        # Bounded between 10 seconds and 60 seconds.
        self.action_space = spaces.Box(
            low=10.0, high=60.0, shape=(4,), dtype=np.float32
        )
        
        self.state = np.zeros(5, dtype=np.float32)

    def step(self, action):
        # 1. Simulate traffic flow for the duration of the chosen actions
        total_cycle_time = np.sum(action)
        
        # [Simulate vehicle departures and arrivals over `total_cycle_time`]
        # self._simulate_traffic(action)
        
        # 2. Calculate Wait Times & Check Emergency Vehicles
        cumulative_wait_time = self._get_cumulative_wait() # Mock function
        ev_delayed = self._is_ev_delayed(action)           # Mock function
        
        # 3. Reward Function (Massive penalty for EV delay)
        reward = -cumulative_wait_time
        if ev_delayed:
            reward -= 10000.0  # Green Corridor Enforcement
            
        # 4. Get next state
        self.state = self._get_current_state()
        
        terminated = False  # Set conditions for episode end if necessary
        truncated = False
        info = {"ev_delayed": ev_delayed}
        
        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(5, dtype=np.float32)
        # Randomize initial traffic for robust training
        return self.state, {}

    # Placeholder simulation methods
    def _get_cumulative_wait(self): return 50.0 
    def _is_ev_delayed(self, action): return False
    def _get_current_state(self): return np.array([12, 15, 8, 20, 0], dtype=np.float32)