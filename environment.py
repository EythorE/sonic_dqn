from collections import deque
import gymnasium as gym
import numpy as np

def ds_grayscale(obs):
    downscaled = obs[::2, ::2, :]
    r, g, b = downscaled[:,:,0], downscaled[:,:,1], downscaled[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray[None]

class StickyAction(gym.Wrapper):
    """
    Sticky action without frame motion blur.

    :param env: Environment to wrap
    :param action_repeat_n: Number of time to repeat the last action
    """
    def __init__(self, env: gym.Env, n_action_repeats: int):
        super().__init__(env)
        self.n_action_repeats = n_action_repeats

        self.height = env.observation_space.shape[0]//2
        self.width = env.observation_space.shape[1]//2

        self.observation_space = gym.spaces.Box(
            low=-128,
            high=127,
            shape=(1, self.height, self.width),
            dtype=np.dtype('int8'),
        )


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        gray = ds_grayscale(obs)
        obs = (gray  - 128).astype(np.int8)
        return obs, info
        

    def step(self, action: int):
        rew_sum = 0
        for _ in range(self.n_action_repeats):
            obs, rew, done, truncated, info = self.env.step(action)
            rew_sum += rew
            if done or truncated:
                break
        
        gray = ds_grayscale(obs)
        obs = (gray  - 128).astype(np.int8)

        reward = rew_sum / self.n_action_repeats
        return obs, reward, done, truncated, info
    

class SlowResponse(gym.Wrapper):
    """
    Sticky action with frame motion blur.

    :param env: Environment to wrap
    :param action_repeat_n: Number of time to repeat the last action
    """
    def __init__(self, env: gym.Env, n_action_repeats: int, frame_diff_length: int):
        super().__init__(env)
        self.n_action_repeats = n_action_repeats
        self.frame_diff_length = frame_diff_length
        self.frame_deque = deque(maxlen=self.frame_diff_length)

        self.height = env.observation_space.shape[0]//2
        self.width = env.observation_space.shape[1]//2

        self.observation_space = gym.spaces.Box(
            low=-128,
            high=127,
            shape=(1, self.height, self.width),
            dtype=np.dtype('int8'),
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frame_deque.clear()
        gray = ds_grayscale(obs)
        obs = (gray  - 128).astype(np.int8)
        return obs, info
        

    def step(self, action: int):
        rew_sum = 0
        for _ in range(self.n_action_repeats):
            obs, rew, done, truncated, info = self.env.step(action)
            rew_sum += rew
            if done or truncated:
                break
        
        gray = ds_grayscale(obs)
        # blur
        self.frame_deque.append(gray.astype(np.uint8))
        if len(self.frame_deque) >= self.frame_diff_length:
            old_gray = self.frame_deque.popleft()
            gray = 2/3*gray + 1/3*old_gray
            
        obs = (gray  - 128).astype(np.int8)
        reward = rew_sum / self.n_action_repeats
        return obs, reward, done, truncated, info


class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.

    Args:
        combos: Ordered list of lists of valid button combinations representing actions
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons  # Get the buttons from the underlying environment
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)  # Initialize a boolean array for the action space
            for button in combo:
                arr[buttons.index(button)] = True  # Set True for buttons included in the combo
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))  # Set discrete action space size

    def action(self, act):
        return self._decode_discrete_action[act].copy()  # Return a copy of the action array

class SonicDiscretizer(Discretizer):
    """
    Define a custom set of discrete actions for a specific game, using simplified button combinations.
    """
    def __init__(self, env):
        # Define custom combos based on the example given in the user's action2vec function
        self.combos = [
            [],            # 0: Noop
            ['B'],         # 1: Jump
            ['RIGHT'],     # 2: Move right
            ['LEFT'],      # 3: Move left
            ['DOWN'],      # 4: Crouch
            ['B', 'RIGHT'],# 5: Jump right
            ['B', 'LEFT'], # 6: Jump left
            ['B', 'DOWN'], # 7: Jump crouch
            ['DOWN', 'RIGHT'], # 8: Crouch right
            ['DOWN', 'LEFT']   # 9: Crouch left
        ]
        super().__init__(env, self.combos)
