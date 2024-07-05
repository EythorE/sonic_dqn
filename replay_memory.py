import numpy as np

class ReplayMemory:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buf = np.empty(shape=maxlen, dtype=object)
        self.index = 0
        self.length = 0
        
    def append(self, data):
        self.buf[self.index] = data
        self.length = min(self.length + 1, self.maxlen)
        self.index = (self.index + 1) % self.maxlen
   
    def __len__(self):
        return self.length

    def sample(self, batch_size, with_replacement=True):
        if with_replacement:
            indices = np.random.randint(self.length, size=batch_size) # faster
        else:
            indices = np.random.permutation(self.length)[:batch_size]
        return self.buf[indices]


    def sample_memories(self, batch_size):
        # state, action, reward, next_state, continue
        cols = [[], [], [], [], []]
        for memory in self.sample(batch_size):
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)
    
    import functools 
    def fill_replay_memory(self, n, env, action_fn):
        # action_fn: action = fn(obs)
        iterator = range(n)
        if n > 10000:
            print("This is a lot!") # We need tqdm
            from tqdm import trange
            iterator = trange(n)
        obs, info = env.reset()
        for _ in iterator:
            action = action_fn(obs)
            next_obs, reward, done, truncated, info = env.step(action)
            self.append((obs, action, reward, next_obs, 1.0 - done))
            obs = next_obs
            if done or truncated:
                obs, info = env.reset()
        return
