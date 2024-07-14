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
    
    def fill_replay_memory(self, n, env, action_fn):
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

class EpisodicReplay(ReplayMemory):

    def sample_memories(self, batch_size, Tmax=None):
        # memories contain: (obs[Ti, ...], actions[Ti], rewards[Ti], done)
        # out: batch_size, Ti, ...
        cols = [[], [], [], []]
        memories = self.sample(batch_size)
        
        if Tmax is None:
            T_memory = [len(m[0]) for m in memories]
            T = max(T_memory) # longest memory in sample
        else:
            T = Tmax+1 # one step will not be used in qnet
        B = batch_size
        C,W,H = memories[0][0][0].shape # [[obs[T0],...], [..,
        
        obs = np.zeros((B, T, C, W, H), dtype='int8')
        actions = np.zeros((B, T), dtype='int')
        rewards = np.zeros((B, T))
        continues = np.ones((B,T), dtype='bool')
        
        for i, (obsi, actionsi, rewardsi, donei) in enumerate(memories):
            Ti = len(obsi)
            if Ti > T:
                # sample from sequence
                si = np.random.randint(Ti-T)
            else:
                si = 0
            if Ti < T:
                Te = Ti
            else:
                Te = T
            obs[i, :Te] = obsi[si:si+T]
            actions[i,:Te] = actionsi[si:si+T]
            rewards[i,:Te] = rewardsi[si:si+T]
            continues[i, Te-1] = (not donei) and (si+T == Ti)
            continues[i, Te:] = False

        return obs, actions, rewards, continues


    def fill_replay_memory(self, n_episodes, env, action_fn):
        iterator = range(n_episodes)
        if n_episodes > 100:
            print("This is a lot!") # We need tqdm
            from tqdm import trange
            iterator = trange(n_episodes)
        for _ in iterator:
            obs, info = env.reset()
            done = truncated = False
            episode_memory = [[obs], [0], [None]]
            while not (done or truncated):
                action = action_fn(episode_memory[0], episode_memory[1])
                obs, reward, done, truncated, info = env.step(action)
                episode_memory[0].append(obs) 
                episode_memory[1].append(action) 
                episode_memory[2].append(reward) 
            # each episode are arrays with time as the first dimensions, done as a single value repr. the episode
            self.append([*episode_memory, done]) # (obs[T,C,W,H], actions[T], rewards[T], done)
        return
