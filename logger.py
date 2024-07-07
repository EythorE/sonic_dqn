from pathlib import Path
from datetime import datetime
import time
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import json

class Logger():

    def __init__(self, logdir=None, log_frequency=1000, name="log", cfg=None):
        self.start_time = datetime.now()
        formatted_time = self.start_time.strftime("%y%m%d-%H%M")
        self.logdir = Path(logdir) / f"{name}_{formatted_time}/"
        self.logdir.mkdir(exist_ok=True, parents=True)
        self.tb = SummaryWriter(self.logdir.as_posix())
        if cfg:
            from pprint import pprint
            print(self.logdir.name)
            pprint(cfg)
            self.tb.add_text('cfg', json.dumps(cfg, indent=2))
            if Path(cfg['environment']['scenario']).is_file():
                with open(cfg['environment']['scenario']) as f:
                    scenario=json.load(f)
                self.tb.add_text('scenario', json.dumps(scenario, indent=2))
                print('scenario:', scenario)
        self.log_frequency = log_frequency

        self.n_episode_steps = 0
        self.episodes_played = 0
        self.current_episode_steps = 0
        self.n_episode = 0
        self.episode_rewards = 0
        self.episode = defaultdict(float)

        self.n_training_steps = 0
        self.n_training = 0
        self.training_loss = 0

        self.sps_time = time.time()
        self.sps_counter = 0
    
    def sps_clock(self):
        time_interval = time.time() - self.sps_time
        steps = self.sps_counter
        sps = steps / time_interval
        self.sps_counter = 0
        self.sps_time = time.time()
        return sps


    def uptime(self):
        duration = datetime.now() - self.start_time
        total_seconds = int(duration.total_seconds())
        hours, remainder = divmod(total_seconds, 60 * 60)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}:{minutes:02}:{seconds:02}"


    def close(self):
        self.tb.close()
        duration = self.uptime()
        print(
            f"Total time: {uptime}\n",
            f"Episodes played: {self.n_episodes_played}",
            f"Number of training_steps: {n_training_steps}",
            f"Total game frames: {n_episode_steps}"
        )


    def training_step(self, loss_info, global_step):
        self.n_training_steps += 1
        self.n_training += 1
        self.training_loss += loss_info["loss"]
        for key, value in loss_info.items():
            self.tb.add_scalar(f'training/{key}', value, global_step)


    def episode_step(self, reward, done, truncated, info, global_step):
        # returns true if log was recorded, else false
        self.sps_counter += 1
        self.n_episode_steps += 1
        self.current_episode_steps += 1
        self.episode_rewards += reward
        
        if done or truncated:
            # end of episode
            self.episodes_played +=1
            self.n_episode += 1
            self.episode["completed"] += (done and info["lives"] == 3) 
            self.episode["terminated"] += done
            self.episode["truncated"] += truncated
            self.episode["cumulative_reward"] += self.episode_rewards
            self.episode["num_steps"] += self.current_episode_steps 
            self.episode['screen_x_frac'] += info['screen_x']/info['screen_x_end']
            self.episode['score'] += info['score']
            self.episode['rings'] += info["rings"]
            self.episode['level_end_bonus'] += info["level_end_bonus"]
            if not (info['screen_x']>=info['screen_x_end']) == (done and info["lives"] >= 3):
                # If the level is completed, sonic should have at least 3 lives,
                # possibly more do to bonuses
                print("warning: screen_x and done(with >=3 lifes) dont agree!")
                print(info, f"done={done}, truncated={truncated}, global_step={global_step}")
            # reset episode cumulators
            self.current_episode_steps = 0
            self.episode_rewards = 0

        if (global_step % self.log_frequency) == 0:
            self.write(global_step)
            return True

        return False


    def write(self, global_step):
        strs = [
            f"global_step: {global_step}",
            f"sps: {self.sps_clock():.2f}",
            f"n_train: {self.n_training_steps}",
            f"n_episodes: {self.n_episode}"
        ]

        if self.n_training_steps:
            self.training_loss /= self.n_training
            strs.append(f"loss: {self.training_loss}")
            self.n_training = 0
            self.training_loss = 0
        
        # average over n_episode
        if self.n_episode:
            self.tb.add_scalar('episode/n_episodes', self.n_episode , global_step)
            for key, value in self.episode.items():
                value = value / self.n_episode
                self.tb.add_scalar('episode/'+key, value, global_step)
                strs.append(f"{key}: {value}")
            self.n_episode = 0
            self.episode = defaultdict(float)
        print(*strs, sep= ', ')

