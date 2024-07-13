# Sonic the Hedgehog 2 - Emerald Hill Zone Part 1 DQN learning
This is a refactor of a [project](https://github.com/EythorE/sonic2_reinforcement_learning) from 2018 where DQN was implemented in Tensorflow 1.x with Openai's [gym-retro](https://github.com/openai/retro). Here it is refactored to Pytorch 2.x using the currently maintained [stable-retro](https://github.com/Farama-Foundation/stable-retro/).

## Setup
```bash
python3 -m venv "venv"
. venv/bin/activate
pip install -r requirements.txt
mkdir roms && wget -P roms https://archive.org/download/ni-roms/roms/Sega%20-%20Mega%20Drive%20-%20Genesis.zip/Sonic%20The%20Hedgehog%202%20%28World%29%20%28Rev%20A%29.zip
python3 -m retro.import ./roms 
```
[[Importing ROMs]](https://stable-retro.farama.org/getting_started/#importing-roms)


## Train
```bash
python train_dqn.py
```
Follow the progress with Tensorboard
```bash
tensorboard --logdir ./runs
```


## Record and play
### Play (and record) an epsiode played with trained DQN
```bash
python play_episode.py path/to/weights.pt greedy --record ./movie.bk2
```
### Record a human playing an episode
```bash
python human_record.py --record ./movie.bk2
```
### Replay an episode from an movie (converts to mp4)
```bash
python3 -m retro.scripts.playback_movie ./movie.bk2
```
