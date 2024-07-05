"""
Script for humans to play an epsiode.
"""

import argparse
import ctypes
import pyglet
from pyglet.gl import *
import retro

class RetroGameWindow(pyglet.window.Window):
    def __init__(self, game, scenario, record):
        self.env = retro.make(game=game, scenario=scenario, record=record, render_mode=None, state=retro.State.DEFAULT, use_restricted_actions=retro.Actions.ALL)
        self.obs, info = self.env.reset()
        screen_height, screen_width = self.obs.shape[:2]

        super(RetroGameWindow, self).__init__(width=2000, height=int(2000 * screen_height / screen_width), vsync=True)
        self.keys = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.keys)

        self.setup_gl(screen_width, screen_height)
        
        self.cumulated_reward = 0
        self.steps = 0

        pyglet.clock.schedule_interval(self.update, 1 / 60.0)  # Schedule update at 60Hz

    def setup_gl(self, width, height):
        glEnable(GL_TEXTURE_2D)
        self.texture_id = GLuint(0)
        glGenTextures(1, ctypes.byref(self.texture_id))
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

    def on_draw(self):
        self.clear()
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        video_buffer = ctypes.cast(self.obs.tobytes(), ctypes.POINTER(ctypes.c_short))
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.obs.shape[1], self.obs.shape[0], GL_RGB, GL_UNSIGNED_BYTE, video_buffer)
        x, y, h, w = 0, 0, self.height, self.width
        pyglet.graphics.draw(4, GL_QUADS, ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]), ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]))

    def update(self, dt):
        self.dispatch_events()

        keys_pressed = {k for k, v in self.keys.items() if v}
        inputs = {
            'A': pyglet.window.key.Z in keys_pressed,
            'B': pyglet.window.key.X in keys_pressed,
            'C': pyglet.window.key.C in keys_pressed,
            'X': pyglet.window.key.A in keys_pressed,
            'Y': pyglet.window.key.S in keys_pressed,
            'Z': pyglet.window.key.D in keys_pressed,
            'UP': pyglet.window.key.UP in keys_pressed,
            'DOWN': pyglet.window.key.DOWN in keys_pressed,
            'LEFT': pyglet.window.key.LEFT in keys_pressed,
            'RIGHT': pyglet.window.key.RIGHT in keys_pressed,
            'MODE': pyglet.window.key.TAB in keys_pressed,
            'START': pyglet.window.key.ENTER in keys_pressed,
        }
        action = [inputs[b] for b in self.env.buttons]
        self.obs, rew, done, truncated, info = self.env.step(action)
        self.cumulated_reward += rew
        self.steps += 1
        print(rew)
        if done or truncated:
            print("Cumulated reward:", self.cumulated_reward)
            pyglet.app.exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', help='retro game to use', default='SonicTheHedgehog2-Genesis')
    parser.add_argument('--scenario', help='scenario to use', default='./scenario.json')
    parser.add_argument('--record', type=str, help='record to a .bk2 file', default=None)
    args = parser.parse_args()

    window = RetroGameWindow(args.game, args.scenario, args.record)
    pyglet.app.run()

if __name__ == "__main__":
    main()

