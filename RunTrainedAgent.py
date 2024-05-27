from stable_baselines3 import PPO
from gym.envs.registration import register
import gym
import time

from FlyingVehicleEnv import FlyingVehicleEnv


register(id='FlyingVehicle2D-v1', entry_point='FlyingVehicleEnv:FlyingVehicleEnv')
droneEnv = gym.make('FlyingVehicle2D-v1', max_time_steps=500)

model = PPO.load("PPOTrainedAgent.zip")

random_seed = int(time.time())
model.set_random_seed(random_seed)

obs, _ = droneEnv.reset()
total_reward = 0
reward = None
terminated = False
truncated = False
done = False
info = None

while not done:
    droneEnv.render()

    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = droneEnv.step(action)
    done = terminated or truncated
    total_reward += reward

print("total reward: " + str(total_reward))

droneEnv.close()

