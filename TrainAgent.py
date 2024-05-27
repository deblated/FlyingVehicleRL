from stable_baselines3 import PPO
from gym.envs.registration import register

import gym

register(id='FlyingVehicle2D-v1', entry_point='FlyingVehicleEnv:FlyingVehicleEnv')
DroneEnv = gym.make('FlyingVehicle2D-v1')

model = PPO("MlpPolicy", DroneEnv, verbose=1)

model.learn(total_timesteps=2000000)
model.save('PPOTrainedAgent')





