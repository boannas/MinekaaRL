from stable_baselines3 import DQN
from ursina_env_zigzag import UrsinaParkourEnv, EpisodeLogger
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

env = UrsinaParkourEnv()

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./checkpoints/",
    name_prefix="dqn_ursina_zigzag_07epr"
)

callback = CallbackList([checkpoint_callback, EpisodeLogger()])

model = DQN(
    "CnnPolicy",
    env,
    seed=42,
    verbose=0,
    policy_kwargs={"normalize_images": False},
    learning_rate=1e-4,
    gamma=0.95,
    buffer_size=10000,
    exploration_fraction=0.7,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    train_freq=100,
    target_update_interval=1000,
    tensorboard_log="./cnn_log/",
    device='cuda',
)

model.learn(
    total_timesteps=10000,
    callback=callback
)

model.save("dqn_timepenal_zigzag")
env.close()