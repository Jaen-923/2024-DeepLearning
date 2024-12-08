import gym

# 환경 불러오기
env=gym.make("FrozenLake-v1",is_slippery=False, map_name="8x8", render_mode="ansi")
print(env.observation_space)
print(env.action_space)

n_trial=20

# 에피소드 수집
env.reset()
episode=[]
for i in range(n_trial):
    action=env.action_space.sample() 
    obs,reward,done,info, _ =env.step(action) 
    episode.append([action,reward,obs])
    env.render()
    if done:
        break

print(episode)
env.close()
