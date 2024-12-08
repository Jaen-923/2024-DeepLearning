import gym
import numpy as np

env=gym.make('FrozenLake-v1',is_slippery=False) 
Q=np.zeros([env.observation_space.n,env.action_space.n]) 

rho=0.90 
lamda=0.99 
eps=1.0 
eps_decay=0.999 

n_episode=3000
length_episode=100

# 최적 행동 가치 함수 찾기
for i in range(n_episode):
    s, _ = env.reset() 
    for j in range(length_episode):
        r=np.random.random()
        eps=max(0.01,eps*eps_decay) 
        if(r<eps): 
            a=np.random.randint(0,env.action_space.n)
        else:
            if np.max(Q[s]) > 0 :
                a = np.argmax(Q[s])
            else:
                a = env.action_space.sample(
        s1, r, done, info, _=env.step(a)
        Q[s,a]=Q[s,a]+rho*(r+lamda*np.max(Q[s1,:])-Q[s,a])
        s=s1
        if done:
            break

np.set_printoptions(precision=2)
print(Q)
env.close()
