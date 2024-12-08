import numpy as np
import random
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# 하이퍼 매개변수 설정
rho=0.9 
lamda=0.99 
eps=0.9
eps_decay=0.999
batch_siz=64
n_episode=100

# 신경망을 설계해주는 함수
def deep_network():
    mlp=Sequential()
    mlp.add(Dense(32,input_dim=env.observation_space.shape[0],activation='relu'))
    mlp.add(Dense(32,activation='relu'))
    mlp.add(Dense(env.action_space.n,activation='linear'))
    mlp.compile(loss='mse',optimizer='Adam')
    return mlp

# DQN 학습
def model_learning():
    mini_batch=np.asarray(random.sample(D,batch_siz))
    state=np.asarray([mini_batch[i]['state'] for i in range(batch_size)])
    action=np.asarray([mini_batch[i]['action'] for i in range(batch_size)])
    reward=np.asarray([mini_batch[i]['reward'] for i in range(batch_size)])
    state1=np.asarray([mini_batch[i]['newstate'] for i in range(batch_size)])
    done=np.asarray([mini_batch[i]['done'] for i in range(batch_size)])

    target=model.predict(state)
    target1=model.predict(state1)

    for i in range(batch_siz):
        if done[i]:
            target[i][action[i]]=reward[i]
        else:
            target[i][action[i]]+=rho*((reward[i]+lamda*np.amax(target1[i]))-target[i][action[i]])
            
    model.fit(state,target,batch_size=batch_siz,epochs=1,verbose=0)

env=gym.make("CartPole-v1")

model=deep_network() 
D=deque(maxlen=2000) # 리플레이 메모리 초기화
scores=[]
max_steps=env.spec.max_episode_steps

# 신경망 학습
for i in range(n_episode):
    s=env.reset()
    long_reward=0
    s=s[0]

    while True:
        r=np.random.random()
        eps=max(0.01,eps*eps_decay) 
        if(r<eps):
            a=np.random.randint(0,env.action_space.n) 
        else:
            q=model.predict(np.reshape(s,[1,4])) 
            a=np.argmax(q[0])
        s1,r,done,info,_=env.step(a)
        if done and long_reward<max_steps-1: 
            r=-100

        D.append({'state':np.array(s),
                  'action':np.array(a),
                  'reward':r,
                  'newstate':np.array(s1),
                  'done':done})

        if len(D)>batch_siz*3:
            model_learning()

        s=s1
        long_reward+=r
        
        if done:
            long_reward=long_reward if long_reward==max_steps else long_reward+100
            print(i,"번째 에피소드의 점수:",long_reward)
            scores.append(long_reward)
            break

    if i>10 and np.mean(scores[-5:])>(0.95*max_steps):
        break
    
# 신경망 저장
model.save("./cartpole_by_DQN.h5")
env.close()

import matplotlib.pyplot as plt

plt.plot(range(1,len(scores)+1),scores)
plt.title('DQN scores for CartPole-v0')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.grid()
plt.show()
