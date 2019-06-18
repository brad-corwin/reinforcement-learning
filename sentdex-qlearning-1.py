import gym
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
# weight, measure of how important we find future rewardss over current reward
DISCOUNT = 0.95
EPISODES = 2000

SHOW_EVERY = 500

env = gym.make('MountainCar-v0')
#env.reset()

#print(env.observation_space.high)
#print(env.observation_space.low)
#print(env.action_space.n)

# separate observation space into 20 buckets for values
DISCRETE_OBSERVATION_SPACE_SIZE = [20] * len(env.observation_space.high)
discrete_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OBSERVATION_SPACE_SIZE

# explore vs exploit
epsilon = 0.5
START_EPSILON_DECAYING = 1
# // divide out to integer 
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# 20 x 20 x 3
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OBSERVATION_SPACE_SIZE + [env.action_space.n]))

# list to contain each episodes rewards
ep_rewards = []
agg_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

# convert continuous states to discrete states
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_size
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        #print(episode)
        #np.save(f"qtables/{episode}-qtable.npy", q_table)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    #print(env.reset())
    #print(discrete_state)
    #print(q_table[discrete_state])
    #print(np.argmax(q_table[discrete_state]))

    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        # state here is position and velocity, but to a model free agent, it doesn't matter
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            # get the q value for the action we took
            current_q = q_table[discrete_state + (action, )]
            # Bellman equation
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.unwrapped.goal_position:
            #print(f"We made it on episode {episode}")
            # reward for completing
            q_table[discrete_state + (action, )] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        agg_ep_rewards['ep'].append(episode)
        agg_ep_rewards['avg'].append(average_reward)
        agg_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        agg_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label="avg")
plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label="min")
plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label="max")
plt.legend(loc='best')
plt.show()

# for plotting actions and making a video over time go to
# https://pythonprogramming.net/q-learning-analysis-reinforcement-learning-python-tutorial/