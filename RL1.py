import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# --- Training Phase ---

env = gym.make("MountainCar-v0")  # No rendering during training

new_state, _ = env.reset()#reset() returns enironment state and environment specific information
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) #creates [20,20] two observations for Mountain Car position and velocity
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE#gives the interval size per observation

epsilon = 0.9
EPISODES = 30000
START_EPSILON_DECAYING = 1
END_EPSILON_DECAY = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAY - START_EPSILON_DECAYING)
MAX_STEPS = 200
LEARNING_RATE = 0.1
DISCOUNT = 0.95
SHOW_EVERY = 200

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))#creates a whole action and observation map
goal_reached_at = None
ep_rewards=[]
aggr_ep_rewards={'ep':[],'avg':[],'min':[],'max':[]}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):
    episode_reward=0
    if episode % SHOW_EVERY == 0:
        print(f"Episode: {episode}")

    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    step = 0

    while not done and step < MAX_STEPS:
        # Epsilon-greedy action selection
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward+=reward
        new_discrete_state = get_discrete_state(new_state)
        step += 1

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] >= env.unwrapped.goal_position:
            q_table[discrete_state + (action,)] = 0
            if goal_reached_at is None:
                goal_reached_at = episode

        discrete_state = new_discrete_state
        ep_rewards.append(episode_reward)


    if END_EPSILON_DECAY >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    ep_rewards.append(episode_reward)
    if not episode%SHOW_EVERY:
       
       average_reward=sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
       aggr_ep_rewards['ep'].append(episode)
       aggr_ep_rewards['avg'].append(average_reward)
       aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
       aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
       print(f"Episode:{episode} Avg:{average_reward} Min:{min(ep_rewards[-SHOW_EVERY:])} Max:{max(ep_rewards[-SHOW_EVERY:])}")


env.close()
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label="avg")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label="min")
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label="max")
plt.legend(loc=4)
plt.show()
# --- Final Remark ---
if goal_reached_at is not None:
    print(f"\n Goal was first reached at Episode {goal_reached_at}!")
else:
    print("\n Goal was not reached during training.")


print("\n Running final demo using trained Q-table...")

env = gym.make("MountainCar-v0", render_mode="human")
state, _ = env.reset()
discrete_state = get_discrete_state(state)
done = False
steps = 0

while not done and steps < MAX_STEPS:
    action = np.argmax(q_table[discrete_state])#use generated q table
    new_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    discrete_state = get_discrete_state(new_state)
    steps += 1

env.close()
