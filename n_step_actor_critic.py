
# TODO 3:
from collections import deque


def n_step_actor_critic(env, policy, n=5, learning_rate=0.0002, value_learning_rate=0.0002, gamma=0.97, n_episodes=10000, plotter=None):
    mean_eval_rewards = []
    states = deque(maxlen=n)
    actions = deque(maxlen=n)
    rewards = deque(maxlen=n)
    next_states = deque(maxlen=n)
    dones = deque(maxlen=n)
    for episode in range(n_episodes):
        state, _ = env.reset()
        I = 1
        count = 0
        value_parameters = policy.get_value_parameters_vector()
        policy_parameters = policy.get_parameters_vector()
        terminated, truncated = False, False
        while (not terminated) and (not truncated):
            count += 1
            action, log_likelihood_grad = policy.sample_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            # store the current state,action, and reward
            states.append(state)
            action.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1 if terminated else 0)

            if terminated:
                count -= 1
                TD_error = 0
                for i in range(count):
                    TD_error += rewards[i] * pow(gamma, i)
                TD_error += pow(gamma, n) * policy.get_value(states[n - 1]) - policy.get_value(states[0])
                # hint: exacly like the main n-step loop, but just teel count, not teel n.
                value_parameters = value_parameters + value_learning_rate * I * TD_error * policy.grad_value(state)
                policy.set_value_parameters_vector(value_parameters)
                policy_parameters = policy_parameters + learning_rate * I * TD_error * log_likelihood_grad
                policy.set_parameters_vector(policy_parameters)
                states.popleft()
                action.popleft()
                rewards.popleft()
                next_states.popleft()
            # n step loop
            # update just the first state!
            if (count == n):
                count -= 1
                # compute TD error by the formula
                TD_error = 0
                for i in range(n):
                    TD_error += rewards[i] * pow(gamma, i)
                TD_error += pow(gamma, n) * policy.get_value(states[n - 1]) - policy.get_value(states[0])

                # calculate the parameters like in the original function
                value_parameters = value_parameters + value_learning_rate * I * TD_error * policy.grad_value(state)
                policy.set_value_parameters_vector(value_parameters)
                policy_parameters = policy_parameters + learning_rate * I * TD_error * log_likelihood_grad
                policy.set_parameters_vector(policy_parameters)

                # delete the oldest state
                states.popleft()
                action.popleft()
                rewards.popleft()
                next_states.popleft()

            I = I * gamma
            state = next_state

        # evaluate:
        if episode % 5 == 0 or episode == n_episodes - 1:
            mean_reward = evaluate_agent(policy, env, n_episodes=5)
            mean_eval_rewards.append(mean_reward)
            if plotter is None:
                print(f"Episode {episode}: Evaluation mean accumulated reward = {mean_reward}")
            else:
                plotter.update_plot(episode, mean_reward)

    return mean_eval_rewards
