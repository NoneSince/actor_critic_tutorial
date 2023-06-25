def reinforce_with_baseline(env, policy, gamma=0.97, learning_rate=0.0005, value_learning_rate=0.0005, n_episodes=100, plotter=None):
    mean_eval_rewards = []
    for episode in range(n_episodes):
        # collect one episode:
        episode_rewards, actions_log_likelihood_grads, states = collect_episode(env, policy)

        # compute returns:
        episode_returns = compute_returns(episode_rewards, gamma)
        episode_returns = normalize_returns(episode_returns)

        # retrive parameters, and change them in the direction of the gradient:
        policy_parameters = policy.get_parameters_vector()
        value_parameters = policy.get_value_parameters_vector()
        for i in range(len(episode_returns)):
            ret = episode_returns[i]
            log_likelihood_grad = actions_log_likelihood_grads[i]
            state = states[i]

            delta = ret - policy.get_value(state)
            value_parameters = value_parameters + value_learning_rate * delta * policy.grad_value(state)
            policy_parameters = policy_parameters + learning_rate * pow(gamma,i) * delta * log_likelihood_grad
        # update the policy with the new parameters:
        policy.set_parameters_vector(policy_parameters)
        policy.set_value_parameters_vector(value_parameters)

        # for visualization only (not a part of the algorithm, and doesnt affect the parameters):
        if episode % 5 == 0 or episode==n_episodes-1:
            mean_reward = evaluate_agent(policy, env, n_episodes=5)
            mean_eval_rewards.append(mean_reward)
            if plotter is None:
                print(f"Episode {episode}: Evaluation mean accumulated reward = {mean_reward}")
            else:
                plotter.update_plot(episode, mean_reward)

    return mean_eval_rewards
