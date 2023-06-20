from pg_tutorial.utils import plot_video
%matplotlib inline

# Define the training loop
for episode in range(num_episodes):
    # Initialize the environment
    state, _ = env.reset()
    done = False
    total_reward = 0
    images = []
    while not done:
        count+=1
        images.append(env.render())
        env.render()
        # Select an action using the agent's policy
        probs, val = agent(torch.tensor(state, dtype=torch.float32))
        action = np.random.choice(np.arange(len(probs)), p=probs.detach().numpy())
        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        #store the current state,action, and reward
        states.append(state)
        action.append(action)
        rewards.append(reward)
        next_stated.append(next_state)
        dones.append(1 if done else 0)
        # Set the state to the next state
        state = next_state
        # compute the average reward and update the network
        if(count>n):
            count=0
            for state,action, in range(n):
                _ , next_val = agent(torch.tensor(next_state, dtype=torch.float32))
                #we need to add here all the calculations and the network update

        #just to remember how to do it on the normal case, I'll delete it later
        _ , next_val = agent(torch.tensor(next_state, dtype=torch.float32))
        err = reward + discount_factor * (next_val * (1 - done)) - val
        actor_loss = -torch.log(probs[action]) * err
        critic_loss = torch.square(err)
        loss = actor_loss + critic_loss

        # Update the network


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    # Print the total reward for the episode
    print(f'Episode {episode}: Total reward = {total_reward}')
    plot_video(images)