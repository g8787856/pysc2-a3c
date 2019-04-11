import time

def rollout(agents, env, max_frames):
	start_time = time.time()

	try:
		while True:
			episode_length = 0
			timesteps = env.reset()
			for agent in agents:
				agent.reset()

			while True:
				episode_length += 1
				last_timesteps = timesteps
				actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
				timesteps = env.step(actions)
				done = (episode_length > max_frames and max_frames != None) or timesteps[0].last()

				yield [last_timesteps[0], actions[0], timesteps[0]], done

				if done:
					break
	except KeyboardInterrupt:
		pass
	finally:
		total_time = time.time() - start_time
		print('took %.3f seconds' % total_time)