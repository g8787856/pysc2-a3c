import sys #系統相關的參數以及函式
import threading #執行多執行續使用
import os
import time
from absl import app, flags
import tensorflow as tf
from pysc2.env import sc2_env

from a3c import A3CAgent
from rollout import rollout

LOCK = threading.Lock()

STEP = 0
FLAGS = flags.FLAGS

flags.DEFINE_string('map', 'MoveToBeacon', 'name of game map')
flags.DEFINE_integer('envs', 8, 'number of environments')
flags.DEFINE_integer('resolution', 32, 'resolution for screen and minimap feature layers')
flags.DEFINE_integer('step_mul', 8, 'game steps per agent step')
flags.DEFINE_bool('render', False, 'if true, render the game')
flags.DEFINE_bool('train', True, 'if true, train a model')
flags.DEFINE_integer('steps_per_batch', 2000, 'number of game batch to train')
flags.DEFINE_float('max_steps', 1e4*5, 'total steps for training')
flags.DEFINE_integer('save_interval', 50, 'number of  steps of saving')
flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')
flags.DEFINE_float('discount', 0.99, 'reward discount factor')
flags.DEFINE_float('entropy_weight', 1e-2, 'weight of entropy loss')
flags.DEFINE_float('value_weight', 0.5, 'weight of value function loss')
flags.DEFINE_bool('has_gpu', True, 'if true, use gpu to run')
flags.DEFINE_string('gpus', "0", 'number of cpus used for training')
flags.DEFINE_string('save_dir', os.path.join('model'), 'model storage')
flags.DEFINE_string('summary_dir', os.path.join('summary'), 'summary storage')
flags.DEFINE_string('model_name', 'my_agent', 'model name')
flags.DEFINE_string('exploration_mode', 'original_ac3', 'if original_ac3, use original a3c mode, else use epsilon-greedy exploration')
flags.DEFINE_bool('set_learning_rate_decay_threshold', True, 'use learning rate decay threshold to accelerate learning efficiency')
flags.DEFINE_integer('reward_cumulate', 20, 'number of reward cumulate')
flags.DEFINE_float('learning_rate_decay_threshold', 0.3, 'reward discount factor')

FLAGS(sys.argv)

if FLAGS.has_gpu:
	ENVS = FLAGS.envs
	DEVICE = ['/gpu:' + dev for dev in FLAGS.gpus]
else:
	ENVS = 1
	DEVICE = ['/cpu:0']

STEPS_PER_BATCH = FLAGS.steps_per_batch
learning_rate = FLAGS.learning_rate

checkpoint_path = os.path.join(FLAGS.save_dir, FLAGS.map)
if FLAGS.train:
	summary_path = os.path.join(FLAGS.summary_dir, FLAGS.map)
else:
	summary_path = os.path.join(FLAGS.summary_dir, 'not_train', FLAGS.map)

def run_thread(agent, visualize, summary_writer, learning_rate):
	with sc2_env.SC2Env(
		map_name=FLAGS.map,
		game_steps_per_episode=0,
		agent_interface_format=[
			sc2_env.AgentInterfaceFormat(
				feature_dimensions=sc2_env.Dimensions(
					screen=(FLAGS.resolution, FLAGS.resolution),
					minimap=(FLAGS.resolution, FLAGS.resolution)
				),
				use_feature_units=True
			)
		],
		step_mul=FLAGS.step_mul,
		visualize=visualize
	) as env:

		replay_buffer = []
		score = 0
		cumulative_score = 0
		count = 0
		max_score = 0
		pre_score = 0
		now_score = 0
		for trajectory, done in rollout([agent], env, STEPS_PER_BATCH):
			obs = trajectory[-1].observation
			score = obs['score_cumulative'][0]
			if FLAGS.train:
				replay_buffer.append(trajectory)
				if done:
					step = 0
					with LOCK:
						global STEP
						STEP += 1
						step = STEP
					if FLAGS.set_learning_rate_decay_threshold:
						if (step + 1) <= FLAGS.reward_cumulate:
							pre_score += obs['score_cumulative'][0]
						elif (step + 1) > FLAGS.reward_cumulate:
							now_score += obs['score_cumulative'][0]
						
						if (step + 1) >= FLAGS.reward_cumulate * 2 and (step + 1) % FLAGS.reward_cumulate == 0 and pre_avg_score != 0 and this_avg_score != 0:
							this_avg_score /= FLAGS.reward_cumulate
							pre_avg_score /= FLAGS.reward_cumulate
							print('this_avg_score ', this_avg_score)
							print('pre_avg_score ', pre_avg_score)
							if (this_avg_score - pre_avg_score) / pre_avg_score >= FLAGS.learning_rate_decay_threshold:
								print('update learning rate!!!!!!!!!!!!!!!!!')
								learning_rate = FLAGS.learning_rate * (1 - 0.9 * step / FLAGS.max_steps)
							pre_score = this_avg_score * FLAGS.reward_cumulate
							now_score = 0
							print('now learning_rate ', learning_rate)
					else:
						learning_rate = FLAGS.learning_rate * (1 - 0.9 * step / FLAGS.max_steps)
					agent.update(replay_buffer, learning_rate, step)
					replay_buffer = []
					if (step + 1) % FLAGS.save_interval == 0:
						agent.save(checkpoint_path, step)
					if step >= FLAGS.max_steps:
						break
			elif done:
				count += 1
				obs = trajectory[-1].observation
				score = obs['score_cumulative'][0]
				if score >= max:
					max = score
				cumulative_score += obs['score_cumulative'][0]
				print('avg score: ', cumulative_score/count)
				print('episode score: ', score)
				print('for now, the max score: ', max)
				summary = tf.Summary()
				summary.value.add(tag='avg_score', simple_value=(cumulative_score/count))
				summary.value.add(tag='episode_score', simple_value=score)
				summary_writer.add_summary(summary, count)
				if count == 1000:
					break

def main(argv):
	agents = []
	for i in range(ENVS):
		agent = A3CAgent(FLAGS.discount, FLAGS.entropy_weight, FLAGS.value_weight, FLAGS.resolution, FLAGS.train, FLAGS.exploration_mode)
		agent.build_model(i > 0, DEVICE[i % len(DEVICE)])
		agents.append(agent)

	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	# allow_soft_placement=True，允許tf自動選擇一個存在並且可用的設備來運行操作。
	sess = tf.Session(config=config)

	summary_writer = tf.summary.FileWriter(summary_path)
	# tf.summary.FileWriter():將視覺化檔案輸出。
	for i in range(ENVS):
		agents[i].setup(sess, summary_writer)

	agent.initialize()
	if os.path.exists(checkpoint_path):
		global STEP
		STEP = agent.load(checkpoint_path)
			
	threads = []
	for i in range(ENVS - 1):
		t = threading.Thread(target=run_thread, args=(agents[i], False, summary_writer, learning_rate))
		threads.append(t)
		t.daemon = True
		t.start()
		time.sleep(5)

	run_thread(agents[-1], FLAGS.render, summary_writer, learning_rate)

	for t in threads:
		t.join()

if __name__ == '__main__':
	app.run(main)
	# main()