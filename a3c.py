import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions, features

from network import build_network
import utils

class A3CAgent():
	def __init__(self, discount, entropy_weight, value_loss_weight, resolution, training):
		self.name = 'a3c'
		self.discount = discount
		self.entropy_weight = entropy_weight
		self.value_loss_weight = value_loss_weight
		self.training = training
		self.summary = []
		self.resolution = resolution
		self.structured_dimensions = len(actions.FUNCTIONS)

	def reset(self):
		self.epsilon = [0.05, 0.2]

	def build_model(self, reuse, device):
		with tf.variable_scope(self.name) and tf.device(device):
			# tf.device:指定用於新創建的操作的默認設備。
			if reuse:
				tf.get_variable_scope().reuse_variables()

			"""
			達到重複利用變量的效果,
			要使用tf.variable_scope(),
			並搭配tf.get_variable() 這種方式產生和提取變量.
			不像tf.Variable() 每次都會產生新的變量,
			tf.get_variable () 如果遇到了同樣名字的變量時,
			它會單純的提取這個同樣名字的變量(避免產生新變量).
			而在重複使用的時候,
			一定要在代碼中強調scope.reuse_variables(),
			否則係統將會報錯, 以為你只是單純的不小心重複使用到了一個變量
			"""

			self.screen = tf.placeholder(tf.float32, [None, utils.screen(), self.resolution, self.resolution], name="screen")
			self.minimap = tf.placeholder(tf.float32, [None, utils.minimap(), self.resolution, self.resolution], name="minimap")
			self.structured = tf.placeholder(tf.float32, [None, self.structured_dimensions], name="structured")

			# build network
			network = build_network(self.structured, self.screen, self.minimap, len(actions.FUNCTIONS))
			self.non_spatial_action, self.spatial_action, self.value = network

			self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name="valid_non_spatial_action")
			self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)], name="non_spatial_action_selected")
			self.valid_spatial_action = tf.placeholder(tf.float32, [None], name="valid_spatial_action")
			self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.resolution ** 2], name="spatial_action_selected")
			self.target_value = tf.placeholder(tf.float32, [None], name="target_value")

			# compute log probability
			valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action, axis=1)
			# tf.reduce_sum:計算張量維度的元素總和。
			valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
			# tf.clip_by_value: 輸入一個張量A，把A中的每一個元素的值都壓縮在min和max之間。小於min的讓它等於min，大於max的元素的值等於max。
			non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
			non_spatial_action_prob /= valid_non_spatial_action_prob
			non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))

			spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
			spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))

			self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))
			self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))

			# compute loss !! 尚未加入entropy以及loss_weight
			action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
			advantage = tf.stop_gradient(self.target_value - self.value)
			policy_loss = -tf.reduce_mean(action_log_prob * advantage)
			# value_loss = -tf.reduce_mean(self.value * advantage)
			value_loss = tf.reduce_mean(tf.square(self.target_value - self.value) / 2.)
			# entropy = -tf.reduce_mean(self.non_spatial_action * tf.log(self.non_spatial_action))
			entropy = -tf.reduce_mean(valid_non_spatial_action_prob * tf.log(valid_non_spatial_action_prob))
			loss = policy_loss + value_loss * self.value_loss_weight - entropy * self.entropy_weight

			self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
			self.summary.append(tf.summary.scalar('value_loss', value_loss))
			self.summary.append(tf.summary.scalar('value', tf.reduce_mean(self.value)))
			self.summary.append(tf.summary.scalar('advantage', tf.reduce_mean(advantage)))
			self.summary.append(tf.summary.scalar('returns', tf.reduce_mean(self.target_value)))

			# optimizer
			self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
			optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
			grads = optimizer.compute_gradients(loss)
			clipped_grads = []
			for grad, var in grads:
				self.summary.append(tf.summary.histogram(var.op.name, var))
				self.summary.append(tf.summary.histogram(var.op.name + '/grad', grad))
				grad = tf.clip_by_norm(grad, 10.0)
				clipped_grads.append([grad, var])
			self.train_op = optimizer.apply_gradients(clipped_grads)
			self.summary_op = tf.summary.merge(self.summary)

			self.saver = tf.train.Saver()

	def setup(self, sess, summary_writer):
		self.sess = sess
		self.summary_writer = summary_writer

	def initialize(self):
		init_op = tf.global_variables_initializer()
		# tf.global_variables_initializer: 將所有全局變量的初始化器匯總，並對其進行初始化。
		self.sess.run(init_op)

	def step(self, obs):
		screen = np.array(obs.observation.feature_screen, dtype=np.float32)
		screen = np.expand_dims(utils.preprocess_screen(screen), axis=0)
		# np.expand_dims: 展開數組的形狀。
		# x = np.array([1,2])
		# x.shape
		# (2,)
		# y = np.expand_dims(x, axis=0)
		# y
		# array([[1, 2]])
		# y.shape
		# (1, 2)
		minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
		minimap = np.expand_dims(utils.preprocess_minimap(minimap), axis=0)
		structured = np.zeros([1, self.structured_dimensions], dtype=np.float32)
		structured[0, obs.observation.available_actions] = 1

		feed_dict = {
			self.screen: screen,
			self.minimap: minimap,
			self.structured: structured
		}
		
		non_spatial_action, spatial_action = self.sess.run(
			[self.non_spatial_action, self.spatial_action],
			feed_dict=feed_dict
		)

		non_spatial_action = non_spatial_action.ravel()
		spatial_action = spatial_action.ravel()
		# np.ravel: 返回一個連續的扁平數組(flatten array)。
		# x = np.array([[1, 2, 3], [4, 5, 6]])
		# print(np.ravel(x))
		# [1 2 3 4 5 6]
		available_actions = obs.observation.available_actions
		# action_id = available_actions[np.argmax(non_spatial_action[available_actions])]
		non_spatial_action = np.array(non_spatial_action[available_actions])
		non_spatial_action /= non_spatial_action.sum()
		action_id = available_actions[np.where(non_spatial_action == np.random.choice(non_spatial_action, p=non_spatial_action))[0][0]]

		# spatial_target = np.where(spatial_action == np.random.choice(spatial_action, p=spatial_action))[0][0]
		# spatial_target = [int(spatial_target // self.resolution), int(spatial_target % self.resolution)]
		spatial_target = np.argmax(spatial_action)
		spatial_target = [int(spatial_target // self.resolution), int(spatial_target % self.resolution)]

		# epsilon-greedy exploration
		# if self.training and np.random.rand() < self.epsilon[0]:
		# 	action_id = np.random.choice(available_actions)
		# if self.training and np.random.rand() < self.epsilon[1]:
		# 	delta_y, delta_x = np.random.randint(-4, 5), np.random.randint(-4, 5)
		# 	spatial_target[0] = int(max(0, min(self.resolution -1, spatial_target[0] + delta_y)))
		# 	spatial_target[1] = int(max(0, min(self.resolution -1, spatial_target[1] + delta_x)))

		action_args = []
		for arg in actions.FUNCTIONS[action_id].args:
			if arg.name in ('screen', 'minimap', 'screen2'):
				action_args.append([spatial_target[1], spatial_target[0]])
			else:
				action_args.append([0])
		return actions.FunctionCall(action_id, action_args)

	def update(self, replay_buffer, learning_rate, step):
		obs = replay_buffer[-1][-1]
		if obs.last():
			reward = 0
		else:
			screen = np.array(obs.observation.feature_screen, dtype=np.float32)
			screen = np.expand_dims(utils.preprocess_screen(screen), axis=0)
			minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
			minimap = np.expand_dims(utils.preprocess_minimap(minimap), axis=0)
			structured = np.zeros([1, self.structured_dimensions], dtype=np.float32)
			structured[0, obs.observation.available_actions] = 1

			feed_dict = {
				self.screen: screen,
				self.minimap: minimap,
				self.structured: structured
			}
			reward = self.sess.run(self.value, feed_dict=feed_dict)

		#compute targets and masks
		screens, minimaps, structureds = [], [], []
		target_value = np.zeros([len(replay_buffer)], dtype=np.float32)
		target_value[-1] = reward

		valid_non_spatial_action = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.float32)
		non_spatial_action_selected = np.zeros([len(replay_buffer), len(actions.FUNCTIONS)], dtype=np.float32)
		valid_spatial_action = np.zeros([len(replay_buffer)], dtype=np.float32)
		spatial_action_selected = np.zeros([len(replay_buffer), self.resolution ** 2], dtype=np.float32)

		record_score = replay_buffer[-1][0].observation['score_cumulative'][0]
		summary = tf.Summary()
		summary.value.add(tag='episode_score', simple_value=record_score)
		print('train!! step %d: score = %f' % (step, record_score))
		self.summary_writer.add_summary(summary, step)

		replay_buffer.reverse()
		# reverse:方法沒有返回值，但是會對列表的元素進行反向排序
		for i, [obs, action, next_obs] in enumerate(replay_buffer):
		# seq = ['one', 'two', 'three']
		# for i, element in enumerate(seq):
		# print i, element
		#	0 one
		#	1 two
		#	2 three
			screen = np.array(obs.observation.feature_screen, dtype=np.float32)
			screen = np.expand_dims(utils.preprocess_screen(screen), axis=0)
			minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
			minimap = np.expand_dims(utils.preprocess_minimap(minimap), axis=0)
			structured = np.zeros([1, self.structured_dimensions], dtype=np.float32)
			structured[0, obs.observation.available_actions] = 1

			screens.append(screen)
			minimaps.append(minimap)
			structureds.append(structured)

			reward = obs.reward
			action_id = action.function
			action_args = action.arguments

			target_value[i] = reward + self.discount * target_value[i - 1]

			available_actions = obs.observation.available_actions
			valid_non_spatial_action[i, available_actions] = 1
			non_spatial_action_selected[i, action_id] = 1

			args = actions.FUNCTIONS[action_id].args
			for arg, action_arg in zip(args, action_args):
				if arg.name in ('screen', 'minimap', 'screen2'):
					spatial_action = action_arg[1] * self.resolution + action_arg[0]
					valid_spatial_action[i] = 1
					spatial_action_selected[i, spatial_action] = 1

		screens = np.concatenate(screens, axis=0)
		minimaps = np.concatenate(minimaps, axis=0)
		structureds = np.concatenate(structureds, axis=0)

		feed_dict = {
			self.screen: screens,
			self.minimap: minimaps,
			self.structured: structureds,
			self.target_value: target_value,
			self.valid_non_spatial_action: valid_non_spatial_action,
			self.non_spatial_action_selected: non_spatial_action_selected,
			self.valid_spatial_action: valid_spatial_action,
			self.spatial_action_selected: spatial_action_selected,
			self.learning_rate: learning_rate
		}
		_, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed_dict)
		self.summary_writer.add_summary(summary, step)

	def save(self, path, step):
		os.makedirs(path, exist_ok=True)
		self.summary_writer.flush()
		self.saver.save(self.sess, os.path.join(path, 'model.ckpt'), global_step=step)
		print('Saving agent to %s, step %d' % (path, step))

	def load(self, path):
		checkpoint_path = tf.train.get_checkpoint_state(path)
		self.saver.restore(self.sess, checkpoint_path.model_checkpoint_path)
		print('Loaded agent!!!')
		return int(checkpoint_path.model_checkpoint_path.split('-')[-1])


