import tensorflow as tf

def build_network(non_spatial, screen, minimap, available_actions):
	non_spatial_fc = tf.contrib.layers.fully_connected(
		inputs=tf.contrib.layers.flatten(non_spatial),
		num_outputs=256,
		activation_fn=tf.tanh,
		scope='non_spatial_fc'
	)

	screen_conv1 = tf.contrib.layers.conv2d(
		inputs=tf.transpose(screen, [0, 2, 3, 1]),
		num_outputs=16,
		kernel_size=5,
		stride=1,
		scope='screen_conv1'
	)

	screen_conv2 = tf.contrib.layers.conv2d(
		inputs=screen_conv1,
		num_outputs=32,
		kernel_size=3,
		stride=1,
		scope='screen_conv2'
	)

	minimap_conv1 = tf.contrib.layers.conv2d(
		inputs=tf.transpose(minimap, [0, 2, 3, 1]),
		num_outputs=16,
		kernel_size=5,
		stride=1,
		scope='minimap_conv1'
	)

	minimap_conv2 = tf.contrib.layers.conv2d(
		inputs=minimap_conv1,
		num_outputs=32,
		kernel_size=3,
		stride=1,
		scope='minimap_conv2'
	)

	spatial_action = tf.contrib.layers.conv2d(
		inputs=tf.concat([screen_conv2, minimap_conv2], axis=3),
		num_outputs=1,
		kernel_size=1,
		stride=1,
		activation_fn=None,
		scope='spatial_action'
	)

	state_representation = tf.concat(
		[
		tf.contrib.layers.flatten(screen_conv2),
		tf.contrib.layers.flatten(minimap_conv2),
		non_spatial_fc
		],
		axis=1
	)
	state_representation_fc = tf.contrib.layers.fully_connected(
		inputs=state_representation,
		num_outputs=256,
		activation_fn=tf.nn.relu,
		scope='state_representation_fc'
	)

	spatial_action = tf.nn.softmax(tf.contrib.layers.flatten(spatial_action))
	non_spatial_action = tf.contrib.layers.fully_connected(
		inputs=state_representation_fc,
		num_outputs=available_actions,
		activation_fn=tf.nn.softmax,
		scope='non_spatial_action'
	)
	value = tf.reshape(
		tf.contrib.layers.fully_connected(
			inputs=state_representation_fc,
			num_outputs=1,
			activation_fn=None,
			scope='value'
		),
		[-1]
	)

	return non_spatial_action, spatial_action, value