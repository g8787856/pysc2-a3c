import numpy as np
from pysc2.lib import features

_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index

def screen():
	screen_channel = 0
	for i in range(len(features.SCREEN_FEATURES)):
		if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
			screen_channel += 1
		elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
			screen_channel += 1
		else:
			screen_channel += features.SCREEN_FEATURES[i].scale
	return screen_channel

def minimap():
	minimap_channel = 0
	for i in range(len(features.MINIMAP_FEATURES)):
		if i == _MINIMAP_PLAYER_ID:
			minimap_channel += 1
		elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
			minimap_channel += 1
		else:
			minimap_channel += features.MINIMAP_FEATURES[i].scale
	return minimap_channel

def preprocess_screen(screen):
	layers = []
	for i in range(len(features.SCREEN_FEATURES)):
		if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE or features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
			layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
		else:
			layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
			for j in range(features.SCREEN_FEATURES[i].scale):
				y, x = (screen[i] == j).nonzero()
				layer[j, y, x] = 1
			layers.append(layer)
	return np.concatenate(layers, axis=0)

def preprocess_minimap(minimap):
	layers = []
	for i in range(len(features.MINIMAP_FEATURES)):
		if i == _MINIMAP_PLAYER_ID or features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
			layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
		else:
			layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
			for j in range(features.MINIMAP_FEATURES[i].scale):
				y, x = (minimap[i] == j).nonzero()
				layer[j, y, x] = 1
			layers.append(layer)
	return np.concatenate(layers, axis=0)