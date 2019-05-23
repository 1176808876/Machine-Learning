#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from node import Node
from constnode import ConstNode
from datetime import datetime

class Layer(object):
	def __init__(self, layer_index, node_count):
		'''
		初始化一层
		layer_index: 层编号
		node_count: 层所包含的节点数
		'''
		print '%s: Layer %u node_count %u' % (datetime.now().strftime("%c"), layer_index, node_count)

		self.layer_index = layer_index
		self.nodes = []
		for i in range(node_count):
			self.nodes.append(Node(layer_index, i))
		self.nodes.append(ConstNode(layer_index, node_count))

	def set_output(self, data, log_file):
		'''
		设置层的输出。 当层是输入层时会用到
		'''
		for i in range(len(data)):
			self.nodes[i].set_output(data[i])

	def calc_output(self):
		'''
		计算层的输出向量
		'''
		for node in self.nodes[:-1]: #最后一个是ConstNode, 不用算
			node.calc_output()

	def dump(self, log_file):
		for node in self.nodes:
			node.log(log_file)
			#print node