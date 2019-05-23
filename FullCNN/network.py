#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from connection import *
from layer import Layer
from datetime import datetime

class Network(object):
	def __init__(self, layers):
		'''
		初始化一个全连接神经网络
		layers: 二维数组，描述神经网络每层节点数
		'''
		#print '\n'
		self.connections = Connections()
		self.layers = []
		layer_count = len(layers)
		node_count = 0
        
		for i in range(layer_count):
			self.layers.append(Layer(i, layers[i])) # why layers[i] ??? it is transted to node_count

		for layer in range(layer_count - 1):
			connections = [Connection(upstream_node, downstream_node)
							for upstream_node in self.layers[layer].nodes
							for downstream_node in self.layers[layer + 1].nodes[:-1]] #[nodes:-1] for last ConstNode
			# ~ ~ 看不懂 ~ ~
			for conn in connections:
				self.connections.add_connection(conn)
				conn.downstream_node.append_upstream_connection(conn)
				conn.upstream_node.append_downstream_connection(conn)


	def train(self, labels, data_set, rate, iteration, log_file):
		'''
		训练神经网络
		labels: 数组，训练样本标签。每个元素是一个样本的标签。
		data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
		'''
		#log_file.write("network: train")
		for i in range(iteration):
			for d in range(len(data_set)):
				self.train_one_sample(labels[d], data_set[d], rate, log_file)

	def train_one_sample(self, label, sample, rate, log_file):
		'''
		内部函数，用一个样本训练网络
		'''
		#log_file.write("network: train_one_sample")
		self.predict(sample, log_file)
		self.calc_delta(label, log_file)
		self.update_weight(rate, log_file)
		#self.dump(log_file)

	def calc_delta(self, label, log_file):
		'''
		内部函数，计算每个节点的delta
		'''
		#反向传播算法，应该从最后一层反过来算
		#log_file.write("network: calc_delta")
		output_nodes = self.layers[-1].nodes #最后一层
		for i in range(len(label)):
			output_nodes[i].calc_output_layer_delta(label[i])

		for layer in self.layers[-2::-1]:
			for node in layer.nodes:
				node.calc_hidden_layer_delta()

	def update_weight(self, rate, log_file):
		'''
		内部函数，更新每个连接权重
		'''
		#log_file.write("network: update_weight") 
		for layer in self.layers[:-1]:
			for node in layer.nodes:
				for conn in node.downstream:
					conn.update_weight(rate)

	def calc_gradient(self, log_file):
		'''
		内部函数，计算每个连接的梯度
		'''
		#log_file.write("network: calc_gradient")
		for layer in self.layers[:-1]:
			for node in layer.nodes:
				for conn in node.downstream:
					conn.calc_gradient()

	def get_gradient(self, label, sample, log_file):
		'''
		获得网络在一个样本下，每个连接上的梯度
		label: 样本标签
		sample: 样本输入
		'''
		self.predict(sample, log_file)
		self.calc_delta(label, log_file)
		self.calc_gradient(log_file)

	def predict(self, sample, log_file):
		'''
		根据输入的样本预测输出值
		sample: 数组，样本的特征，也就是网络的输入向量
		'''
		#log_file.write("network: predict")
		self.layers[0].set_output(sample, log_file)
		for i in range(1, len(self.layers)):
			self.layers[i].calc_output()
		return map(lambda node: node.output, self.layers[-1].nodes[:-1])

	def dump(self, log_file):
		'''
		打印网络信息
		'''
		for layer in self.layers:
			layer.dump(log_file)
