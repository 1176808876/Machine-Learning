#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import math

def sigmoid(inx):
	if inx < 0:
		return 1 - 1 / (1 + math.exp(inx))
	return 1.0 / (1 + math.exp(-inx))

#节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算
class Node(object):
	def __init__(self, layer_index, node_index):
		'''
		layer_index: 节点所属的层的编号
		node_index: 节点的编号
		'''
		self.layer_index = layer_index
		self.node_index = node_index
		self.downstream = []
		self.upstream = []
		self.output = 0
		self.delta = 0

	def set_output(self, output):
		'''
		设置节点的输出值。如果节点属于输入层会用到这个函数。
		'''
		self.output = output

	def append_downstream_connection(self, conn):
		'''
		添加一个到下游节点的连接
		'''
		self.downstream.append(conn)


	def append_upstream_connection(self, conn):
		'''
		添加一个到上游节点的连接
		'''
		self.upstream.append(conn)


	def calc_output(self):
		'''
		根据式 y=sigmoid(x) 计算节点的输出
		'''
		output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0.0)
		self.output = sigmoid(output)

	def calc_hidden_layer_delta(self):
		'''
		节点属于隐藏层时，根据式(4) delta = a_i*(1-a_i)*reduce(w_ki*delta_k) 计算delta
		'''
		downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0)
		self.delta = self.output * (1 - self.output) * downstream_delta

	def calc_output_layer_delta(self, label):
		'''
		节点属于输出层时，根据式(3) delta = y_i(1-y_i)(t_i-y_i) 计算delta
		'''
		self.delta = self.output * (1 - self.output) * (label - self.output)

	def log(self, log_file):
		'''
		打印节点信息到日志
		'''
		node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
		downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
		upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
		log_file.write(node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str)

	def __str__(self):
		'''
		打印节点信息
		'''
		node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
		downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
		upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
		return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

