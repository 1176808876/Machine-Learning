#!/usr/bin/env python
# -*- coding: UTF-8 -*-

#实现一个输出恒为1的节点（计算偏置项 w_b 时需要
class ConstNode(object):
	def __init__(self, layer_index, node_index):
		'''
		layer_index: 节点所属的层的编号
		node_index: 节点的编号
		'''
		self.layer_index = layer_index
		self.node_index = node_index
		self.downstream = []
		self.output = 1

	def append_downstream_connection(self, conn):
		'''
		添加一个到下游节点的连接
		'''
		self.downstream.append(conn)

	def calc_hidden_layer_delta(self):
		'''
		节点属于隐藏层时，根据式4 delta = a_i*(1-a_i)*reduce(w_ki*delta_k) 计算delta
		'''
		downstream_delta = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
		self.delta = self.output * (1 - self.output) * downstream_delta

	def log(self, log_file):
		node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
		downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
		log_file.write(node_str + '\n\tdownstream:' + downstream_str)

	def __str__(self):
		'''
		打印节点信息
		'''
		node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
		downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
		return node_str + '\n\tdownstream:' + downstream_str