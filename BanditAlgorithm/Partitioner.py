# -*- coding: utf-8 -*-

import TreeNode
import random
import math

"""
    This class is built around the ideas that are thoroughly explained in the paper
    "X-armed Bandits" by Bubeck et al., 2011.
"""


class HOO(object):
    """
    The hierarchical optimistic optimization algorithm.
    """

    def __init__(self, v1, ro, covering_generator_function):
        """
        :param v1: v1 must be > 0. the diameter of all tree coverings must less than v1*ro^h where h
        is the depth of the node with given covering.
        :param ro: ro must be in (0,1) exclusively. the diameter of all tree coverings must less than v1*ro^h where h
        is the depth of the node with given covering.
        :param covering_generator_function: this function generates a subspace of space X,
        by taking the height(h) and the order-in-the-level(i) of a binary tree node. (For instance a root node
        has h = 0 and i = 1, its children would have h = 1, and i = 1 and i = 2 from left to right.) This func. must
        return an object with two properties, namely "upper" & "lower" which define the boundaries of the subset.
        """
        # ro needs to be between 0 and 1 exclusively.
        if 0 >= ro:
            ro = 0.001
        elif ro >= 1:
            ro = 0.999

        self.ro = ro
        self.v1 = v1
        self.tree_coverings = covering_generator_function
        self.tree = TreeNode.TreeNode() # this is the root
        self.create_children_of_node(self.tree) # create first two leafs with infinite B values
        self.last_arm = None

    def set_time_horizon(self, max_plays):
        """
        :param max_plays: maximum number of rounds.
        """
        self.max_plays = max_plays

    def set_environment(self, environment_function):
        """
        :param environment_function: environment function must take a vector as a parameter that is of size that is
        specified to covering_generator_function, and return a scalar as reward(positive) and loss(negative).
        """
        self.environment_function = environment_function

    def create_children_of_node(self, node):
        """
        :param node: parent node.
        """
        node.activated = True

        node.left = TreeNode.TreeNode()
        node.left.B_value = float("inf")

        node.right = TreeNode.TreeNode()
        node.right.B_value = float("inf")

    def selection_of_arm_from_covering(self, h, i):
        """
        :param h: chosen node's height.
        :param i: chosen node's order in the level.
        :return: received reward.
        """
        subset_space = self.tree_coverings(h, i)
        # now we need to arbitrarily select one arm from this subset of the space.
        # the chosen strategy is selecting the midpoint of the subset but this can
        # be further modified.
        selected_arm = (1.0*subset_space.upper + subset_space.lower) / 2.0
        #print("selected arm = {0}".format(selected_arm))
        self.last_arm = selected_arm
        reward = self.environment_function(selected_arm)
        return reward

    def recursively_update_tree_U_values(self, node, round, height):
        """
        :param node: current node.
        :param round: round t.
        :param height: current height.
        :return: None
        """
        if not node.activated:
            return

        node.U_value = node.mean + math.sqrt( 2.0* math.log(round) / node.counter) + self.v1*(self.ro**height)
        self.recursively_update_tree_U_values(node.left, round, height+1)
        self.recursively_update_tree_U_values(node.right, round, height+1)
        return

    def recursively_copy_active_nodes(self, node, copy_list):
        """
        :param node: current node.
        :param copy_list: the list where all nodes are copied to via a post-order traversal.
        """
        # we do a post-order traversal because leafs should be added first to the list
        if not node.activated:
            return
        self.recursively_copy_active_nodes(node.left, copy_list)
        self.recursively_copy_active_nodes(node.right, copy_list)
        copy_list.append(node)

    def run_hoo(self):
        """
        run the agent for "max_plays" rounds.
        """
        for round in range(1, self.max_plays):

            traversed_path = []
            current_node_depth = 0;
            current_node_in_level = 1;
            current_node = self.tree
            traversed_path.append(current_node)

            # traverse down the tree to find the leaf with highest B value.
            while current_node.activated:

                if current_node.left.B_value > current_node.right.B_value:
                    current_node = current_node.left
                    current_node_in_level = current_node_in_level * 2

                elif current_node.left.B_value < current_node.right.B_value:
                    current_node = current_node.right
                    current_node_in_level = current_node_in_level * 2 - 1

                else:
                    # tie breaking rule here is defined as choosing a child at random
                    rand = random.uniform(0,1)

                    if rand > 0.5:
                        current_node = current_node.left
                        current_node_in_level = current_node_in_level * 2

                    else:
                        current_node = current_node.right
                        current_node_in_level = current_node_in_level*2 - 1

                # append to traversed path and update height as well as order in level
                current_node_depth += 1
                traversed_path.append(current_node)

            # now we have selected the most promising child from the tree we had,
            # we can draw an arm in X from the coverings
            reward = self.selection_of_arm_from_covering(current_node_depth, current_node_in_level)
            self.create_children_of_node(current_node)

            # update counter values and means in the traversed path
            for node in traversed_path:
                node.counter += 1
                node.mean = (1.0 - 1.0/node.counter) * node.mean + reward*1.0/node.counter

            self.recursively_update_tree_U_values(node=self.tree, round=round, height=0)
            tree_copy = []
            self.recursively_copy_active_nodes(node=self.tree, copy_list=tree_copy)

            # backward computation to update all the B_values.
            while tree_copy.__len__() > 1:
                node = tree_copy.pop(0)
                node.B_value = min(node.U_value, max(node.left.B_value, node.right.B_value))

            if round % 100 == 0:
                print ("@round {0}".format(round))

        print("done!")
        return
