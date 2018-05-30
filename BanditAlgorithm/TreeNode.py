# -*- coding: utf-8 -*-

class TreeNode(object):

    def __init__(self):
        """"
        This is a binary tree node
        """

        # left&right pointers
        self.left = None
        self.right = None

        # values required by HOO and the covering generator algorithm
        self.activated = False
        self.counter = 0
        self.B_value = None
        self.mean = 0
        self.U_value = None
