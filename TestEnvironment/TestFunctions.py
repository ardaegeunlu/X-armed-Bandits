# -*- coding: utf-8 -*-
import numpy as np
import random
import math

class TestFunctions(object):

    def __init__(self, functionName, **kwargs):
        """
        :param functionName: choose one of the following:
        1. "analytical_g", n-Dimensional function where n >= 1.
        2. "SixHumpCamelback", 2-Dimensional function.
        3. "hyper_ellipsoid", n-Dimensional function where n >= 1.

        :param kwargs: supply with the following conditions:
        - if chosen 'hyper_ellipsoid', provide 'dimensions' = dimensions in generically measurable space-X.
        - if chosen 'analytical_g', provide 'g_params' = [a_1, a_2, ..., a_n] where ai >= 0. there would be n dimensions.
        """
        self.function = functionName
        self.bests = []  # the values drawn by the best-fixed arm in hindsight.
        self.drawn_values = []  # the values drawn by the agent.
        self.parameters_for_g_func = kwargs.get('g_params')
        self.dimensions = kwargs.get('dimensions')
        self.best_analytical_g = 0

        if functionName == "analytical_g":
            self.best_analytical_g = self.calculate_best_of_analytical_g()

    def draw_value(self, input):
        """
        :param input: input vector.
        :return: reward.
        """
        if self.function == "SixHumpCamelback":
            return self.draw_sixhump_camelback(input_array=input)
        elif self.function == "analytical_g":
            return self.draw_analytical_g(input_array=input)
        elif self.function == "hyper_ellipsoid":
            return self.draw_hyper_ellipsoid(input_array=input)

    def get_bounds(self):
        """
        :return: max-min bounds of the domain of the functions.
        """
        if self.function == "SixHumpCamelback":
            return self.get_sixhump_camelback_bounds()
        elif self.function == "analytical_g":
            return self.get_analytical_g_bounds()
        elif self.function == "hyper_ellipsoid":
            return self.get_hyper_ellipsoid_bounds()

    def get_sixhump_camelback_bounds(self):
        return (np.array([-2,-1]), np.array([2,1]))

    def get_analytical_g_bounds(self):
        size = np.size(self.parameters_for_g_func)
        return (np.zeros(size)-2, np.ones(size)+1)

    def get_hyper_ellipsoid_bounds(self):
        size = self.dimensions
        return (np.zeros(size)-size*3, np.zeros(size)+size*3)

    def draw_analytical_g(self, input_array):
        """
        :param input_array: x.
        :return: result of the analytical_g function (a.k.a sobol function).
        """
        g_funcs = (np.absolute(4*input_array-2) + self.parameters_for_g_func) / (1.0 + self.parameters_for_g_func)
        value = np.prod(g_funcs)

        # now add some noise
        value_noisy = np.random.normal(loc=value, scale=1, size=1)
        self.drawn_values.append(value_noisy)
        self.bests.append(np.random.normal(loc=self.best_analytical_g, scale=1))

        return value_noisy

    def draw_hyper_ellipsoid(self, input_array):
        """
        :param input_array: input vector, x.
        :return: 500-sigma(i*(x_i-i)^2) where sigma is from 1 to self.dimensions.
        """
        indices = np.arange(1, input_array.size+1)

        value = np.sum(np.multiply(indices, np.power(input_array-indices,2)))
        value = -(value - 500)

        value_noisy = np.random.normal(loc=value, scale=0.1, size=1)
        self.drawn_values.append(value_noisy)
        self.bests.append(np.random.normal(500, 0.1))

        return value_noisy

    def calculate_best_of_analytical_g(self):
        """
        :return: global maximum while the analyical g is constrained by the bounds given above.
        """
        input_array = np.empty(np.size(self.parameters_for_g_func))
        input_array.fill(-2.0)

        g_funcs = (np.absolute(4 * input_array - 2) + self.parameters_for_g_func) / (1.0 + self.parameters_for_g_func)
        value = np.prod(g_funcs)

        return value

    def draw_sixhump_camelback(self, input_array):
        """
        :param input_array: x.
        :return: result of the six hump camelback function.
        """
        x1 = input_array[0]
        x2 = input_array[1]

        value = -1*( (4.0 - 2.1*(x1**2) + (x1**4)/3.0) * x1**2 + x1*x2 + (-4 + 4*(x2**2))*(x2**2) )
        # now add some noise
        value_noisy = np.random.normal(loc=value, scale = 0.02, size=1)
        self.drawn_values.append(value_noisy)
        self.bests.append(np.random.normal(1.0316, 0.02))

        return value_noisy


