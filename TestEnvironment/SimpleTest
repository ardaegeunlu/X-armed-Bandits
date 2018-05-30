# -*- coding: utf-8 -*-
import Partitioner
import TestFunctions
import HOO
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def simple_test():

    # choose a test function.
    testFunction = TestFunctions.TestFunctions(functionName="hyper_ellipsoid", dimensions=10)
    #testFunction = TestFunctions.TestFunctions(functionName="analytical_g", g_params=np.array([0.1, 0.3, 1, 3, 10, 30, 90, 300]))
    #testFunction = TestFunctions.TestFunctions(functionName="SixHumpCamelback")
    
    # get the min and max bounds of the domain of the test function. 
    bounds = testFunction.get_bounds()
    
    #initialize a partitioner that will generate the tree covering sequence from the space-X defined by the above "bounds."
    partitioner = Partitioner.Partitioner(min_values=bounds[0], max_values=bounds[1])
    
    # initialize the bandit algorithm with the following.
    # ro = 0.5 is generally a good choice, for symmetric or near-symmetric X-spaces.
    # a good choice of v1 would be >= dimensions*6 for "hyper_ellipsoid function",
    # and >= 6 for "analytical_g" and "sixhumpcamelback"
    
    x_armed_bandit = HOO.HOO(v1=60, ro=0.5, covering_generator_function=partitioner.halve_one_by_one)
    x_armed_bandit.set_time_horizon(max_plays=3000)
    x_armed_bandit.set_environment(environment_function=testFunction.draw_value)
    x_armed_bandit.run_hoo()
    
    # this is the most rewarding point explored so far after "max_plays" rounds.
    print ("last selected arm was: {0}".format(x_armed_bandit.last_arm))
    
    # the rewards that are received by the bandit should be stored by the environment, as well as the best-fixed strategy.
    rewards = testFunction.drawn_values
    best = testFunction.bests

    # plotting the results.
    # -----------------------------------------------------------------------------------------
    plt.figure(0)

    cum_best = np.cumsum(np.array(best))
    plt.plot(cum_best, label="best strategy reward")
    plt.annotate('%0.2f' % cum_best[- 1], xy=(1, cum_best[- 1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

    cum_rewards = np.cumsum(np.array(rewards))
    plt.plot(cum_rewards, label="agent reward")

    cum_regret = cum_best - cum_rewards
    plt.plot(cum_regret, label="regret")

    plt.annotate('%0.2f' % cum_rewards[-1], xy=(1, cum_rewards[-1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')
    plt.annotate('%0.2f' % cum_regret[- 1], xy=(1, cum_regret[- 1]), xytext=(8, 0),
                 xycoords=('axes fraction', 'data'), textcoords='offset points')

    plt.xlabel("rounds")
    plt.ylabel("cumulative rewards/regret")

    plt.legend()
    plt.show()
    
   
simple_test()
