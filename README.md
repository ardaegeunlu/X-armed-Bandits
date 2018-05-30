# X-armed-Bandits

We consider a generalization of stochastic bandits where the set of arms, X, is allowed to be a generic measurable space and
the mean-payoff function is “locally Lipschitz” with respect to a dissimilarity function that is known to the decision maker.
Under this condition we construct an arm selection policy, called HOO (hierarchical optimistic optimization), with improved
regret bounds compared to previous results for a large class of problems. In particular, our results imply that if X is the
unit hypercube in a Euclidean space and the mean-payoff function has a finite number of global maxima around which the
behavior of the function is locally continuous with a known smoothness degree, then the expected regret of HOO is bounded up
to a logarithmic factor by √n, that is, the rate of growth of the regret is independent of the dimension of the space.¹

The HOO strategy incrementally builds an estimate of the mean-payoff function f over X. The core idea is to estimate f
precisely around its maxima, while estimating it loosely in other parts of the space X. To implement this idea, HOO maintains
a binary tree whose nodes are associated with measurable regions of the arm-space X such that the regions associated with
nodes deeper in the tree represent increasingly smaller subsets of X. The tree is built in an incremental manner. At each node
of the tree, HOO stores some statistics based on the information received in previous rounds. In particular, HOO keeps track
of the number of times a node was traversed up to round n and the corresponding empirical average of the rewards received so
far. Based on these, HOO assigns an optimistic estimate (denoted by B) to the maximum mean-payoff associated with each node.
These estimates are then used to select the next node to “play”. This is done by traversing the tree, beginning from the root,
and always following the node with the highest B-value. Once a node is selected, a point in the region associated with it is
chosen (line 16) and is sent to the environment. Based on the point selected and the received reward, the tree is updated.²


¹² S. Bubeck, R. Munos, G Stoltz, and C. Szepesvari. X-armed Bandits. Journal of Machine Learning Research 12 (2011) 1655-1695, 2011.
