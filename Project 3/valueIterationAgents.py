# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** CS3568 YOUR CODE HERE ***"
        
        states = self.mdp.getStates()

        # iterate for 0 to given interation value
        for i in range(0, self.iterations):
            
            counter_values = util.Counter()
            
            # calculate qvalues for all action which is related to current state value
            for state in states:
                # Get list of actions for current state
                action = self.getAction(state)
                
                # If we don't have action, then we will move for another state, else, we will compute Q values for all actions
                if action is not None:
                    counter_values[state] = self.getQValue(state, action)
            
            self.values = counter_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** CS3568 YOUR CODE HERE ***"

        # Initialize total Q value to 0
        total_qvalue = 0
        
        # Get all nextState value avaialable for current (state, action) value
        transition_state_value = self.mdp.getTransitionStatesAndProbs(state, action)
        
        # Iterate through all nextState value
        for nextState, probability_value in transition_state_value:  
            
            # Get reward for current action and nextState
            reward = self.mdp.getReward(state, action, nextState)
            
            # Get nextState qvalue
            nextState_value = self.getValue(nextState)
            discount_value = self.discount
            
            # calculate Q value for current action and nextState value using Reward and Probability value
            total_qvalue = total_qvalue + (probability_value*(reward + (discount_value * nextState_value)))

        return total_qvalue
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** CS3568 YOUR CODE HERE ***"

        # If curretn state is terminal, it means we don't have any action to calculate, we will return None value
        if self.mdp.isTerminal(state):
            return None
        
        # Get all actions related to current state
        actions = self.mdp.getPossibleActions(state)
        qvalue = util.Counter()
        
        # Iterate through all actions to calcualte Qvalue
        for action in actions:
            # Get Qvalue for current stae and action
            value = self.getQValue(state, action)
            # store qvalues in array
            qvalue[action] = value
        
        # return action who has maximum Qvalue
        return qvalue.argMax()

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
