# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** CS3568 YOUR CODE HERE ***"
        self.qValue = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** CS3568 YOUR CODE HERE ***"
        # simply return 0.0 if state is not present in Qvalue, else return qValue for found state and action
        if (state, action) not in self.qValue:
            return 0.0
        else:
            return self.qValue[(state,action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** CS3568 YOUR CODE HERE ***"
        
        # Get all actions realted to the given state
        actions = self.getLegalActions(state)
        qval = util.Counter()
        
        # If we don't find any action, it means we are at the Terminal state so, we will return 0.0
        if len(actions) == 0:
            return 0.0
        # For each action, we will add Qvalue in list and will return maximum Qvalue at the end
        else:
            for action in actions:
                value = self.getQValue(state, action)
                qval[action] = value
            
            max_action = qval.argMax()
            return self.getQValue(state, max_action)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** CS3568 YOUR CODE HERE ***"
        
        # Get all actions realted to the given state
        actions = self.getLegalActions(state)
        qval = util.Counter()
        
        # If we don't find any action, it means we are at the Terminal state so, we will return None
        if len(actions) == 0:
            return None
        # For each action, we will add Qvalue in list and will return action who has maximum Qvalue
        else:
            for action in actions:
                value = self.getQValue(state, action)
                qval[action] = value
            
            return qval.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** CS3568 YOUR CODE HERE ***"
        
        # If we don't any actions on given state, it means we are at the Termnal state, we will return None
        if len(legalActions) == 0:
            return None
        else:
            # Flipcoin is used here, to select random action values from all actions
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
                return action
            # If we don't find any epsilon value, we will compute action using their qvalues
            else:
                action = self.computeActionFromQValues(state)
                return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** CS3568 YOUR CODE HERE ***"
        
        # If we don't have any state,action in qvalue, we will update it to 0.0
        if (state, action) not in self.qValue:
            self.qValue[(state, action)] = 0.0
        
        # calculate Qvalue for next state value
        qValue_for_nextState = self.computeValueFromQValues(nextState)
        
        # calculate Qvalue for current state and action
        qvalue_for_currentState = self.qValue[(state, action)]
        
        # store alpha value
        alpha_value = self.alpha
        
        #store discount factor
        discount_factor = self.discount
        
        # calculate updated qValue for state and current action
        updated_qvalue = qvalue_for_currentState + (alpha_value * (reward + (discount_factor * qValue_for_nextState) - qvalue_for_currentState))

        self.qValue[(state,action)] = updated_qvalue
        
    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** CS3568 YOUR CODE HERE ***"
        
        # get all feature(possible distances) for state
        values = self.featExtractor.getFeatures(state, action)
        totalQval = 0
        weight = self.getWeights()

        # return product of weight and feature values
        return values * weight

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** CS3568 YOUR CODE HERE ***"
        # get all feature(possible distances) for state
        values = self.featExtractor.getFeatures(state, action)
        
        # calculate Qvalue for current state and action
        qvalue_for_currentState = self.getQValue(state, action)
        # calculate Qvalue for next state value
        qvalue_for_nextState = self.getValue(nextState)
        
        for value in values:
            # In Question, we have given a formula to update weights for feature, I have implemented that formula in below section. I have calculated correction value which includes normal Q-learning calculation and then added in weights section with alpha and feature values
            correction = (reward + (self.discount*qvalue_for_nextState)) - qvalue_for_currentState
            self.weights[value] = self.weights[value] + self.alpha * correction * values[value]
        
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** CS3568 YOUR CODE HERE ***"
            pass
