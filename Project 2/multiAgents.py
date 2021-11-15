# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluate_Reflex(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluate_Reflex(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPosition = successorGameState.getGhostPositions()
        newCapsulePosition = successorGameState.getCapsules()

        "*** CS3568 YOUR CODE HERE ***"
        "Decribe your function:"
        
        ##In Evaluation Function, I am using nearest food and nearest ghost distance from current position of our actor. 
        ##We are calculating distance using manhattan distance.  
        ##Then, we find minimum distance from calculated list of food_distance
        ##we find minimum distance from calculated list of ghost_distance
        ##In addition, we have calculated distance of capsule also and find nearest capsule from current location
        ##If ghost is near to the our actor/pacman, then we are  going to check that any capsule is near from its location. If it is near from current locatioon, then we are updating score, else we are decrementing score by some penalty.
        ##We are returning score value, by getting fraction of nearest_food_distance and nearest_ghost_distance with boundry of current game state wall to avoid unnecessary computation
        
        score = 0
        
        score = score + successorGameState.getScore() 
        
        ##caculating distance from current position to nearest food position, then find nearest food distance from current position
        distance_current_to_food= []
        food_distance = 10000
        for food in newFood:
            ##print(food)
            distance = calculateManhattanDistance(food, newPos)
            distance_current_to_food.append(distance)
            if food_distance > distance:    
                food_distance = distance

        ##caculating distance from current position to nearest ghost position, then find nearest ghost location
        distance_current_to_ghost = []
        ghost_distance = 10000
        for ghost in newGhostPosition:
            ghost_d = calculateManhattanDistance(ghost, newPos)
            distance_current_to_ghost.append(ghost_d)
            if ghost_distance > ghost_d:
                ghost_distance = ghost_d
        
        ##calculating distance from current position to nearest capsule position, then find nearest capsule position
        distance_current_to_capsule = []
        capsule_distance = 10000
        for capsule in newCapsulePosition:
            capsule_d = calculateManhattanDistance(capsule, newPos)
            distance_current_to_capsule.append(capsule_d)
            if capsule_distance > capsule_d:
                capsule_distance = capsule_d
        
        ##checking if ghost distance is not less than 3, then we will check capsule_distnace is less than 2 (checking current location to capsule is near or not), if current location is near to capsule, update score or else decrement score value
        if ghost_distance < 3:
            if capsule_distance < 2:
                score = score + 25.0
            else:
                score = score - 50.0

        ##if everything is ok, we will add successor's score with fraction of nearest fooddistance and ghost distance by current wall position  which helps us to avoid calculating unwanted index from game stage
        score = score + (1.0/(food_distance + ghost_distance))/ ((currentGameState.getWalls().height - 2)+(currentGameState.getWalls().width - 2))
        
        return score

def calculateManhattanDistance(xy1, xy2):
    distance = abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    return distance

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    
    ##calculating score with minmax function. For this, we will consider state of agent, depth of tree, agent position.
    def calculate_Minmax(self, state, depth, agent):
        ## if state is lose or win or if depth is equal current depth, return evaluated score value
        if state.isLose() or state.isWin() or depth == self.depth:
            return self.evaluate_Minmax(state)
        ##if agent value is equal to zero, it means, we are at the root element of the tree and we are going to return maximze value of the node value
        if agent == 0:
            actions = state.getLegalActions(agent)
            score = -999999
            for action in actions:
                val = self.calculate_Minmax(state.generateSuccessor(agent, action), depth, 1)
                if score < val:
                    score = val
            return score
        ##if it is child, calculate minimum value of state. increment agentPos value by 1, then check num_agents with agentPos, if is equal, it means we are done with all ghost value, so we reset to 0
        else:
            new_agent = agent + 1
            if state.getNumAgents() == new_agent:
                new_agent = 0
            if new_agent == 0:
                depth = depth + 1
            ## get all legal action related to agent position, we will calculate minimum value of score from minmax function
            actions = state.getLegalActions(agent)
            score = 9999999
            for action in actions:
                val = self.calculate_Minmax(state.generateSuccessor(agent, action), depth, new_agent)
                if score > val:
                    score = val
            return score

    
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** CS3568 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        
        ##we are calculating minmax value for each action from state legal actions and then we are selecting maximum value after each calculation
        actions = gameState.getLegalActions()
        
        score = -99999
        
        ##calculate action for agent with minmax method, here we are caculating a max score
        for action in actions:
            val = self.calculate_Minmax(gameState.generateSuccessor(0, action), 0, 0) 
            if val > score:
                score = val
                next_action = action
                
        return next_action
    
    def evaluate_Minmax(self,currentGameState):
        # Useful information you can extract from a GameState (pacman.py)
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood().asList()
        newGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPosition = currentGameState.getGhostPositions()
        newCapsulePosition = currentGameState.getCapsules()

        "*** CS3568 YOUR CODE HERE ***"
        "Decribe your function:"
            
        ##In Evaluation Function, I am using nearest food and nearest ghost distance from current position of our actor. 
        ##We are calculating distance using manhattan distance.  
        ##Then, we find minimum distance from calculated list of food_distance
        ##we find minimum distance from calculated list of ghost_distance
        ##In addition, we have calculated distance of capsule also and find nearest capsule from current location
        ##If ghost is near to the our actor/pacman, then we are  going to check that any capsule is near from its location. If it is near from current locatioon, then we are updating score, else we are decrementing score by penalty.
        ##We are returning score value, by getting fraction of nearest_food_distance and nearest_ghost_distance with boundry of current game state wall to avoid unnecessary computation
            
        score = 0
            
        score = score + currentGameState.getScore()
        
        ##caculating distance from current position to nearest food position, then find nearest food distance from current position
        distance_current_to_food= []
        food_distance = 10000
        for food in newFood:
            ##print(food)
            distance = calculateManhattanDistance(food, newPos)
            distance_current_to_food.append(distance)
            if food_distance > distance:    
                food_distance = distance

        ##calculating distance from current position to nearest capsule position, then find nearest capsule position
        distance_current_to_capsule = []
        capsule_distance = 10000
        for capsule in newCapsulePosition:
            capsule_d = calculateManhattanDistance(capsule, newPos)
            distance_current_to_capsule.append(capsule_d)
            if capsule_distance > capsule_d:
                capsule_distance = capsule_d
            
        ##check ghost distance is less than 5 and if scaredtime of ghost = 0 (it means, ghost is not scared), and capsule distance is more than 3, then we will decrement score value. 
        ##else if, ghost distance is less than 5 and scared time of ghost is more than 0 (it means, ghost is scare , we can eat it), so we will increment score to eat it.
        for ghost_state in currentGameState.getGhostStates():
            ghost_d = calculateManhattanDistance(ghost_state.configuration.getPosition(), newPos)
            if ghost_d < 5 and ghost_state.scaredTimer == 0 and capsule_distance > 5:
                score = score - 200.0
            elif ghost_d < 5 and ghost_state.scaredTimer > 0:
                score = score + 150.0
            elif ghost_d < 5 and ghost_state.scaredTimer == 0 and capsule_distance < 5:
                score = score + 50.0

        ##we are checking position of capsule with current position of pacman, then we are updating score value
        x = newPos[0]
        y = newPos[1]
        for capsule in newCapsulePosition:
            if (capsule[0] == x) and (capsule[1] == y):
                score = score + 50.0
        
        ##we are checking position of ghost with current position of pacman with scaredtime is greater than 0, then we are updating score value
        for ghost_state in currentGameState.getGhostStates():
            ghostPos = ghost_state.configuration.getPosition()
            if ghost_state.scaredTimer > 0 and (ghostPos[0] == x) and (ghostPos[1] == y):
                score = score + 200.0

        ##At the end, we are calculating difference between minimum food distance with length of distance_current_to_food and divide it to score to get fractinal value , then add it to score
        v = (food_distance - len(distance_current_to_food)*10.0)
        if v == 0:
            v = food_distance
        score = score + (10.0/v)
      
        return score    

    
##its the same evaluation function which is defined in ReflexAgent, just mention differently to avoid confusion    

    
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def calculate_AlphaBeta(self, alpha, beta, state, depth, agent):
        
        if state.isLose() or state.isWin() or depth == self.depth:
            return self.evaluate_AlphaBeta(state)
        if agent == 0:
            actions = state.getLegalActions(agent)
            score = -999999
            for action in actions:
                val = self.calculate_AlphaBeta(alpha, beta, state.generateSuccessor(agent, action), depth, 1)
                if score < val:
                    score = val
                if alpha < score:
                    alpha = score
                if alpha >= beta:
                    break
                    
            return score
        else:
            new_agent = agent + 1
            if state.getNumAgents() == new_agent:
                new_agent = 0
            if new_agent == 0:
                depth = depth + 1
            
            actions = state.getLegalActions(agent)
            score = 9999999
            for action in actions:
                val = self.calculate_AlphaBeta(alpha, beta, state.generateSuccessor(agent, action), depth, new_agent)
                if score > val:
                    score = val
                if beta > score:
                    beta = score
                if alpha >= beta:
                    break;
                    
            return score

    
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** CS3568 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        actions = gameState.getLegalActions()
        
        score = -99999
        
        alpha = -99999999
        beta = 99999999
        
        for action in actions:
            val = self.calculate_AlphaBeta(alpha, beta, gameState.generateSuccessor(0, action), 0, 0) 
            if val > score:
                score = val
                next_action = action
                
        return next_action
    
    def evaluate_AlphaBeta(self, currentGameState):
        # Useful information you can extract from a GameState (pacman.py)
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood().asList()
        newGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPosition = currentGameState.getGhostPositions()
        newCapsulePosition = currentGameState.getCapsules()

        "*** CS3568 YOUR CODE HERE ***"
        "Decribe your function:"
            
        ##In Evaluation Function, I am using nearest food and nearest ghost distance from current position of our actor. 
        ##We are calculating distance using manhattan distance.  
        ##Then, we find minimum distance from calculated list of food_distance
        ##we find minimum distance from calculated list of ghost_distance
        ##In addition, we have calculated distance of capsule also and find nearest capsule from current location
        ##If ghost is near to the our actor/pacman, then we are  going to check that any capsule is near from its location. If it is near from current locatioon, then we are updating score, else we are decrementing score by penalty.
        ##We are returning score value, by getting fraction of nearest_food_distance and nearest_ghost_distance with boundry of current game state wall to avoid unnecessary computation
            
        score = 0
            
        score = score + currentGameState.getScore()
        
        count = 0
        
        ##caculating distance from current position to nearest food position, then find nearest food distance from current position
        distance_current_to_food= []
        food_distance = 10000
        for food in newFood:
            ##print(food)
            distance = calculateManhattanDistance(food, newPos)
            distance_current_to_food.append(distance)
            if food_distance > distance:    
                food_distance = distance
                count = count + 1
                
        if count > 1:
            score = score + 10.0

        ##calculating distance from current position to nearest capsule position, then find nearest capsule position
        distance_current_to_capsule = []
        capsule_distance = 10000
        for capsule in newCapsulePosition:
            capsule_d = calculateManhattanDistance(capsule, newPos)
            distance_current_to_capsule.append(capsule_d)
            if capsule_distance > capsule_d:
                capsule_distance = capsule_d
        
        ##check ghost distance is less than 5 
        ##and if scaredtime of ghost = 0(it means, ghost is not scared) with capsule distane less han 5 (it means we can eat capsule to increase power of pacman, then we will decrement score value. 
        ##else if, scared time of ghost is more than 0 (it means, ghost is scare , we can eat it), so we will increment score to eat it.
        ##else, if none of the condition is match, it means our pacman is in danger and we will decrement score
        
        for ghost_state in currentGameState.getGhostStates():
            ghost_d = calculateManhattanDistance(ghost_state.configuration.getPosition(), newPos)
            if ghost_d < 5:
                if capsule_distance < 5 and ghost_state.scaredTimer == 0:
                    score = score + 50.0
                elif ghost_state.scaredTimer > 0:
                    score = score + 70.0
                else:
                    score = score - 100
        
        ##we are checking position of capsule with current position of pacman, then we are updating score value
        x = newPos[0]
        y = newPos[1]
        for capsule in newCapsulePosition:
            if (capsule[0] == x) and (capsule[1] == y):
                score = score + 50.0

        ##we are checking position of ghost with current position of pacman with scaredtime is greater than 0, then we are updating score value
        for ghost_state in currentGameState.getGhostStates():
            ghostPos = ghost_state.configuration.getPosition()
            if ghost_state.scaredTimer > 0 and (ghostPos[0] == x) and (ghostPos[1] == y):
                score = score + 50.0
            
        ##At the end, we are calculating difference between minimum food distance with length of distance_current_to_food and divide it to score to get fractinal value , then add it to score
        v = (food_distance - len(distance_current_to_food)*10.0)
        if v == 0:
            v = food_distance
        score = score + (10.0/v)
            
        return score    


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def calculate_Expectimax(self, state, depth, agent):
        
        if state.isLose() or state.isWin() or depth == self.depth:
            return self.evaluate_Expectimax(state)
        if agent == 0:
            actions = state.getLegalActions(agent)
            score = -999999
            for action in actions:
                val = self.calculate_Expectimax(state.generateSuccessor(agent, action), depth, 1)
                if score < val:
                    score = val
            return score
        else:
            new_agent = agent + 1
            if state.getNumAgents() == new_agent:
                new_agent = 0
            if new_agent == 0:
                depth = depth + 1
            
            actions = state.getLegalActions(agent)
            score = 0
            for action in actions:
                val = self.calculate_Expectimax(state.generateSuccessor(agent, action), depth, new_agent)
                score = score + val
            score = score / (len(actions))
            
            return score
    
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** CS3568 YOUR CODE HERE ***"
        "PS. It is okay to define your own new functions. For example, value, min_function,max_function"
        actions = gameState.getLegalActions()
        
        score = -99999

        for action in actions:
            val = self.calculate_Expectimax(gameState.generateSuccessor(0, action), 0, 0) 
            if val > score:
                score = val
                next_action = action
                
        return next_action
    
    def evaluate_Expectimax(self, currentGameState):
        # Useful information you can extract from a GameState (pacman.py)
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood().asList()
        newGhostStates = currentGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPosition = currentGameState.getGhostPositions()
        newCapsulePosition = currentGameState.getCapsules()

        "*** CS3568 YOUR CODE HERE ***"
        "Decribe your function:"
            
        ##In Evaluation Function, I am using nearest food and nearest ghost distance from current position of our actor. 
        ##We are calculating distance using manhattan distance.  
        ##Then, we find minimum distance from calculated list of food_distance
        ##we find minimum distance from calculated list of ghost_distance
        ##In addition, we have calculated distance of capsule also and find nearest capsule from current location
        ##If ghost is near to the our actor/pacman, then we are  going to check that any capsule is near from its location. If it is near from current locatioon, then we are updating score by 1, else we are decrementing score by penalty of 50.
        ##We are returning score value, by getting fraction of nearest_food_distance and nearest_ghost_distance with boundry of current game state wall to avoid unnecessary computation
            
        score = 0
            
        score = score + currentGameState.getScore()
        
        count = 0
        
        ##caculating distance from current position to nearest food position, then find nearest food distance from current position
        distance_current_to_food= []
        food_distance = 10000
        for food in newFood:
            ##print(food)
            distance = calculateManhattanDistance(food, newPos)
            distance_current_to_food.append(distance)
            if food_distance > distance:    
                food_distance = distance
                count = count + 1
                
        if count > 1:
            score = score+10.0

            ##calculating distance from current position to nearest capsule position, then find nearest capsule position
        distance_current_to_capsule = []
        capsule_distance = 10000
        for capsule in newCapsulePosition:
            capsule_d = calculateManhattanDistance(capsule, newPos)
            distance_current_to_capsule.append(capsule_d)
            if capsule_distance > capsule_d:
                capsule_distance = capsule_d
            
        ##check ghost distance is less than 5 
        ##and if scaredtime of ghost = 0(it means, ghost is not scared) with capsule distane less han 5 (it means we can eat capsule to increase power of pacman, then we will decrement score value. 
        ##else if, scared time of ghost is more than 0 (it means, ghost is scare , we can eat it), so we will increment score to eat it.
        ##else, if none of the condition is match, it means our pacman is in danger and we will decrement score
        
        for ghost_state in currentGameState.getGhostStates():
            ghost_d = calculateManhattanDistance(ghost_state.configuration.getPosition(), newPos)
            if ghost_d < 5:
                if capsule_distance < 3 and ghost_state.scaredTimer == 0:
                    score = score + 50.0
                elif ghost_state.scaredTimer > 0:
                    score = score + 70.0
                else:
                    score = score - 100
        
        ##we are checking position of capsule with current position of pacman, then we are updating score value
        x = newPos[0]
        y = newPos[1]
        for capsule in newCapsulePosition:
            if (capsule[0] == x) and (capsule[1] == y):
                score = score + 50.0

        ##we are checking position of ghost with current position of pacman with scaredtime is greater than 0, then we are updating score value
        for ghost_state in currentGameState.getGhostStates():
            ghostPos = ghost_state.configuration.getPosition()
            if ghost_state.scaredTimer > 0 and (ghostPos[0] == x) and (ghostPos[1] == y):
                score = score + 50.0
            
        ##At the end, we are calculating difference between minimum food distance with length of distance_current_to_food and divide it to score to get fractinal value , then add it to score
        v = (food_distance - len(distance_current_to_food)*10.0)
        if v == 0:
            v = food_distance
        score = score + (10.0/v)
            
        return score
    

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** CS3568 YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newGhostPosition = currentGameState.getGhostPositions()
    newCapsulePosition = currentGameState.getCapsules()

    "*** CS3568 YOUR CODE HERE ***"
    "Decribe your function:"
        
        ##In Evaluation Function, I am using nearest food and nearest ghost distance from current position of our actor. 
        ##We are calculating distance using manhattan distance.  
        ##Then, we find minimum distance from calculated list of food_distance
        ##we find minimum distance from calculated list of ghost_distance
        ##In addition, we have calculated distance of capsule also and find nearest capsule from current location
        ##If ghost is near to the our actor/pacman, then we are  going to check that any capsule is near from its location. If it is near from current locatioon and if scaredTimer is 0 or if capsule is not near from current location but ghost scaredTimer is greater than 0, then we are updating score by 1, else we are decrementing score by penalty of 50.
        ##We are returning score value, by getting fraction of nearest_food_distance minus lenth of total food distance
        
    score = 0
    
    x = newPos[0]
    y = newPos[1]
    
    score = score + currentGameState.getScore()    
    
    count = 0
    ##caculating distance from current position to nearest food position, then find nearest food distance from current position
    distance_current_to_food= []
    food_distance = 10000
    for food in newFood:
        ##print(food)
        distance = calculateManhattanDistance(food, newPos)
        distance_current_to_food.append(distance)
        if food_distance >= distance:    
            food_distance = distance
            score = score + food_distance*10.0

    ##calculating distance from current position to nearest capsule position, then find nearest capsule position
    distance_current_to_capsule = []
    capsule_distance = 10000
    capsulePos_x = 0
    capsulePos_y = 0
    for capsule in newCapsulePosition:
        capsule_d = calculateManhattanDistance(capsule, newPos)
        distance_current_to_capsule.append(capsule_d)
        if capsule_distance >= capsule_d:
            capsule_distance = capsule_d
            capsulePos_x = capsule[0]
            capsulePos_y = capsule[1]


    ghost_distance = 100000  
    ghostPos = 0
    for ghost_state in currentGameState.getGhostStates():
        ghost_d = calculateManhattanDistance(ghost_state.configuration.getPosition(), newPos)
        if ghost_distance >= ghost_d:
            ghost_distance = ghost_d
            ghost_Pos = ghost_state.configuration.getPosition()
            if ghost_distance < 5 and ghost_state.scaredTimer == 0:
                if len(distance_current_to_capsule) != 0:
                    if capsule_distance < 5:
                        score = score + 150.0
                    else:
                        score = score - 150.0
                else:
                    score = score - 200.0
            elif ghost_state.scaredTimer > 0:
                score = score + 100.0

    
    if ghost_Pos[0] == x and ghost_Pos[1] == y:
        score = score + 200.0
    
    
    if (capsulePos_x == x) and (capsulePos_y == y):
        score = score + 100.0

    ##At the end, we are calculating difference between minimum food distance with length of distance_current_to_food and divide it to score to get fractinal value , then add it to score    
    v = (food_distance - len(distance_current_to_food)*10.0)
    
    if v <= 0:
        v = food_distance
    score = score + (10.0/v)
    
    return score    
    

# Abbreviation
better = betterEvaluationFunction
