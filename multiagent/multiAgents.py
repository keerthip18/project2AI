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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        # food
        foodList = newFood.asList()
        if foodList:
            distances = [manhattanDistance(newPos, food) for food in foodList]
            closestFoodDist = min(distances)

            # reciprocal of closest food distance (closer food is better)
            score += 1.0/closestFoodDist

        # ghosts
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)

            if ghostState.scaredTimer > 0:
            # if ghost is scared, closer is better (we want to eat it)
                score += 2.0 / ghostDist
        else:
            # if ghost is not scared, being too close is very bad
            if ghostDist <= 1:
                score -= 500
            else:
                score -= 1.0 / ghostDist

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"
        # use a helper so we can recurse with depth and which agent's turn it is
        def minimax(state, depth, agentIdx):
            # stop when we hit depth limit or game over
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            n = state.getNumAgents()

            if agentIdx == 0:
                # pacman tries to maximize
                best = float("-inf")
                for act in state.getLegalActions(0):
                    succ = state.generateSuccessor(0, act)
                    best = max(best, minimax(succ, depth, 1))
                return best
            else:
                # ghost minimizes. figure out whose turn is next
                nextIdx = agentIdx + 1
                if nextIdx == n:
                    nextIdx = 0
                    depth = depth + 1
                # else same depth, next ghost
                best = float("inf")
                for act in state.getLegalActions(agentIdx):
                    succ = state.generateSuccessor(agentIdx, act)
                    best = min(best, minimax(succ, depth, nextIdx))
                return best

        # at root we need to pick the action that gives best score
        bestScore = float("-inf")
        result = None
        for act in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, act)
            v = minimax(succ, 0, 1)
            if v > bestScore:
                bestScore = v
                result = act
        return result


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphabeta(state, depth, agentIndex, alpha, beta):
            # stop if game over or depth reached
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            numAgents = state.getNumAgents()

            # pacman (max)
            if agentIndex == 0:
                value = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(successor, depth, 1, alpha, beta))
                    if value > beta:  # prune (strict >)
                        return value
                    alpha = max(alpha, value)
                return value

            # ghost(min)
            else:
                value = float("inf")
                nextAgent = agentIndex + 1

                # go back to Pacman and increase depth if last ghost
                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth = depth + 1
                else:
                    nextDepth = depth

                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    if value < alpha:  # prune (strict <)
                        return value
                    beta = min(beta, value)
                return value

        # root call (choose best action for pacman)
        bestValue = float("-inf")
        bestAction = None
        alpha = float("-inf")
        beta = float("inf")

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, 0, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIdx):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            n = state.getNumAgents()

            if agentIdx == 0:
                # pacman maxes
                best = float("-inf")
                for act in state.getLegalActions(0):
                    succ = state.generateSuccessor(0, act)
                    best = max(best, expectimax(succ, depth, 1))
                return best
            else:
                # ghost = expectation (uniform random over actions)
                nextIdx = agentIdx + 1
                if nextIdx == n:
                    nextIdx = 0
                    depth = depth + 1
                legal = state.getLegalActions(agentIdx)
                if len(legal) == 0:
                    return self.evaluationFunction(state)
                total = 0.0
                for act in legal:
                    succ = state.generateSuccessor(agentIdx, act)
                    total += expectimax(succ, depth, nextIdx)
                return total / len(legal)

        bestScore = float("-inf")
        result = None
        for act in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, act)
            v = expectimax(succ, 0, 1)
            if v > bestScore:
                bestScore = v
                result = act
        return result

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: This evaluation function estimates the quality of a 
    Pacman game state by combining the current game score, 
    proximity to the nearest food, distance from ghosts, 
    proximity to power capsules, and the number of remaining 
    food pellets. Closer food and capsules increase the score, 
    scared ghosts are encouraged to be eaten, active ghosts are 
    avoided, and fewer remaining food pellets are rewarded. 
    This allows Pacman to efficiently eat food, chase scared
    ghosts, and avoid danger while progressing toward 
    winning the game.
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # distance to nearest food
    foodList = food.asList()
    if foodList:
        minFoodDist = min([manhattanDistance(pacmanPos, f) for f in foodList])
        score += 1.0 / (minFoodDist + 1) 
    
    # ghosts
    for ghost in ghostStates:
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(pacmanPos, ghostPos)
        if ghost.scaredTimer > 0:
            score += 10.0 / (ghostDist + 1)
        else:
            if ghostDist <= 1:
                score -= 500  
            else:
                score -= 2.0 / ghostDist

    # capsules
    if capsules:
        minCapsuleDist = min([manhattanDistance(pacmanPos, c) for c in capsules])
        score += 5.0 / (minCapsuleDist + 1)

    # remaining food penalty
    score -= 4 * len(foodList)

    return score

# Abbreviation
better = betterEvaluationFunction
