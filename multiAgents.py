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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        
        capsules = successorGameState.getCapsules()
        score = successorGameState.getScore()

        # Process the nearest food distance
        food_positions = newFood.asList()
        if len(food_positions) > 0:
            food_distances = [manhattanDistance(newPos, food) for food in food_positions]
            score += 1.0 / min(food_distances)

        # Process the nearest ghost distance
        ghost_distances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        if len(ghost_distances) > 0:
            closest_ghost = min(ghost_distances)
            ghost_index = ghost_distances.index(closest_ghost)
            if closest_ghost > 0:
                score += 1.0 / closest_ghost if newScaredTimes[ghost_index] > 0 else -1.0 / closest_ghost

        # Process the nearest capsule distance
        if len(capsules) > 0:
            capsule_distances = [manhattanDistance(newPos, capsule) for capsule in capsules]
            score += 1.0 / min(capsule_distances)

        # Penalize for remaining food count
        score -= len(food_positions)

        return score

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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        def minimax(agent, currentDepth, state):
            """
            Recursive function that returns the minimax value for a given agent and state.
            Switches between maximizing for Pacman and minimizing for each ghost,
            and terminates when maximum depth or a terminal state is reached.
            """
            # Terminal state or maximum depth reached
            if currentDepth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Pacman's turn to maximize value
            if agent == 0:
                return pacmanTurn(agent, currentDepth, state)
            # Ghosts' turn to minimize value
            else:
                return ghostTurn(agent, currentDepth, state)

        def pacmanTurn(agent, depth, state):
            """
            Maximizing function for Pacman's turn.
            """
            highest_score = float('-inf')
            chosen_action = None
            for move in state.getLegalActions(agent):
                successor_state = state.generateSuccessor(agent, move)
                score = minimax(1, depth, successor_state)
                if score > highest_score:
                    highest_score = score
                    chosen_action = move
            return chosen_action if depth == 0 else highest_score

        def ghostTurn(agent, depth, state):
            """
            Minimizing function for each ghost's turn.
            """
            lowest_score = float('inf')
            next_agent = agent + 1
            if next_agent == state.getNumAgents():
                next_agent = 0
                depth += 1
            for move in state.getLegalActions(agent):
                successor_state = state.generateSuccessor(agent, move)
                score = minimax(next_agent, depth, successor_state)
                lowest_score = min(lowest_score, score)
            return lowest_score

        return minimax(0, 0, gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alphabeta(agent, currentDepth, state, alpha, beta):
            """
            Executes the alpha-beta pruning algorithm, alternating between maximizing (Pacman)
            and minimizing (ghosts), with cutoffs based on alpha and beta values.
            """
            # Terminal state or maximum depth reached
            if currentDepth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Pacman's turn to maximize value
            if agent == 0:
                return pacmanMax(agent, currentDepth, state, alpha, beta)
            # Ghosts' turn to minimize value
            else:
                return ghostMin(agent, currentDepth, state, alpha, beta)

        def pacmanMax(agent, depth, state, alpha, beta):
            """
            Maximizing function for Pacman's turn.
            """
            max_score = float('-inf')
            best_action = None
            for move in state.getLegalActions(agent):
                successor_state = state.generateSuccessor(agent, move)
                score = alphabeta(1, depth, successor_state, alpha, beta)
                if score > max_score:
                    max_score = score
                    best_action = move
                # Prune the search
                if max_score > beta:
                    return max_score
                alpha = max(alpha, max_score)
            return best_action if depth == 0 else max_score

        def ghostMin(agent, depth, state, alpha, beta):
            """
            Minimizing function for each ghost's turn.
            """
            min_score = float('inf')
            next_agent = agent + 1
            if next_agent == state.getNumAgents():
                next_agent = 0
                depth += 1

            for move in state.getLegalActions(agent):
                successor_state = state.generateSuccessor(agent, move)
                score = alphabeta(next_agent, depth, successor_state, alpha, beta)
                min_score = min(min_score, score)
                # Prune the search
                if min_score < alpha:
                    return min_score
                beta = min(beta, min_score)
            return min_score

        return alphabeta(0, 0, gameState, float('-inf'), float('inf'))

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax(agent, currentDepth, state):
            """
            Recursively calculates the expectimax value for a given agent and state.
            Maximizes value for Pacman and averages values for ghosts, terminating at max depth or end states.
            """
            # Terminal state or maximum depth reached
            if currentDepth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # Pacman's turn to maximize value
            if agent == 0:
                return pacmanMax(agent, currentDepth, state)
            # Ghosts' turn, treated as chance nodes
            else:
                return ghostExpect(agent, currentDepth, state)

        def pacmanMax(agent, depth, state):
            """
            Maximizing function for Pacman's turn.
            """
            max_score = float('-inf')
            chosen_action = None
            for move in state.getLegalActions(agent):
                successor_state = state.generateSuccessor(agent, move)
                score = expectimax(1, depth, successor_state)
                if score > max_score:
                    max_score = score
                    chosen_action = move
            return chosen_action if depth == 0 else max_score

        def ghostExpect(agent, depth, state):
            """
            Expectation function for ghosts' turn.
            """
            expected_value = 0
            actions = state.getLegalActions(agent)
            probability = 1.0 / len(actions)
            next_agent = agent + 1
            if next_agent == state.getNumAgents():
                next_agent = 0
                depth += 1

            for move in actions:
                successor_state = state.generateSuccessor(agent, move)
                score = expectimax(next_agent, depth, successor_state)
                expected_value += probability * score
            return expected_value

        return expectimax(0, 0, gameState)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    The evaluation function calculates the score based on the following features:
    - The distance to the closest food pellet
    - The distance to the closest ghost
    - The distance to the closest capsule
    - The number of scared ghosts
    - The remaining food pellets
    - The remaining capsules


    The evaluation function uses the following arbitrary weights:
    - Food pellet weight: 10.0
    - Ghost weight: -200.0
    - Capsule weight: 100.0
    - Scared ghost weight: 200.0

    The evaluation function calculates the score based on the following formula:
    score = current score + food weight / closest food distance + capsule weight / closest capsule distance + scared ghost weight * sum(scared times)

    The evaluation function penalizes the score if the Pacman is too close to a ghost, and penalizes the remaining food and capsules.
    """
    # Information from the current game state
    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghosts = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghosts]

    score = currentGameState.getScore()

    # Calculate distances to the closest food
    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in food.asList()]
    if foodDistances:
        closestFoodDist = min(foodDistances)
    else:
        closestFoodDist = 1

    # Calculate distances to the ghosts
    ghostDistances = [manhattanDistance(pacmanPos, ghost.getPosition()) for ghost in ghosts]
    if ghostDistances:
        closestGhostDist = min(ghostDistances)
    else:
        closestGhostDist = 1

    # Calculate distances to the capsules
    capsuleDistances = [manhattanDistance(pacmanPos, capsule) for capsule in capsules]
    if capsuleDistances:
        closestCapsuleDist = min(capsuleDistances)
    else:
        closestCapsuleDist = 1

    # Feature weights
    foodWeight = 10.0
    ghostWeight = -200.0
    capsuleWeight = 100.0
    scaredGhostWeight = 200.0

    # Calculate the evaluation score
    score += foodWeight / closestFoodDist + capsuleWeight / closestCapsuleDist + scaredGhostWeight * sum(scaredTimes)

    # Penalize being too close to a ghost
    if closestGhostDist < 2:
        score += ghostWeight / (closestGhostDist + 1)

    # Penalize remaining food and capsules
    score -= len(food.asList()) + len(capsules)

    return score

better = betterEvaluationFunction