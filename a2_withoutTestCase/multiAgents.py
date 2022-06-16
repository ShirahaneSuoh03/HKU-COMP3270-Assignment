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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        heuristic=0
        position=newPos
        ghostPos=successorGameState.getGhostPositions()
        ghostDist=[]
        for i in ghostPos:
            ghostDist.append(abs(newPos[0]-i[0])+abs(newPos[1]-i[1]))
        totalGhostDist=0
        for i in ghostDist:
            totalGhostDist+=i
        foodDist=[]
        foodList=newFood.asList()
        foodDist = []
        distBetweenFood = []
        for food in foodList:
            foodDist.append(abs(position[0] - food[0]) + abs(position[1] - food[1]))
            temp = []
            for food1 in foodList:
                distBetweenFood.append(abs(food1[0] - food[0]) + abs(food1[1] - food[1]))
        maxDistBet = 0
        if (len(distBetweenFood) > 0):
            maxDistBet = max(distBetweenFood)
            pos = distBetweenFood.index(maxDistBet)
            x, y = pos // len(foodList), pos % len(foodList)
            distx = abs(position[0] - foodList[x][0]) + abs(position[1] - foodList[x][1])
            disty = abs(position[0] - foodList[y][0]) + abs(position[1] - foodList[y][1])
            if (distx > disty):
                heuristic = disty
            else:
                heuristic = distx
            heuristic += maxDistBet

        else:
            if (len(foodDist) > 0):
                heuristic = max(foodDist)
        # for x in range(newFood.width):
        #     dist=0
        #     for y in range(newFood.height):
        #         dist= abs(x-newPos[0]) + abs(y-newPos[1])
        #         foodDist.append(dist)
        # totalDist=0
        # for i in foodDist:
        #     totalDist+=i
        if(heuristic==0):
            return 99999
        if(prevFood==newFood and (min(ghostDist)>=4)):
            eva= 0
            return eva
        if ((totalGhostDist>= (len(ghostDist)*4)) and (min(ghostDist)>=4)):
            eva = (999 * successorGameState.getScore()) / heuristic
            return eva
        if (min(newScaredTimes)>0):
            eva= (999 * successorGameState.getScore()) / heuristic
            return eva
        eva = ((totalGhostDist) * successorGameState.getScore()) / heuristic
        return eva
        return successorGameState.getScore()
        

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
        evaList=[]
        actionList=[]
        for legalAction in gameState.getLegalActions(0):
            eva = self.minimax(1,self.depth,gameState.generateSuccessor(0,legalAction))
            evaList.append(eva)
            actionList.append(legalAction)
        maxVal=max(evaList)
        count=0
        for i in evaList:
            if (i == maxVal):
                break
            count+=1
        return actionList[count]

        util.raiseNotDefined()

    def minimax(self, agent, depth, gameState):
        
        if (depth == 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(0)) == 0):
            return self.evaluationFunction(gameState)
        numOfAgent = gameState.getNumAgents()

        if agent == 0:

            child = []
            self.currentIndex = 1
            for legalAction in gameState.getLegalActions(self.index):
                child.append(self.minimax(1, depth, gameState.generateSuccessor(self.index, legalAction)))
            return max(child)
        else:

            next=agent+1
            if(next == numOfAgent):
                next=0
                depth-=1

            child = []
            
            for legalAction in gameState.getLegalActions(agent):
                child.append(self.minimax(next,depth, gameState.generateSuccessor(agent, legalAction)))
            return min(child)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        alpha=float('-inf')
        beta=float('inf')
        eva=float('-inf')
        evaList = []
        actionList = []
        bestAction=None
        for legalAction in gameState.getLegalActions(0):
            eva = self.minimax(1, self.depth, gameState.generateSuccessor(0, legalAction),alpha,beta)
            if(alpha<eva):
                alpha=eva
                bestAction=legalAction
        return bestAction
        #     evaList.append(eva)
        #     actionList.append(legalAction)
        # maxVal = max(evaList)
        # count = 0
        # for i in evaList:
        #     if (i == maxVal):
        #         break
        #     count += 1
        # return actionList[count]

        util.raiseNotDefined()

    def minimax(self, agent, depth, gameState, alpha, beta):

        if (depth == 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(0)) == 0):
            return self.evaluationFunction(gameState)
        numOfAgent = gameState.getNumAgents()

        if agent == 0:
            v = float('-inf')
            child = []
            self.currentIndex = 1
            for legalAction in gameState.getLegalActions(self.index):
                v=max(v,self.minimax(1, depth, gameState.generateSuccessor(self.index, legalAction), alpha, beta))
                if (v>beta):
                    return v
                alpha=max(alpha,v)
            return v
        else:
            v=float('inf')
            next = agent + 1
            if (next == numOfAgent):
                next = 0
                depth -= 1

            # child = []

            for legalAction in gameState.getLegalActions(agent):
                v=min(v,self.minimax(next, depth, gameState.generateSuccessor(agent, legalAction), alpha, beta))
                if(v<alpha):
                    return v
                beta=min(beta,v)
            return v
        

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
        "*** YOUR CODE HERE ***"
        evaList = []
        actionList = []
        for legalAction in gameState.getLegalActions(0):
            eva = self.minimax(1, self.depth, gameState.generateSuccessor(0, legalAction))
            evaList.append(eva)
            actionList.append(legalAction)
        maxVal = max(evaList)
        count = 0
        for i in evaList:
            if (i == maxVal):
                break
            count += 1
        return actionList[count]

        util.raiseNotDefined()

    def minimax(self, agent, depth, gameState):

        if (depth == 0 or gameState.isWin() or gameState.isLose() or len(gameState.getLegalActions(0)) == 0):
            return self.evaluationFunction(gameState)
        numOfAgent = gameState.getNumAgents()

        if agent == 0:

            child = []
            self.currentIndex = 1
            for legalAction in gameState.getLegalActions(self.index):
                child.append(self.minimax(1, depth, gameState.generateSuccessor(self.index, legalAction)))
            return max(child)
        else:

            next = agent + 1
            if (next == numOfAgent):
                next = 0
                depth -= 1

            child = []

            for legalAction in gameState.getLegalActions(agent):
                child.append(self.minimax(next, depth, gameState.generateSuccessor(agent, legalAction)))
            expectValue=0.0
            for value in child:
                expectValue+=value
            expectValue=expectValue/len(child)
            return expectValue

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # prevFood = currentGameState.getFood()
    # successorGameState = currentGameState.generatePacmanSuccessor(action)

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    heuristic = 0
    position = newPos
    ghostPos = currentGameState.getGhostPositions()
    ghostDist = []
    for i in ghostPos:
        ghostDist.append(abs(newPos[0] - i[0]) + abs(newPos[1] - i[1]))
    totalGhostDist = 0
    for i in ghostDist:
        totalGhostDist += i
    foodDist = []
    foodList = newFood.asList()
    foodDist = []
    distBetweenFood = []
    for food in foodList:
        foodDist.append(abs(position[0] - food[0]) + abs(position[1] - food[1]))
        temp = []
        for food1 in foodList:
            distBetweenFood.append(abs(food1[0] - food[0]) + abs(food1[1] - food[1]))
    maxDistBet = 0
    if (len(distBetweenFood) > 0):
        maxDistBet = max(distBetweenFood)
        pos = distBetweenFood.index(maxDistBet)
        x, y = pos // len(foodList), pos % len(foodList)
        distx = abs(position[0] - foodList[x][0]) + abs(position[1] - foodList[x][1])
        disty = abs(position[0] - foodList[y][0]) + abs(position[1] - foodList[y][1])
        if (distx > disty):
            heuristic = disty
        else:
            heuristic = distx
        heuristic += maxDistBet

    else:
        if (len(foodDist) > 0):
            heuristic = max(foodDist)
    
    if(currentGameState.isLose()):
        return float('-inf')
    # if (heuristic == 0):
    #     return 99999

    # if ((totalGhostDist >= (len(ghostDist) * 3)) or (min(ghostDist) >= 3)):
    #     eva = (currentGameState.getScore() * totalGhostDist) / (heuristic*currentGameState.getNumFood())
    #
    #     return eva
    # if (min(newScaredTimes) > 0):
    #     eva = (9999999 * currentGameState.getScore()) / heuristic
    #     return eva


    nearestGhostDistance = float("inf")
    ghostEva = 0
    counter = 0
    for gd in ghostDist:
        if newScaredTimes[counter] == 0:
            nearestGhostDistance = gd
        if newScaredTimes[counter] > gd:
            ghostEva += 100 - gd

        counter+=1
    if nearestGhostDistance == float("inf"):
        nearestGhostDistance = 0
    ghostEva += nearestGhostDistance

    minDist=0
    if(len(foodDist)>0):
        minDist=min(foodDist)

    # eva = currentGameState.getScore() + 250 / heuristic - 10 * currentGameState.getNumFood() - 2 * min(foodDist) - 0.75 * max(foodDist) + 3.5 * min(ghostDist) + 2 * len(currentGameState.getCapsules())
    eva = currentGameState.getScore() - 3 * heuristic - 2 * minDist + 2 * len(currentGameState.getCapsules()) - 10 * currentGameState.getNumFood() + 1 * ghostEva

    return eva




# Abbreviation
better = betterEvaluationFunction

