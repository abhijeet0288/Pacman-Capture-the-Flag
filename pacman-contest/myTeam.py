# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
import util
from game import Actions
from util import nearestPoint
from game import GameStateData


DEBUG = False

  # Agents

  
class PacmanLayout():

    def __init__(self, startState, states, walls):
      self.startState = startState
      self.states = states
      self.walls = walls

      allowedActions = util.Counter()
      for state in states:
        x, y = state
        if (x - 1, y) in states and (x - 1, y) not in self.walls:
            allowedActions[(state, Directions.WEST)] = (x - 1, y)
        if (x + 1, y) in states and (x + 1, y) not in self.walls:
            allowedActions[(state, Directions.EAST)] = (x + 1, y)
        if (x, y - 1) in states and (x, y - 1) not in self.walls:
            allowedActions[(state, Directions.SOUTH)] = (x, y - 1)
        if (x, y + 1) in states and (x, y + 1) not in self.walls:
            allowedActions[(state, Directions.NORTH)] = (x, y + 1)

      self.possibleActions = allowedActions
      self.rewards = util.Counter()

    def getStartState(self):

      return self.startState

    def getStates(self):

      return self.states

    def addReward(self, state, reward):
        self.rewards[state] += reward

    def addRewardToStateAndNeighbouringStates(self, state, reward):
        x, y = state
        self.rewards[state] += reward
        self.rewards[(x - 1, y)] += reward / 2.25
        self.rewards[(x + 1, y)] += reward / 2.25
        self.rewards[(x, y - 1)] += reward / 2.25
        self.rewards[(x, y + 1)] += reward / 2.25

    def getTransitionState(self, state, action):

     return self.possibleActions[(state, action)]

    def getPossibleActions(self, state):
      
      return [element[1] for element in self.possibleActions.keys() if element[0] == state]

    def getReward(self, state, action, nextState):

     if action == Directions.STOP:
            return -15
     return self.rewards[nextState]

class OnPolicyLearningAgent():

    def __init__(self, pacmanLayout, discountFactor, learningRate, startState, numberOfIterations, gameState):

      self.pacmanLayout = pacmanLayout
      self.discountFactor = discountFactor
      self.learningRate = learningRate
      self.numberOfIterations = numberOfIterations
      self.startState = startState
      self.values = util.Counter()
      self.gameState = gameState
      
      epsilon = random.random()
      
      for iteration in range(0, numberOfIterations):
        
        state = startState
        iterationValues = util.Counter()
        visitedStates = []
        nextAction = None
        notVisitedStates = pacmanLayout.getStates()
        nextState = ()
        previousActions = []

        for state in pacmanLayout.getStates():
            visitedStates.append(state)
            actionValues = util.Counter()
            nextActionValues = util.Counter()
            
            if nextState:

                state = nextState

            if nextAction == None:

                for action in pacmanLayout.getPossibleActions(state):

                    actionValues[action] = self.computeQValueForAction(state, action)

                if len(actionValues.keys()) == 0:
                    return None
                bestActions = max(actionValues.items(), key = lambda x : x[1])
                bestAction = bestActions[0]
                
                iterationValues[state] = self.computeQValueForAction(state, bestAction)

                nextState = pacmanLayout.getTransitionState(state, bestAction)
                
                for action in pacmanLayout.getPossibleActions(nextState):

                    nextActionValues[action] = self.computeQValueForAction(nextState, action)

                if len(nextActionValues.keys()) == 0:
                    return None
                nextActions = max(nextActionValues.items(), key = lambda x : x[1])
                nextAction = nextActions[0]

            else:

                bestAction = nextAction
                
                iterationValues[state] = self.computeQValueForAction(state, bestAction)
                
                nextState = pacmanLayout.getTransitionState(state, bestAction)
                
                for action in pacmanLayout.getPossibleActions(nextState):

                    nextActionValues[action] = self.computeQValueForAction(nextState, action)

                if len(nextActionValues.keys()) == 0:
                    return None
                nextActions = max(nextActionValues.items(), key = lambda x : x[1])
                nextAction = nextActions[0]
                
        self.values = iterationValues.copy()


    def computeQValueForAction(self, state, action):
      
      qValue = 0
      nextState = self.pacmanLayout.getTransitionState(state, action)
      reward = self.pacmanLayout.getReward(state, action, nextState)
      qValue = self.values[state] + self.learningRate * (reward + self.discountFactor * self.values[nextState] - self.values[state])

      return qValue

    def getValue(self, state):

      return self.values[state]

    def getNextAction(self, state):

      bestActionValues = util.Counter()

      for action in self.pacmanLayout.getPossibleActions(state):
        
        bestActionValues[action] = self.computeQValueForAction(state, action)
      
      if len(bestActionValues.keys()) == 0:
          return None
      bestActions = max(bestActionValues.items(), key = lambda x : x[1])
      bestAction = bestActions[0]

      return bestAction


      # ------------------------------------------------------------


  #################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
                first = 'OffensiveReflexAgent', second = 'ReflexCaptureAgentForDefense'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


class ReflexCaptureAgent(CaptureAgent):

    def registerInitialState(self, gameState):
      
      self.initialState = gameState.getAgentPosition(self.index)
      CaptureAgent.registerInitialState(self, gameState)
      
      self.enemyIndices = self.getOpponents(gameState)
      
      self.previousActions = []

      self.initialCapsules = self.getCapsules(gameState)

      self.walls = set(gameState.data.layout.walls.asList())
      
      self.maxx = max([wall[0] for wall in self.walls])
      self.maxy = max([wall[1] for wall in self.walls])

      if gameState.isOnRedTeam(self.index):
        self.sign = 1
      else:
        self.sign = -1

      self.gameStartTime = time.time()

      self.xHomeBoundary = self.initialState[0] + ((self.maxx // 2 - 1) * self.sign)
      cells = [(self.xHomeBoundary, y) for y in range(1, self.maxy)]
      self.homeBoundaryCellsExcludingWalls = [element for element in cells if element not in self.walls]

      self.enemyXHomeBoundary = self.xHomeBoundary + self.sign
      cells = [(self.enemyXHomeBoundary, y) for y in range(1, self.maxy)]
      self.enemyHomeBoundaryCellsExcludingWalls = [element for element in cells if element not in self.walls]

      cellsExcludingWalls = self.getGridExcludingWalls(1, 1, self.maxx, self.maxy)
      self.numberOfLegalActions = util.Counter()

      for cell in cellsExcludingWalls:
        x, y = cell
        if (x - 1, y) in cellsExcludingWalls:
            self.numberOfLegalActions[cell] += 1
        if (x + 1, y) in cellsExcludingWalls:
            self.numberOfLegalActions[cell] += 1
        if (x, y - 1) in cellsExcludingWalls:
            self.numberOfLegalActions[cell] += 1
        if (x, y + 1) in cellsExcludingWalls:
            self.numberOfLegalActions[cell] += 1 

    def getGridExcludingWalls(self, xmin, ymin, xmax, ymax):

      xmin = int(max(1, xmin))
      ymin = int(max(1, ymin))
      xmax = int(min(self.maxx, xmax))
      ymax = int(min(self.maxy, ymax))

      cells = set()

      for x in range(xmin, xmax + 1):

        for y in range(ymin, ymax + 1):

          cells.add((x,y))

      return cells.difference(self.walls)

    def getDistanceHome(self, currentPosition):
        x, y = currentPosition

        if (self.homeBoundaryCellsExcludingWalls[0][0] - x) * self.sign > 0:
            return 0
        distances = [self.distancer.getDistance(currentPosition, cell) for cell in self.homeBoundaryCellsExcludingWalls]

        return min(distances)

    def isHomeTerritory(self, cell):
        x, y = cell
        xInitial = self.initialState[0]
        xBoundary = self.xHomeBoundary
        return xInitial <= x <= xBoundary or xBoundary <= x <= xInitial

    def assignRewards(self, grid, pacmanLayout, rewardType, currentPosition, rewardPosition):
      rewards = []
      distanceToRewardPosition = self.distancer.getDistance(rewardPosition, currentPosition)
      for cell in grid:
          distance = self.distancer.getDistance(cell, rewardPosition)
          if distance <= distanceToRewardPosition:
              reward = rewardType / max(float(distance), .5)
              rewardRange = reward / 5.
              reward += random.uniform(-rewardRange, rewardRange)
              pacmanLayout.addReward(cell, reward)
              rewards.append((currentPosition, cell, distance, reward))
      return rewards

    def chooseAction(self, gameState):

      startTime = time.time()

      proximity = 7
      rewardForFood = 0.15
      rewardForGhostNearby = -1.5
      rewardForCapsule = 1
      rewardForCapsuleWhenGhostNearby = 1
      rewardForGettingTrapped = -0.15
      rewardForReturningHome = 1.25
      rewardForFoodWhenNoCapsule = 0.20
      rewardForFoodAfterCapsule = 0.20
      goalStates = []
      nonGoalStates = []
      midHomeBoundaryCellsExcludingWalls = []
      enemyInHomeTerritory = False

      currentState = gameState.getAgentState(self.index)
      currentPosition = currentState.getPosition()

      distanceHome = self.getDistanceHome(currentPosition)
      currentX, currentY = currentPosition


      nearbyGrid = self.getGridExcludingWalls(currentX - proximity, currentY - proximity, currentX + proximity, currentY + proximity)
      nearbyGrid = {cell for cell in nearbyGrid if self.distancer.getDistance(currentPosition, cell) <= proximity}
      pacmanLayout = PacmanLayout(currentPosition, nearbyGrid, set(gameState.data.layout.walls.asList()))

      foodPositions = self.getFood(gameState).asList()
      foodRemaining = len(foodPositions)

      if foodRemaining > 2 and self.getCapsules(gameState):

        for foodPosition in foodPositions:
          self.assignRewards(nearbyGrid, pacmanLayout, rewardType = rewardForFood, currentPosition = currentPosition, rewardPosition = foodPosition)
      if foodRemaining > 2 and self.initialCapsules == 0:
        for foodPosition in foodPositions:
          self.assignRewards(nearbyGrid, pacmanLayout, rewardType = rewardForFoodWhenNoCapsule, currentPosition = currentPosition, rewardPosition = foodPosition)
      if foodRemaining > 2 and not self.getCapsules(gameState) and self.initialCapsules != 0:
        for foodPosition in foodPositions:
          self.assignRewards(nearbyGrid, pacmanLayout, rewardType = rewardForFoodAfterCapsule, currentPosition = currentPosition, rewardPosition = foodPosition)
        

      enemyNearby = False
      ghostNearby = False
      enemies = []

      for state in self.getOpponents(gameState):
        enemyState = gameState.getAgentState(state)
        enemyPosition = enemyState.getPosition()

        if enemyPosition:
            enemyDistance = self.distancer.getDistance(currentPosition, enemyPosition)
            if enemyDistance < 6 and enemyState.isPacman:
                enemyNearby = True
                enemies.append((enemyState, enemyPosition))
            if enemyDistance < 6 and not enemyState.isPacman:
                ghostNearby = True
                enemies.append((enemyState, enemyPosition))

      if ghostNearby:

        minimumEnemyDistance = 5
        enemyScaredTimer = min([enemy.scaredTimer for enemy, _ in enemies])

        for enemyState, enemyPosition in enemies:
            if enemyState.scaredTimer > 3:
                continue
            enemyDistance = self.distancer.getDistance(currentPosition, enemyPosition)
            reward = rewardForGhostNearby * foodRemaining * (proximity + 1 - enemyDistance)

            pacmanLayout.addRewardToStateAndNeighbouringStates(enemyPosition, reward)
            minimumEnemyDistance = min(minimumEnemyDistance, enemyDistance)

        for cell in nearbyGrid:
            if self.isHomeTerritory(cell) or enemyScaredTimer > 10:
                    continue
            
            numberOfLegalActions = self.numberOfLegalActions[cell]

            if numberOfLegalActions == 1 and minimumEnemyDistance <= 3:
                trapPositionReward = rewardForGettingTrapped * (7 - minimumEnemyDistance)
                pacmanLayout.addRewardToStateAndNeighbouringStates(cell, trapPositionReward)
            
            if numberOfLegalActions == 2 and minimumEnemyDistance <= 3:
                trapPositionReward = rewardForGettingTrapped * (7 - enemyDistance) / 2
                pacmanLayout.addRewardToStateAndNeighbouringStates(cell, trapPositionReward)

            for capsulePosition in self.getCapsules(gameState):
                self.assignRewards(nearbyGrid, pacmanLayout, rewardType = rewardForCapsuleWhenGhostNearby, currentPosition = currentPosition, rewardPosition = capsulePosition)


            eatenFood = min(currentState.numCarrying, 8)
            reward = rewardForReturningHome * max(eatenFood - 1, 0) / 10
            self.assignRewardsForReturningHome(nearbyGrid, pacmanLayout, reward, currentPosition)


      for capsulePosition in self.getCapsules(gameState):
          self.assignRewards(nearbyGrid, pacmanLayout, rewardType = rewardForCapsuleWhenGhostNearby, currentPosition = currentPosition, rewardPosition = capsulePosition)

      if currentState.numCarrying > 4 and self.getDistanceHome(currentPosition) < 8:
            self.assignRewardsForReturningHome(nearbyGrid, pacmanLayout, rewardForReturningHome, currentPosition)

      if currentState.numCarrying > 8 and self.getDistanceHome(currentPosition) < 12:
            self.assignRewardsForReturningHome(nearbyGrid, pacmanLayout, rewardForReturningHome, currentPosition)

      foodRemainingInHomeTerritory = self.getFoodYouAreDefending(gameState).asList()

      for cell in self.homeBoundaryCellsExcludingWalls:
          x, y = cell
          distance = abs((gameState.data.layout.height / 2) - y)
          if distance <= 1:
              midHomeBoundaryCellsExcludingWalls.append(cell)

      currentTime = time.time()

      learner = OnPolicyLearningAgent(pacmanLayout, discountFactor = 0.75, learningRate = 0.6, startState = self.initialState, numberOfIterations = 100, gameState = gameState)

      nextAction = learner.getNextAction(currentPosition)

      reverse = False
      if self.previousActions:
         if len(self.previousActions) >= 5:
             size = len(self.previousActions)
             lastFiveActions = self.previousActions[size - 5: size]
             if lastFiveActions[0] == Directions.NORTH and lastFiveActions[1] == Directions.SOUTH and lastFiveActions[2] == Directions.NORTH and lastFiveActions[3] == Directions.SOUTH:
                 reverse = True
             if lastFiveActions[0] == Directions.EAST and lastFiveActions[1] == Directions.WEST and lastFiveActions[2] == Directions.EAST and lastFiveActions[3] == Directions.WEST:   
                 reverse = True
         self.previousActions.append(nextAction)
      else:
         self.previousActions.append(nextAction)

      if reverse == True:

            nextAction = random.choice(pacmanLayout.getPossibleActions(currentPosition))

      timeElapsed = time.time() - startTime

      return nextAction


    def assignRewardsForReturningHome(self, grid, pacmanLayout, rewardType, currentPosition):
        for targetCell in self.homeBoundaryCellsExcludingWalls:
            rewards = self.assignRewards(grid, pacmanLayout, rewardType=rewardType, currentPosition=currentPosition, rewardPosition=targetCell)
        return rewards

    def final(self, gameState):
      # print "Warnings count:", self.warnings
      CaptureAgent.final(self, gameState)

    def getSuccessor(self, gameState, action):
      """
      Finds the next successor which is a grid position (location tuple).
      """
      successor = gameState.generateSuccessor(self.index, action)
      pos = successor.getAgentState(self.index).getPosition()
      if pos != nearestPoint(pos):

          return successor.generateSuccessor(self.index, action)
      else:
          return successor

    def evaluate(self, gameState, action):
      """
      Computes a linear combination of features and feature weights
      """
      features = self.getFeatures(gameState, action)
      weights = self.getWeights(gameState, action)
      return features * weights

    def getFeatures(self, gameState, action):
      """
      Returns a counter of features for the state
      """
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)
      features['successorScore'] = self.getScore(successor)
      return features

    def getWeights(self, gameState, action):
      """
      Normally, weights do not depend on the gamestate.  They can be either
      a counter or a dictionary.
      """
      return {'successorScore': 1.0}

# -----------------------------------------------------------------------------------------------------------

class OffensiveReflexAgent(ReflexCaptureAgent):

    def getFeatures(self, gameState, action):
        features = util.Counter()
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

  
# -----------------------------------------------------------------------------------------------------------

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        return None

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

# -----------------------------------------------------------------------------------------------------------

class ReflexCaptureAgentForDefense(CaptureAgent):

    def registerInitialState(self, gameState):

        self.initialPosition = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        enemyPacmanPositions = []
        

        self.walls = set(gameState.data.layout.walls.asList())

        self.maxx = max([wall[0] for wall in self.walls])
        self.maxy = max([wall[1] for wall in self.walls])

        self.midY = (gameState.data.layout.height / 2) - 2
        self.initialFood = self.getFoodYouAreDefending(gameState).asList()
        self.initialFoodCount = len(self.initialFood)
        

        self.previousFoodStates = []
        self.previousFoodStates.append(self.initialFood)
        self.lastEatenFood = []
        self.lastEatenFood.append(random.choice(self.initialFood))

        if gameState.isOnRedTeam(self.index):
            self.sign = 1
        else:
            self.sign = -1

        self.xHomeBoundary = self.initialPosition[0] + ((self.maxx // 2 - 1) * self.sign)

        for y in range(1, self.maxy):

            homeBoundaryCells = [(self.xHomeBoundary, y)]

        for cell in homeBoundaryCells:
            if cell not in self.walls:
                self.homeBoundaryCellsExcludingWalls = [cell]

        cellsExcludingWalls = self.getGridExcludingWalls(1, 1, self.maxx, self.maxy)

        self.numberOfLegalActions = util.Counter()

        for cell in cellsExcludingWalls:
            x, y = cell
            if (x - 1, y) in cellsExcludingWalls:
                self.numberOfLegalActions[cell] += 1
            if (x + 1, y) in cellsExcludingWalls:
                self.numberOfLegalActions[cell] += 1
            if (x, y - 1) in cellsExcludingWalls:
                self.numberOfLegalActions[cell] += 1
            if (x, y + 1) in cellsExcludingWalls:
                self.numberOfLegalActions[cell] += 1 

    def getGridExcludingWalls(self, xmin, ymin, xmax, ymax):

      xmin = int(max(1, xmin))
      ymin = int(max(1, ymin))
      xmax = int(min(self.maxx, xmax))
      ymax = int(min(self.maxy, ymax))

      cells = set()

      for x in range(xmin, xmax + 1):

        for y in range(ymin, ymax + 1):

          cells.add((x,y))

      return cells.difference(self.walls)
        

    def isPacman(self, gameState, index):

        position = gameState.getAgentPosition(index)
        x, y = position

        if position is None:
            return False

        if gameState.isOnRedTeam(index) and x >= gameState.getWalls().width / 2:
            return True
        if not gameState.isOnRedTeam(index) and x < gameState.getWalls().width / 2:
            return True
        return False

    def isGhost(self, gameState, index):

        position = gameState.getAgentPosition(index)
        x, y = position

        if position is None:
            return False
        
        if gameState.isOnRedTeam(index) and x < gameState.getWalls().width / 2:
            return True
        if not gameState.isOnRedTeam(index) and x >= gameState.getWalls().width / 2:
            return True
        return False

    def isScared(self, gameState, index):

        if gameState.data.agentStates[index].scaredTimer:
            isScared = True
        else:
            isScared = False
        
        return isScared

    def closestFoodLocation(self, gameState, currentPosition):

        homeFood = self.getFoodYouAreDefending(gameState).asList()
        foodDistance = util.Counter()
        closestFood = []
        minimumDistance = 1000
        for food in homeFood:
            if food != currentPosition:
                foodDistance[food] = self.distancer.getDistance(currentPosition, food)
                if foodDistance[food] < minimumDistance:
                    minimumDistance = foodDistance[food]
                    closestFood = food
        return closestFood

    def getDefenseGrid(self, gameState):

        cells = set()

        if gameState.isOnRedTeam(self.index):
            cells = self.getGridExcludingWalls(1, 1, self.xHomeBoundary, self.maxy)

        if not gameState.isOnRedTeam(self.index):
            cells = self.getGridExcludingWalls(self.xHomeBoundary, 1, self.maxx, self.maxy)
        
        return cells


    def aStarSearch(self, currentPosition, goalStates, nonGoalStates, gameState):

        from util import PriorityQueue

        frontier = PriorityQueue()

        actions = []
        closed = []
        evaluation = 0
        cost = 0
        minimum = 10000
        if currentPosition in goalStates:
            return None
        for positions in self.defensivePositions:
            distance = self.getMazeDistance(positions, currentPosition)
            if distance < minimum:
                minimum = distance
        frontier.push((currentPosition, []), minimum)
        while (True):
            
            if frontier.isEmpty():
                return None
            

            node, path = frontier.pop()

            if node not in closed:
                closed.append(node)
                if node in goalStates:
                    return path
                for successor in self.getSuccessors(node):
                    actions = path + [successor[1]]
                    cost += 1
                    evaluation = cost + self.heuristic(gameState, successor[0], goalStates, nonGoalStates)
                    frontier.push((successor[0], actions), evaluation)
            
        util.raiseNotDefined()


    def heuristic(self, gameState, currentPosition, goalStates, nonGoalStates):

        minimum = 10000
        for position in goalStates:
            distance = self.getMazeDistance(position, currentPosition)
            if distance < minimum:
                minimum = distance
        return minimum
                

    def getSuccessors(self, state):

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if (nextx, nexty) not in self.walls:
                nextState = (nextx, nexty)
                cost = 1
                successors.append( ( nextState, action, cost) )

        return successors

    def attackQvalue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def computeActionFromQValues(self, gameState):
        """
          compute the next best action based on the Q values of the baseline Agent
        """
        bestValue = -999999
        bestActions = None
        for action in gameState.getLegalActions(self.index):
            # for each action in the legal actions get the maximum Q value
            value = self.attackQvalue(gameState, action)
            if (DEBUG):
                print
                "ACTION: " + action + "           QVALUE: " + str(value)
            if value > bestValue:
                bestActions = [action]
                bestValue = value
            elif value == bestValue:
                bestActions.append(action)
        if bestActions == None:
            return Directions.STOP  # If no legal actions return None
        return random.choice(bestActions)  # Else choose one of the best actions randomly

    def chooseAction(self, gameState):

        startTime = time.time()
        proximity = 8

        currentState = gameState.getAgentState(self.index)
        currentPosition = currentState.getPosition()
        currentX, currentY = currentPosition

        enemyIndices = self.getOpponents(gameState)
        enemies = []
        attackPacman = []
        avoidPacman = []
        avoidGhost = []
        foodRemaining = set()
        foodRemainingLastTime = set()
        foodEaten = []
        goalStates = []
        nonGoalStates = []
        enemyInHomeTerritory = False

        for index in enemyIndices:
            enemyState = gameState.getAgentState(index)
            enemyPosition = enemyState.getPosition()
            if enemyPosition:

                enemyDistance = self.distancer.getDistance(currentPosition, enemyPosition)
                if self.isPacman(gameState, index) and self.isGhost(gameState, self.index):
                    attackPacman.append((enemyState, enemyPosition))
                if self.isPacman(gameState, index) and self.isScared(gameState, self.index):
                    avoidPacman.append((enemyState, enemyPosition))
                if self.isGhost(gameState, index) and self.isPacman(gameState, self.index):
                    avoidGhost.append((enemyState, enemyPosition))
        
        if gameState.isOnRedTeam(self.index):

            defenseCells = []
            for boundary in range(0, 2):
                for cell in self.getDefenseGrid(gameState):
                    x, y = cell
                    if x == ((gameState.data.layout.width / 2) - 1) - boundary:
                        if y >= (((gameState.data.layout.height / 2) - 1) - boundary) and y <= (((gameState.data.layout.height / 2) - 1) + boundary):
                            defenseCells.append(cell)
            self.defensivePositions = defenseCells

        if not gameState.isOnRedTeam(self.index):
            
            defenseCells = []
            for boundary in range(0, 2):
                for cell in self.getDefenseGrid(gameState):
                    x, y = cell
                    if x == ((gameState.data.layout.width / 2) + 1) + boundary:
                        if y >= (((gameState.data.layout.height / 2) - 1) - boundary) and y <= (((gameState.data.layout.height / 2) - 1) + boundary):
                            defenseCells.append(cell)
            self.defensivePositions = defenseCells

        if len(avoidGhost) > 0:
            for enemyState, enemyPosition in avoidGhost:
                nonGoalStates.append(enemyPosition)

        for index in enemyIndices:
            if gameState.getAgentState(index).isPacman:
                enemyInHomeTerritory = True

        foodRemaining = set(self.getFoodYouAreDefending(gameState).asList())
        foodRemainingLastTime = set(self.previousFoodStates.pop())

        if len(foodRemaining) != len(foodRemainingLastTime):
            foodEaten = foodRemainingLastTime.difference(foodRemaining)

        lastFood = self.lastEatenFood.pop()
        if len(foodEaten) > 0:
            for foodPosition in foodEaten:
                self.lastEatenFood.append(foodPosition)
        else:
            self.lastEatenFood.append(lastFood)
        

        if self.isScared(gameState, self.index):
            for index in enemyIndices:
                enemyState = gameState.getAgentState(index)
                enemyPosition = enemyState.getPosition()
                if enemyPosition:
                    nonGoalStates.append(enemyPosition)


        if self.isScared(gameState, self.index):

            if enemyInHomeTerritory and gameState.getAgentState(self.index).scaredTimer < 5:

                if len(attackPacman) > 0:
                    for enemyState, enemyPosition in attackPacman:

                        goalStates.append(enemyPosition)

                else:
                    if len(foodEaten) > 0:
                        for foodPosition in foodEaten:
                            goalStates.append(foodPosition)
                            goalStates.append(self.closestFoodLocation(gameState, foodPosition))
                    else:
                        goalStates.append(lastFood)

            elif gameState.getAgentState(self.index).scaredTimer > 16:
                    
                    foodRemaining = self.getFood(gameState).asList()
                    capsules = self.getCapsules(gameState)
                    for food in foodRemaining:
                        goalStates.append(food)
                    for capsule in capsules:
                        goalStates.append(capsule)

            elif gameState.getAgentState(self.index).scaredTimer < 16 and gameState.getAgentState(self.index).scaredTimer > 5:
                    
                    for state in self.defensivePositions:
                
                        goalStates.append(state)
                    

        else:

            if enemyInHomeTerritory and not gameState.getAgentState(self.index).isPacman:

                if len(attackPacman) > 0:
                    for enemyState, enemyPosition in attackPacman:

                        goalStates.append(enemyPosition)

                else:
                    if len(foodEaten) > 0:
                        for foodPosition in foodEaten:
                            goalStates.append(foodPosition)
                            goalStates.append(self.closestFoodLocation(gameState, foodPosition))
                    else:
                        goalStates.append(lastFood)

            else:
                for state in self.defensivePositions:
                
                    goalStates.append(state)

        self.previousFoodStates.append(foodRemaining)
        
        if goalStates:

            actionPath = self.aStarSearch(currentPosition, goalStates, nonGoalStates, gameState)

        else:

            actionPath = None

        if actionPath:

            nextAction =  actionPath[0]

        else:

            nextAction = self.computeActionFromQValues(gameState)

        timeElapsed = time.time() - startTime

        return nextAction


    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


