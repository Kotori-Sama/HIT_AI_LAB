# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined,Counter


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        
        tot=self.total() 
        if tot!=0:
            for key in self.keys():
                self[key]=self[key]/tot
        
        # raiseNotDefined()

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"
        
        if len(self.keys())==0:
            raise ValueError('概率分布为空！')
        
        self.normalize() # 计算概率分布
        
        rand_value = random.uniform(0, 1) # 均匀分布中采样
        sum_pro = 0

        for key, prob in self.items():
            sum_pro += prob 
            if rand_value <= sum_pro: # 找采样点
                return key
        
        # raiseNotDefined()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        '''
        if the ghost's position is the jail position, then the observation is None with probability 1, 
        and everything else with probability 0.
        '''
        if ghostPosition == jailPosition: #在监狱里
            if noisyDistance == None:
                return 1.0  
            else:
                return 0.0
        elif noisyDistance == None: # 不在监狱里，但是distance reading是None，那么ghost一定在监狱里
            return 0.0  
        else: # 观察值不是None，这个时候ghost不可能在监狱里
            # manhattanDistance function to find the distance between Pacman's location and the ghost's location
            p_g_distance=manhattanDistance(pacmanPosition, ghostPosition) # trueDistance
            # P(noisyDistance | trueDistance)
            return busters.getObservationProbability(noisyDistance, p_g_distance)
        # raiseNotDefined()

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"
        
        #raiseNotDefined()
        
        # 使用 gameState.getPacmanPosition() 获取 Pacman 位置，使用 self.getJailPosition() 获取监狱位置。
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition = self.getJailPosition()
        
        for ghostPosition in self.allPositions:
            #P(observation | pacmanPosition, ghostPosition)
            observatiion_prob = self.getObservationProb(observation, pacmanPosition, ghostPosition, jailPosition)
            #P(observation | pacmanPosition, ghostPosition)*P( ghostPosition )
            self.beliefs[ghostPosition] *= observatiion_prob
        
        self.beliefs.normalize() # 计算概率分布

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"
        # my tips
        """
        Your agent has access to the action distribution for the ghost through self.getPositionDistribution .
        newPosDist = self.getPositionDistribution(gameState, oldPos)
        """
        newBeliefs = DiscreteDistribution() # setdefault将所有值设定为0，下面直接加就行
        
        for oldPos in self.allPositions:
            # 对每个oldPos都要计算出一个新位置的概率，然后求和
            newPosDist = self.getPositionDistribution(gameState, oldPos)
            for newPos,prob in newPosDist.items():
                # P( oldPos )*P( newPos | oldPos) 当前oldPos下新的位置分布
                newBeliefs[newPos] += self.beliefs[oldPos] * prob # 求和
        
        newBeliefs.normalize()
        self.beliefs = newBeliefs
        
        
        #raiseNotDefined()

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        numParticles = self.numParticles
        legalPositions = self.legalPositions
        # 根据合法位置的数量，将particle均匀分布
        particles = legalPositions * (numParticles // len(legalPositions)) 
        # 处理剩余的particle
        particles += legalPositions[:numParticles % len(legalPositions)]
        
        self.particles = particles
        # raiseNotDefined()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        
        # mytip
        '''
         use the function self.getObservationProb to find the probability of an observation 
         given Pacman's position, 
         a potential ghost position, 
         and the jail position.
        '''
        pacmanPosition = gameState.getPacmanPosition()
        jailPosition=self.getJailPosition()
        weights = DiscreteDistribution()
        # 计算每个particle的权重
        for ghostPosition in self.particles :
            # P(observation | pacmanPosition, ghostPosition)
            observation_prob=self.getObservationProb(observation, pacmanPosition, ghostPosition, jailPosition)
            weights[ghostPosition] += observation_prob
        
        if weights.total() == 0:# 如果权重都是0
            self.initializeUniformly(gameState)# 重新初始化
        else:
            weights.normalize()
            self.beliefs = weights # 归一化后的权重作为新的置信分布
            self.particles = [weights.sample() for _ in range(self.numParticles)] # 在新的分布上重采样
        
        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        
        
        newParticles = []
        for oldPos in self.particles:
            newPosDist = self.getPositionDistribution(gameState, oldPos) # 获取每个Particle的新位置分布
            newParticles.append(newPosDist.sample())  # 重采样
            
        self.particles = newParticles
        # raiseNotDefined()

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"
        
        beliefDistribution = DiscreteDistribution()
        
        for particle in self.particles:
            beliefDistribution[particle] += 1
            
        beliefDistribution.normalize() #计数之后归一化就行
        return beliefDistribution
        #raiseNotDefined()


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        legalPositions = self.legalPositions # ghost可能的位置
        numParticles = self.numParticles
        numGhosts = self.numGhosts
        # itertools.product to get an implementation of the Cartesian product
        # 对ghost位置作笛卡尔积，得到所有可能的ghost位置排列，这些排列就作为新的合法位置了
        GhostPositions = list(itertools.product(legalPositions, repeat=numGhosts))
        
        # 同样的方法：把legalPositions变成排列GhostPositions即可
        # 根据合法位置的数量，将particle均匀分布
        particles = GhostPositions * (numParticles // len(GhostPositions)) 
        # 处理剩余的particle
        particles += GhostPositions[:numParticles % len(GhostPositions)]
        
        self.particles = particles
        
        # raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        
        # mytips
        '''
        A correct implementation will weight and resample the entire list of particles 
        based on the observation of all ghost distances.
        
        get the jail position for a ghost, use self.getJailPosition(i) , 
        since now there are multiple ghosts each with their own jail positions.
        '''
        pacmanPosition = gameState.getPacmanPosition()
        weights = DiscreteDistribution()
        # 类似的，计算权重;只不过每个ghost都要计算
        for ghostPosition in self.particles:
            observation_prob=1
            for i in range(self.numGhosts):
                jailPosition = self.getJailPosition(i)
                # P(observation | pacmanPosition, ghostPosition)
                observation_prob*=self.getObservationProb(observation[i], pacmanPosition, ghostPosition[i], jailPosition)
            weights[ghostPosition] += observation_prob
                
        if weights.total() == 0:# 如果权重都是0
            self.initializeUniformly(gameState)# 重新初始化
        else:
            weights.normalize()
            self.beliefs = weights # 归一化后的权重作为新的置信分布
            self.particles = [weights.sample() for _ in range(self.numParticles)] # 在新的分布上重采样
            
        
        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        # my tips
        """
        Then, assuming that i refers to the index of the ghost, 
        to obtain the distributions over new positions for that single ghost, 
        given the list ( prevGhostPositions ) of previous positions of all of the ghosts,
        """
        # 参考q7的代码，只是每个ghost不同而已
        
        # newParticles = []
        # for oldPos in self.particles:
        #     newPosDist = self.getPositionDistribution(gameState, oldPos) # 获取每个Particle的新位置分布
        #     newParticles.append(newPosDist.sample())  # 重采样
            
        # self.particles = newParticles
        
        
        newParticles = []
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            for i in range(self.numGhosts):
                newPosDist = self.getPositionDistribution(gameState, oldParticle, i, self.ghostAgents[i])
                newParticle[i] = newPosDist.sample()# 采样的到新的ghost位置
            
            #raiseNotDefined()

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist




# dist = DiscreteDistribution()
# dist['a'] = 1
# dist['b'] = 2
# dist['c'] = 2
# dist['d'] = 0

# N = 100000.0
# samples = [dist.sample() for _ in range(int(N))]

# # 输出样本在每个键上的比例
# print(round(samples.count('a') * 1.0 / N, 1))
# print(round(samples.count('b') * 1.0 / N, 1))
# print(round(samples.count('c') * 1.0 / N, 1))
# print(round(samples.count('d') * 1.0 / N, 1))

# if __name__ == '__main__':
#     dist = DiscreteDistribution()
#     dist['a'] = 1
#     dist['b'] = 2
#     dist['c'] = 2
#     dist['d'] = 0
#     dist.normalize()
#     print(list(sorted(dist.items())))

#     dist['e'] = 4
#     print(list(sorted(dist.items())))

#     empty = DiscreteDistribution()
#     empty.normalize()
#     print(empty)