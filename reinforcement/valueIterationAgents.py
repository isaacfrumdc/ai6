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


import mdp
import util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            newValues = self.values.copy()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                newVal = None
                for action in actions:
                    qval = self.getQValue(state, action)
                    if newVal == None or qval > newVal:
                        newVal = qval
                if newVal == None:
                    newVal = 0
                newValues[state] = newVal

            self.values = newValues

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
        "*** YOUR CODE HERE ***"

        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        qval = 0
        for nextState, prob in transitions:
            reward = self.mdp.getReward(state, action, nextState)
            expr = prob * (reward + self.discount * self.getValue(nextState))
            qval += expr

        return qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)

        maxQ = None
        bestAction = None
        for action in actions:
            qval = self.getQValue(state, action)
            if maxQ == None or qval > maxQ:
                maxQ = qval
                bestAction = action
        return bestAction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        state_index = 0
        states = self.mdp.getStates()
        for iteration in range(self.iterations):
            newValues = self.values.copy()
            state = states[state_index]
            actions = self.mdp.getPossibleActions(state)
            newVal = None
            for action in actions:
                qval = self.getQValue(state, action)
                if newVal == None or qval > newVal:
                    newVal = qval
            if newVal == None:
                newVal = 0
            newValues[state] = newVal

            self.values = newValues

            if state_index + 1 >= len(states):
                state_index = 0
            else:
                state_index += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        predecessors = self.getPredecessors()
        pq = self.getPriorityQueue()   

        for iteration in range(self.iterations):
            if pq.isEmpty():
                return
            s = pq.pop()
            if self.mdp.isTerminal(s):
                continue
            actions = self.mdp.getPossibleActions(s)
            qvalues = []
            for action in actions:
                qval = self.getQValue(s, action)
                qvalues.append(qval)
            self.values[s] = max(qvalues)

            for p in predecessors[s]:   
                if self.mdp.isTerminal(p):
                    continue
                actions = self.mdp.getPossibleActions(p)
                qvalues = []
                for action in actions:
                    qval = self.getQValue(p, action)
                    qvalues.append(qval)
                diff = abs(self.values[p] - max(qvalues))
                if diff > self.theta:
                    pq.update(p, -diff)

        
    # Create dictionary of sets of predecessors for each state
    def getPredecessors(self):

        states = self.mdp.getStates()
        predecessors = dict() 

        for state in states:
            predecessors[state] = set()

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, prob in transitions:
                    if prob > 0:
                        predecessors[nextState].add(state)

        return predecessors

    def getPriorityQueue(self):
        states = self.mdp.getStates()
        pq = util.PriorityQueue()

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            actions = self.mdp.getPossibleActions(state)
            qvalues = []
            for action in actions:
                qval = self.getQValue(state, action)
                qvalues.append(qval)
                diff = abs(self.values[state] - max(qvalues)) * -1
            pq.update(state, diff)

        return pq
