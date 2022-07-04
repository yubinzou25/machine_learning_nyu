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


import mdp, util

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
        self.runValueIteration()

    def runValueIteration(self):
        # BEGIN SOLUTION
        states = self.mdp.getStates()
        for i in range(self.iterations):
            newValues = util.Counter()
            for state in states:
                newValues[state] = self.getBestActionAndValue(state)[1]
            self.values = newValues
        # END SOLUTION


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getBestActionAndValue(self, state):
        if self.mdp.isTerminal(state): return (None, 0)
        return max(((action, self.computeQValueFromValues(state, action)) for action in self.mdp.getPossibleActions(state)),
                   key = lambda x: x[1])

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        ret = 0
        for (next, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
            ret += prob * (self.mdp.getReward(state, action, next) +
                           self.discount * self.getValue(next))
        return ret

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        return self.getBestActionAndValue(state)[0]

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
        # BEGIN SOLUTION
        states = self.mdp.getStates()
        currIndex = 0
        for i in range(self.iterations):
            state = states[currIndex]
            if not self.mdp.isTerminal(state):
                self.values[state] = max(self.computeQValueFromValues(state, a) \
                                           for a in self.mdp.getPossibleActions(state))
            currIndex = (currIndex + 1) % len(states)
        # END SOLUTION

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
        # BEGIN SOLUTION
        states = self.mdp.getStates()
        preds = {}
        for state in states:
            for action in self.mdp.getPossibleActions(state):
                for (next, prob) in self.mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        preds.setdefault(next, set()).add(state)

        pq = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                value = max(self.computeQValueFromValues(state, a) \
                                for a in self.mdp.getPossibleActions(state))
                diff = abs(value - self.values[state])
                pq.update(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                break
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = max(self.computeQValueFromValues(state, a) \
                                             for a in self.mdp.getPossibleActions(state))
            for prevState in preds[state]:
                prevStateValue = max(self.computeQValueFromValues(prevState, a) \
                                         for a in self.mdp.getPossibleActions(prevState))
                diff = abs(prevStateValue - self.values[prevState])
                if diff > self.theta:
                    pq.update(prevState, -diff)
        # END SOLUTION

