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
        self.state_list = self.mdp.getStates()
        [self.values[state] for state in self.state_list]
        self.runValueIteration()




    def runValueIteration(self):
        for iter in range(self.iterations):
            values = util.Counter()
            for state in self.state_list:
                values[state] = self.getMaxQValues(state)
            self.values = values


    def getMaxQValues(self, state):

        if self.mdp.isTerminal(state):
            return 0
        qValues = util.Counter()
        actions = self.mdp.getPossibleActions(state)
        for action in actions:
            qValues[action] = self.getQValue(state, action)

        return qValues[qValues.argMax()]



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

        successors = self.mdp.getTransitionStatesAndProbs(state, action)
        qValues = 0
        for node in successors:
            next_state = node[0]
            prob = node[1]
            reward = self.mdp.getReward(state, action, next_state)
            qValues += prob*(reward + self.discount * self.values[next_state])
        return qValues

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        qValues = util.Counter()
        for action in actions:
            if action == 'exit':
                return action
            qValues[action] = self.getQValue(state, action)
        return qValues.argMax()


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
        for iter in range(self.iterations):
            state = self.state_list[iter % len(self.state_list)]
            self.values[state] = self.getMaxQValues(state)




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
        predecessor_list = self.creatPredecessors()
        queue = util.PriorityQueue()
        for state in self.state_list:
            if not self.mdp.isTerminal(state):
                diff = abs(self.values[state] - self.getMaxQValues(state))
                queue.update(state, -diff)
        for iter in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()
            self.values[state] = self.getMaxQValues(state)
            predecessors = predecessor_list[state]
            for predecessor in predecessors:
                diff = abs(self.values[predecessor]-self.getMaxQValues(predecessor))
                if diff > self.theta:
                    queue.update(predecessor, -diff)

    def creatPredecessors(self):
        predecessors = util.Counter()
        for state in self.state_list:
            predecessors[state] = self.getPredecessor(state)
        return predecessors

    def getPredecessor(self, goal):
        predecessor = set()
        for state in self.state_list:
            if self.mdp.isTerminal(state):
                continue
            if state in predecessor:
                continue
            searched_state = self.depthFirstSearch(state, goal, predecessor)
            if searched_state is not None:
                [predecessor.add(n) for n in searched_state]
        return predecessor

    def depthFirstSearch(self, state, goal, predecessor):
        stack = util.Stack()
        check_list = set()
        stack.push((state, [state]))
        check_list.add(state)
        while True:
            if stack.isEmpty():
                return None
            now_state, last_state = stack.pop()
            actions = self.mdp.getPossibleActions(now_state)
            for action in actions:
                successors = self.mdp.getTransitionStatesAndProbs(now_state, action)
                for nodes in successors:
                    next_state = nodes[0]
                    prob = nodes[1]
                    if prob > 0:
                        if (next_state == goal) or (next_state in predecessor):
                            return last_state
                        if next_state not in check_list:
                            stack.push((next_state, last_state + [next_state]))
                            check_list.add(next_state)





