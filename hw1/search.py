# search.py
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

# "I affirm that I will not give or receive any unauthorized help on this academic activity, and that all work I submit is my own."
"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


# def dfsRecursive(now_state, visited_state, problem, trajectory):
#
#     if problem.isGoalState(now_state):
#         return True
#     next_step = problem.getSuccessors(now_state)
#     for node in next_step:
#         next_state = node[0]
#         next_direction = node[1]
#         if not next_state in visited_state:
#             visited_state.append(next_state)
#             trajectory.append(next_direction)
#             if dfsRecursive(next_state, visited_state, problem, trajectory):
#                 return True
#     trajectory.pop()
#
# def depthFirstSearchRecursive(problem):
#
#     now_state = problem.getStartState()
#     visited_state = [now_state]
#     trajectory = []
#     if not dfsRecursive(now_state, visited_state, problem, trajectory):
#         raise Exception('No Path to the Goal')
#     return trajectory


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    now_state = problem.getStartState()
    state_stack = util.Stack()
    state_list = set()
    state_stack.push((now_state, []))
    while (1):
        if state_stack.isEmpty():
            raise Exception('No Path to the Goal')
        now_state, last_trajectory = state_stack.pop()
        if problem.isGoalState(now_state):
            break
        if now_state not in state_list:
            state_list.add(now_state)
            next_step = problem.getSuccessors(now_state)
            for node in next_step:
                next_state = node[0]
                next_direction = node[1]
                if not next_state in state_list:
                    state_stack.push((next_state, last_trajectory + [next_direction]))
    return last_trajectory



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    now_state = problem.getStartState()
    state_queue = util.Queue()
    state_list = set()
    state_queue.push((now_state, []))
    while (1):
        if state_queue.isEmpty():
            raise Exception('No Path to the Goal')
        now_state, last_trajectory = state_queue.pop()
        if problem.isGoalState(now_state):
            break
        if now_state not in state_list:
            state_list.add(now_state)
            for nodes in problem.getSuccessors(now_state):
                next_state = nodes[0]
                next_direction = nodes[1]
                if next_state not in state_list:
                    trajectory = last_trajectory + [next_direction]
                    state_queue.push((next_state, trajectory))
    return last_trajectory


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    now_state = problem.getStartState()
    state_list = set()
    state_queue = util.PriorityQueue()
    state_queue.push((now_state, []), 0)
    while (1):
        if state_queue.isEmpty():
            raise Exception('No Path to the Goal')
        now_state, last_trajectory = state_queue.pop()
        if problem.isGoalState(now_state):
            break
        if now_state not in state_list:
            state_list.add(now_state)
            for nodes in problem.getSuccessors(now_state):
                next_state = nodes[0]
                next_direction = nodes[1]
                if next_state not in state_list:
                    trajectory = last_trajectory + [next_direction]
                    state_queue.push((next_state, trajectory), problem.getCostOfActions(trajectory))
    return last_trajectory


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
