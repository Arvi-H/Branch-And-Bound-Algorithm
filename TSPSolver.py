#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT6':
	from PyQt6.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

from TSPClasses import *

import time
import copy
import numpy as np
import heapq

class States:
	def __init__(self):
		self.statePath = [] # list to store path of states
		self.costMatrix = None # cost matrix for states
		self.bound = 0 # bound cost for current path
		self.currentIndex = 0 # index tracking current state in path
	
    # Compares current state to another via bound cost and path length
	# Time complexity: O(1) | Space complexity: O(1)
	def __lt__(self, other):
		return (self.bound, len(self.statePath)) < (other.bound, len(other.statePath))

class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution,
		time spent to find solution, number of permutations tried during search, the
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		This is the entry point for the greedy solver, which you must implement for
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this
		algorithm</returns>
	'''

	def greedy(self, time_allowance=60.0):
		results = {}
		start_time = time.time()
		
		cities = self._scenario.getCities()
		ncities = len(cities)
		
		route = [cities[0]]
		unvisited = set(cities)
		unvisited.remove(route[0])
		
		while unvisited:
			current = route[-1]
			next_city = min(unvisited, key=lambda city: current.costTo(city))
			route.append(next_city)
			unvisited.remove(next_city)
			
		bssf = TSPSolution(route)
		
		end_time = time.time()
		
		results['cost'] = bssf.cost
		results['time'] = end_time - start_time
		results['count'] = 1
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		
		return results


	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution,
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints:
		max queue size, total number of states created, and number of pruned states.</returns>
	'''

	def branchAndBound( self, time_allowance=60.0 ):

		def citiesMatrix():
			cities = self._scenario.getCities()
			numCities = len(cities)
			matrix = np.zeros((numCities, numCities))

			for i in range(len(cities)):
				for j in range(len(cities)):
					matrix[i, j] = cities[i].costTo(cities[j])

			return matrix

		def calcBound(matrix):
			bound = 0;
			matrix_cpy = copy.deepcopy(matrix)

			for i in range (len(matrix_cpy)):
				smallest = np.min(matrix_cpy[i])
				if(math.isinf(smallest)):
					continue
				matrix_cpy[i] -= smallest
				bound += smallest

			for j in range (len(matrix_cpy)):
				smallest = np.min(matrix_cpy[:, j])
				if (math.isinf(smallest)):
					continue
				matrix_cpy[:, j] -= smallest
				bound += smallest

			return matrix_cpy, bound

		def expand(mainNode: States, unvisited: set, ) -> list:
			
			matrix_copy = copy.deepcopy(mainNode.costMatrix)

			expanded_nodes = []

			for i in unvisited:
				stateMatrix = copy.deepcopy(matrix_copy)
				stateMatrix[mainNode.currentIndex, :] = math.inf
				stateMatrix[:, i] = math.inf

				reduced_matrix, bound = calcBound(stateMatrix)

				stateNode = States()
				stateNode.costMatrix = reduced_matrix
				stateNode.bound = bound + mainNode.bound + mainNode.costMatrix[mainNode.currentIndex, i]
				stateNode.currentIndex = i
				stateNode.statePath = mainNode.statePath + [i]

				expanded_nodes.append(stateNode)

			return expanded_nodes

		def is_complete_solution(node : States) -> bool:
			if not node.statePath:
				return False

			if set(range(len(node.costMatrix))) != set(node.statePath):
				return False

			if node.statePath[0] != node.statePath[-1]:
				return False

			return True

		startTime = time.time()
		solution = 0
		maxQueue = 0
		states = 0
		pruned = 0

		start = np.array(citiesMatrix())
		startNode = States()
		startNode.currentIndex = 0
		startNode.statePath = [startNode.currentIndex]

		startNode.costMatrix, startNode.bound = calcBound((start))

		S = [startNode]

		BSSF: tuple[float, States] = (math.inf, States())

		while len(S) > 0:
			if time.time() - startTime > time_allowance:
				break

			maxQueue = max(maxQueue, len(S))

			P = heapq.heappop(S)

			if is_complete_solution(P) and P.bound < BSSF[0]:
				BSSF = (P.bound, P)
				solution += 1

			unvisited = set(range(len(start)))
			# Assuming P is an instance of Node
			if P.statePath is not None:
				unvisited.difference_update(P.statePath)

			if not unvisited:
				unvisited.add(0)

			if P.bound < BSSF[0]:
				for node in expand(P, unvisited):
					states += 1
					if math.isinf(node.bound) or node.bound > BSSF[0]:
						pruned += 1
						continue

					heapq.heappush(S, node)

			else:
				pruned += 1

		endTime = time.time()

		route = [self._scenario.getCities()[index] for index in BSSF[1].statePath[:-1]]


		return {
			'cost': BSSF[0],
			'time': endTime - startTime,
			'count': solution,
			'soln': TSPSolution(route) if len(BSSF[1].statePath) > 0 else None,
			'max': maxQueue,
			'total': states,
			'pruned': pruned
		}
	
	def fancy( self,time_allowance=60.0 ):
		pass