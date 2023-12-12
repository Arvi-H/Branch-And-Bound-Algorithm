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

	# Builds a TSP tour by progressively visiting the nearest unvisited city
	# Overall time complexity: O(N^2) | Overall space complexity: O(N)
	def greedy(self, time_allowance=60.0):
		results = {}
		start_time = time.time()
		
		# Initialize data structures
    	# Time: O(N) Space: O(N)
		cities = self._scenario.getCities()
		
		route = [cities[0]]
		unvisited = set(cities)
		unvisited.remove(route[0])
		
		while unvisited:
			current = route[-1]
			next_city = min(unvisited, key=lambda city: current.costTo(city))
			route.append(next_city)
			unvisited.remove(next_city)
			
		# Wrap up
    	# Time: O(N) Space: O(N)
		bssf = TSPSolution(route)
		
		end_time = time.time()
		
		# Populate return values
    	# Time: O(1) Space: O(1)
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
		
		# Retrieves a list of cities from the scenario and computes a matrix representing the cost to travel between each pair of cities.
		# Time complexity: O(n^2) | Two nested loops iterating over all cities to calculate the cost between each pair 
		# Space complexity: O(n^2) | Resulting matrix has dimensions n x n, and each element of the matrix stores the cost between two cities.
		def generateTravelCostMatrix():
			cities = self._scenario.getCities()

			# Inner list comprehension calculates the cost to travel from the current city to each other city 
			# Outer list comprehension creates a row for each city in the matrix
			return np.array([[city.costTo(other) for other in cities] for city in cities])

		#  Calculates the bound of a given matrix by iteratively subtracting the smallest element from each row and column, ensuring non-infinite values are considered.
		# Time Complexity: O(m * n) | Space Complexity: O(m * n)
		def calculateLowerBound(matrix):
			bound = 0;
			matrixCopy = copy.deepcopy(matrix)

			# Subtract the smallest element from each row
			# Time Complexity: O(m * n) | Space Complexity: O(m * n) for the deep copy of the matrix
			for i in range (len(matrixCopy)):
				smallestElement = np.min(matrixCopy[i])
				if(math.isinf(smallestElement)):
					continue
				matrixCopy[i] -= smallestElement
				bound += smallestElement

			# Subtract the smallest element from each column
			# Time Complexity: O(m * n) | Space Complexity: O(1)  
			for j in range (len(matrixCopy)):
				smallestElement = np.min(matrixCopy[:, j])
				if (math.isinf(smallestElement)):
					continue
				matrixCopy[:, j] -= smallestElement
				bound += smallestElement

			return matrixCopy, bound


		# Expands the given node to generate successor nodes
		# Time complexity: O(N^2) | Space complexity: O(N)
		def expand(mainNode: States, unvisited: set) -> list:

			# Copy cost matrix to avoid mutating main node's matrix
			# Time complexity: O(N^2) to copy matrix | Space complexity: O(N^2) for matrix copy
			matrix_copy = copy.deepcopy(mainNode.costMatrix)

			# List to store expanded nodes  
			# Space complexity: O(N)  
			expanded_nodes = []

			# Iterate over unvisited nodes
			# Time complexity: O(N) for iteration over N nodes
			for i in unvisited:
				
				# Create copy of cost matrix for new child state
				# Time complexity: O(N^2) | Space complexity: O(N^2)
				stateMatrix = copy.deepcopy(matrix_copy)
				
				# Update costs in child matrix
				# Time: O(N)
				stateMatrix[mainNode.currentIndex, :] = math.inf    
				stateMatrix[:, i] = math.inf

				# Calculate reduced cost lower bound  
				# Time complexity: O(N^3) | Space complexity: O(N^2)  
				reduced_matrix, bound = calculateLowerBound(stateMatrix)

				# Create new child state node
				# Time complexity: O(1) | Space complexity: O(1) 
				stateNode = States()
				
				# Set state properties
				# Time complexity: O(N) | Space complexity: O(N)
				stateNode.costMatrix = reduced_matrix  
				stateNode.bound = bound + mainNode.bound + mainNode.costMatrix[mainNode.currentIndex, i]
				stateNode.currentIndex = i
				stateNode.statePath = mainNode.statePath + [i]

				# Add to expanded nodes
				# Time complexity: O(1)
				expanded_nodes.append(stateNode)

			# Return list of expanded nodes 
			return expanded_nodes

		# Checks if a given node represents a complete solution to the path finding problem
		# Time Complexity: O(n) | Space Complexity: O(n)
		def isValidPath(node):
			# Check if state path is empty 
    		# Time Complexity: O(1) | Space Complexity: O(1)
			if not node.statePath: 
				return False

			# Get set of visited states  
			# Time Complexity: O(n) where n is length of statePath | Space Complexity: O(n)  
			visited = set(node.statePath)


			# Check that 1. All states visited and 2. Path starts and ends at same state
    		# Time Complexity: O(1) | Space Complexity: O(1)
			return (len(visited) == len(node.costMatrix) and set(range(len(node.costMatrix))) == visited and node.statePath[0] == node.statePath[-1])

		startTime = time.time()
		solution = 0
		maxQueue = 0
		states = 0
		pruned = 0

		# Initialize start state
		# Time Complexity: O(N^3) for cost calculation | Space Complexity: O(N^2) for state  
		start = np.array(generateTravelCostMatrix())
		startNode = States()
		startNode.currentIndex = 0
		startNode.statePath = [startNode.currentIndex]

		startNode.costMatrix, startNode.bound = calculateLowerBound((start))

		# Frontier queue  
		# Space Complexity: O(N)
		S = [startNode]

		# Store best solution   
		# Space Complexity: O(1)
		BSSF: tuple[float, States] = (math.inf, States())

		# Search loop
		while len(S) > 0:
			# Check time allowance
    		# Time Complexity: O(1)
			if time.time() - startTime > time_allowance:
				break

			# Track max queue size 
			# Time Complexity: O(1)
			maxQueue = max(maxQueue, len(S))

			# Get node from frontier
    		# Time Complexity: O(logN) for heappop
			P = heapq.heappop(S)

			# Check if new best solution
			# Time Complexity: O(N)
			if isValidPath(P) and P.bound < BSSF[0]:
				BSSF = (P.bound, P)
				solution += 1

			# Get unvisited states
			# Time Complexity: O(N)
			unvisited = set(range(len(start)))
			if P.statePath is not None:
				unvisited.difference_update(P.statePath)

			# Handle goal state case
    		# Time Complexity: O(1)
			if not unvisited:
				unvisited.add(0)

			# Prune if worse than best solution 
    		# Time Complexity: O(1)
			if P.bound < BSSF[0]:
				# Expand node 
        		# Time Complexity: O(N^3) | Space Complexity: O(N)
				for node in expand(P, unvisited):
					states += 1
					if math.isinf(node.bound) or node.bound > BSSF[0]:
						pruned += 1
						continue

					heapq.heappush(S, node)

			else:
				# Increment counter for pruned nodes 
        		# Time Complexity: O(1)
				pruned += 1

		# Finalize best route         
		# Time Complexity: O(N) | Space Complexity: O(N)
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
	
	# Overall time complexity: O(N^2) | Overall space complexity: O(N)
	def fancy( self,time_allowance=60.0 ):
		pass