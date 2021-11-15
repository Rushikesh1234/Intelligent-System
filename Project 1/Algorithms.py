import util

class DFS(object):
    def depthFirstSearch(self, problem):
        """
        Search the deepest nodes in the search tree first
        [2nd Edition: p 75, 3rd Edition: p 87]

        Your search algorithm needs to return a list of pacman_path that reaches
        the goal.  Make sure to implement a graph search algorithm
        [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

        To get started, you might want to try some of these simple commands to
        understand the search problem that is being passed in:
        """
        
        ##print ("Start:", problem.getStartState())
        ##print ("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        ##print ("Start's successors:", problem.getSuccessors(problem.getStartState()))   
        
        "*** TTU CS3568 YOUR CODE HERE ***"
      
        node = {'state' : problem.getStartState() , 'action' : [], 'cost' : 0 } ##get cuurent state of our pacman
        
        if problem.isGoalState(node['state']):          ##if our pacman is standing at the end of travelling, then we can return empty path
            return []
        
        pacman_path = []
        
        checker_stack = util.Stack()    ##Stack used to store its child in DFS
        checker_stack.push(node)        ##insert initial visited path block
        
        get_new_path = set()      ##create a set to store child node of parent node, it means we are saving possible available route to avoid revisiting of that path
        
        while True:
            if checker_stack.isEmpty():                 ##If any node is not available, it means , there is no path for pacman
                break
            else:
                parent = checker_stack.pop()

                if parent['state'] not in get_new_path:     ##if child is not in get_new_path; it means we haven't visited that path yet, so consider that path block for pacman path exploring 
                    
                    get_new_path.add(parent['state'])  ##add visited path block in get_new_path to avoid repitition of path 
                    
                    if problem.isGoalState(parent['state']):   ##if we reach the goal, then return pacman's explored path
                        
                        end_node = parent
                                
                        while 'previous_path_node' in end_node:      ##get previous visited path block and add in our pacman_path
                            pacman_path.append(end_node['action'])
                            end_node = end_node['previous_path_node']
                        
                        pacman_path.reverse()       ##As we stored our visited path from down to up, so to print correctly, we have to send in reverse order       
                        return pacman_path
                    
                    else:
                        
                        unvisited_child_nodes_path = problem.getSuccessors(parent['state'])   ##parent node is looking for its child node, it means pacman is looking for next available path
                        
                        for unvisited_child in unvisited_child_nodes_path:
                            
                            child = {'state' : unvisited_child[0], 'action' : unvisited_child[1], 'cost' : unvisited_child[2], 'previous_path_node' : parent}
                            
                            checker_stack.push(child)      ##make an entry of your child node in checker_stack
        
        util.raiseNotDefined()

class BFS(object):
    def breadthFirstSearch(self, problem):
        "*** TTU CS3568 YOUR CODE HERE ***"
        ##print ("Start:", problem.getStartState())
        ##print ("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        ##print ("Start's successors:", problem.getSuccessors(problem.getStartState())) 
        
        
        checker_queue = util.Queue()    ##Stack used to store its child in BFS
        
        pacman_path = []    ##used to store exploring path for pacman
        
        node = {'state' : problem.getStartState() , 'action' : [], 'cost' : 0 } ##get cuurent state of our pacman
        
        checker_queue.push(node)        ##insert initial visited path block
        
        
        get_new_path = []      ##create a set to store child node of parent node, it means we are saving possible available route to avoid revisiting of that path
        
        while True:
            if not checker_queue.isEmpty():                 ##If any node is not available, it means , there is no path for pacman
                
                parent = checker_queue.pop()

                if parent['state'] not in get_new_path:     ##if child is not in get_new_path; it means we haven't visited that path yet, so consider that path block for pacman path exploring 
                    
                    get_new_path.append(parent['state'])  ##add visited path block in get_new_path to avoid repitition of path 
                    
                    if problem.isGoalState(parent['state']):   ##if we reach the goal, then return pacman's explored path
                        
                        end_node = parent
                                
                        while 'previous_path_node' in end_node:      ##get previous visited path block and add in our pacman_path
                            pacman_path.append(end_node['action'])
                            end_node = end_node['previous_path_node']
                        
                        pacman_path.reverse()       ##As we stored our visited path from down to up, so to print correctly, we have to send in reverse order       
                        return pacman_path
                    
                    else:
                        
                        unvisited_child_nodes_path = problem.getSuccessors(parent['state'])   ##parent node is looking for its child node, it means pacman is looking for next available path

                        for unvisited_child in unvisited_child_nodes_path:
                           
                            child = {'state' : unvisited_child[0], 'action' : unvisited_child[1], 'cost' : unvisited_child[2], 'previous_path_node' : parent}
                            
                            checker_queue.push(child)      ##make an entry of your child node in checker_queue
        
        
        util.raiseNotDefined()

class UCS(object):
    def uniformCostSearch(self, problem):
        "*** TTU CS3568 YOUR CODE HERE ***"

        ##print ("Start:", problem.getStartState())
        ##print ("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        ##print ("Start's successors:", problem.getSuccessors(problem.getStartState()))
        
        checker_queue = util.PriorityQueue()    ##Stack used to store its child in UCS
        
        pacman_path = []
        
        node = {'state' : problem.getStartState(), 'action': [], 'cost' : 0 } ##get cuurent state of our pacman
        
        checker_queue.push(node,0)        ##insert initial visited path block
        
        get_new_path = []      ##create a set to store child node of parent node, it means we are saving possible available route to avoid revisiting of that path
        
        
        while True:
            if checker_queue.isEmpty():                 ##If any node is not available, it means , there is no path for pacman
                break
            else:
                parent = checker_queue.pop()
                
                parent_cost = parent['cost']
                
                if parent['state'] not in get_new_path:     ##if child is not in get_new_path; it means we haven't visited that path yet, so consider that path block for pacman path exploring 
                    
                    get_new_path.append(parent['state'])  ##add visited path block in get_new_path to avoid repitition of path 
                    
                    if problem.isGoalState(parent['state']):   ##if we reach the goal, then return pacman's explored path
                        
                        end_node = parent
                                
                        while 'previous_path_node' in end_node:      ##get previous visited path block and add in our pacman_path
                            pacman_path.append(end_node['action'])
                            end_node = end_node['previous_path_node']
                        
                        pacman_path.reverse()       ##As we stored our visited path from down to up, so to print correctly, we have to send in reverse order       
                        return pacman_path
                    
                    else:
                        
                        unvisited_child_nodes_path = problem.getSuccessors(parent['state'])   ##parent node is looking for its child node, it means pacman is looking for next available path
                        
                        for unvisited_child in unvisited_child_nodes_path:
                            
                            child = {'state' : unvisited_child[0], 'action' : unvisited_child[1], 'cost' : unvisited_child[2], 'previous_path_node' : parent}
                            
                            next_cost = parent['cost'] + child['cost']
                            
                            child = {'state' : unvisited_child[0], 'action' : unvisited_child[1], 'cost' : next_cost, 'previous_path_node' : parent}  ## just update latest cost of our path in our child
                            
                            checker_queue.push(child, next_cost)      ##make an entry of your child node in checker_queue
        
        
        util.raiseNotDefined()
        
class aSearch (object):
    def nullHeuristic( state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0
    def aStarSearch(self,problem, heuristic=nullHeuristic):
        "Search the node that has the lowest combined cost and heuristic first."
        "*** TTU CS3568 YOUR CODE HERE ***"
        
 
        ##print ("Start:", problem.getStartState())
        ##print ("Is the start a goal?", problem.isGoalState(problem.getStartState()))
        ##print ("Start's successors:", problem.getSuccessors(problem.getStartState()))
        
        checker_queue = util.PriorityQueue()    ##Stack used to store its child in Astar
        
        pacman_path = []
        
        node = {'state' : problem.getStartState(), 'action': [], 'cost' : 0 } ##get cuurent state of our pacman
            
        checker_queue.push(node,0)        ##insert initial visited path block

        get_new_path = []      ##create a set to store child node of parent node, it means we are saving possible available route to avoid revisiting of that path
        
        
        while True:
            if not checker_queue.isEmpty():                 ##If any node is not available, it means , there is no path for pacman
               
                parent = checker_queue.pop()            ##take first element from queue for pacman journey
                
                parent_cost = parent['cost']   
                    
                if parent['state'] not in get_new_path:     ##if child is not in get_new_path; it means we haven't visited that path yet, so consider that path block for pacman path exploring 
                    
                    get_new_path.append(parent['state'])  ##add visited path block in get_new_path to avoid repitition of path 

                    if problem.isGoalState(parent['state']):   ##if we reach the goal, then return pacman's explored path
                        
                        end_node = parent
                                
                        while 'previous_path_node' in end_node:      ##get previous visited path block and add in our pacman_path
                            pacman_path.append(end_node['action'])
                            end_node = end_node['previous_path_node']
                        
                        pacman_path.reverse()       ##As we stored our visited path from down to up, so to print correctly, we have to send in reverse order       
                        return pacman_path
                    
                    else:
                        
                        unvisited_child_nodes_path = problem.getSuccessors(parent['state'])   ##parent node is looking for its child node, it means pacman is looking for next available path
                        
                        for unvisited_child in unvisited_child_nodes_path:
                            
                            child = {'state' : unvisited_child[0], 'action' : unvisited_child[1], 'cost' : unvisited_child[2], 'previous_path_node' : parent}
                            
                            next_cost = parent['cost'] + child['cost']      ## calculating our g(n) value, it means, we are calculating cost of parent and child node
                            heuristic_cost = next_cost + heuristic(child['state'], problem)  ## calculating our f(n) value, it means, we are calculating our g(n) and h(n) 
          
                            child = {'state' : unvisited_child[0], 'action' : unvisited_child[1], 'cost' : next_cost, 'previous_path_node' : parent}  ## just update latest cost of our path in our child
                                
                            checker_queue.push(child, heuristic_cost)      ##make an entry of your child node in checker_queue with our heuristic value
        
               
        
        util.raiseNotDefined()

