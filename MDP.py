# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:46:00 2020
This code solves for the optimal policy and values of a small MDP using both
(1)Value iteration and (2)Policy iteration. I borrowed the code for value iteration
from a online open class by Stanford and the link is 
https://stanford-cs221.github.io/autumn2019/live/mdp1/
I worked out the MDP and policy iteration on my own. 

Markov Decision Process (MDP):
    S: 1.On top of a hill
       2.Rolling down the hill
       3.At the bottom of the hill
    A: Drive, does not drive
    R: driving(-1), top of hill(+3), rolling(+1), bottom of hill(+1)
    P: transition probability
    discount: 0.8
@author: Liying Lu

Output:
***********************************************************************
Question 2: Value Iteration
state                V(s)            policy(s)           
bottom               8.670788252988093 drive               
rolling              7.936630602292576 not drive           
top                  9.783616691937087 not drive           
***********************************************************************
Question 3: Policy Iteration
state                V(s)            policy(s)           
bottom               8.670788252992018 drive               
rolling              7.936630602296501 not drive           
top                  9.783616691941013 not drive           
***********************************************************************
Question 4.1: Change discount to 0.99. The optimal policy did not change 
but the optimal value increased significantly. I tried to change the 
discount to 1 but the policy does not seem to converge. 
state                V(s)            policy(s)           
bottom               182.88333774754966 drive               
rolling              182.05450436997478 not drive           
top                  183.94197979733906 not drive           
***********************************************************************
Question 4.2: Change transition probability. I changed the transition 
probability for rolling down and drives. It now has a probability of 
.1, .1, and .8 to reach the bottom of the hill, still rolling, and top 
of the hill. The optimal policy for rolling changes to drive because 
the robot has a greater expected value if it drives when rolling down 
the hill.
state                V(s)            policy(s)           
bottom               9.564356435186083 drive               
rolling              10.257425742116776 drive               
top                  11.049504950037567 not drive           
***********************************************************************
Question 4.3: Change reward. I changed the reward for the robot at the 
top of the hill and not drive. It will get a reward of 3 if it becomes 
rolling on the hill and 1 if it stays at the top. The optimal policy 
for top changes to drive because the robot gets more reward if it drives 
and try to stay at the top of the hill. 
state                V(s)            policy(s)           
bottom               7.755511021593313 drive               
rolling              7.204408817184494 not drive           
top                  8.486973947445017 drive
"""

class MDP:
    def __init__(self,discount):
        self.discount = discount
    def actions(self,state):
        return ('drive', 'not drive')
    def step(self, state, action):
        # return list of (newState, prob, reward) triples
        # state = s, action = a, newState = s'
        # prob = T(s, a, s'), reward = Reward(s, a, s')
        result = []
        if state == 'bottom':
            if action == 'drive':
                result.append(('bottom', 0.4, 0))
                result.append(('top', 0.6, 2))
            else:
                result.append(('bottom',1,1))
        elif state == 'rolling':
            if action == 'drive':
                result.append(('bottom', 0.1, 0))
                result.append(('rolling', 0.6, 0))
                result.append(('top', 0.3, 2))
            else:
                result.append(('bottom',1,1))
        else: # for State.TOP
            if action == 'drive':
                result.append(('rolling', 0.1, 0))
                result.append(('top', 0.9, 2))
            else:
                result.append(('rolling', 0.3, 1))
                result.append(('top', 0.7, 3))
        return result
    def states(self):
        return ('bottom','rolling','top')

def valueIteration(mdp):
    # initialize
    V = {} # state -> Vopt[state]
    for state in mdp.states():
        V[state] = 0.0 # start with values set to zero

    def Q(state, action):
        return sum(prob*(reward + mdp.discount*V[newState]) \
                for newState, prob, reward in mdp.step(state, action))

    while True:
        # compute the new values (newV) given the old values (V)
        newV = {}
        for state in mdp.states():
            newV[state] = max(Q(state, action) for action in mdp.actions(state))
            #newV[state] = Q(state, action) for action in mdp.actions(state)
        # check for convergence
        if max(abs(V[state]-newV[state]) for state in mdp.states()) < 1e-10:
            break
        V = newV

    # find the policy after convergence
    policy = {}
    for state in mdp.states():
        policy[state] = max((Q(state, action), action) for action in mdp.actions(state))[1]

    # print optimal values and optimal policy
    print('{:20} {:15} {:20}'.format('state', 'V(s)', 'policy(s)'))
    for state in mdp.states():
        print('{:20} {:15} {:20}'.format(state, V[state], policy[state]))

    return V, policy

def policyIteration(mdp):
    # start with policy that never drives
    policy = {}
    V = {}
    for state in mdp.states():
        policy[state] = 'not drive' 
        V[state] = 0.0
    #print(policy)
    #print(V)

    def Q(state,action):
        return sum(prob*(reward + mdp.discount*V[newState]) \
                   for newState, prob, reward in mdp.step(state, action))
    
    while True:
        # policy evaluation
        while True:
            newV = {}
            for state in mdp.states():
                newV[state] = Q(state,policy[state])
            if max(abs(V[state]-newV[state]) for state in mdp.states()) < 1e-10:
                break
            V = newV
        
        # policy improvement
        policy_stable = True
        new_policy = {}
        for state in mdp.states():
            new_policy[state] = max((Q(state, action), action) for action in mdp.actions(state))[1]
        
        for state in mdp.states():
            if (policy[state] != new_policy[state]):
                policy_stable = False
        policy = new_policy
        
        if policy_stable==True:
            break
        
    # print optimal values and optimal policy
    print('{:20} {:15} {:20}'.format('state', 'V(s)', 'policy(s)'))
    for state in mdp.states():
        print('{:20} {:15} {:20}'.format(state, V[state], policy[state]))

    return V, policy


# Question 2: Value iteration with discount=0.8
print('***********************************************************************')
print('Question 2: Value Iteration')
mdp = MDP(0.8)
valueIteration(mdp)

# Question 3: Policy iteration with discount=0.8
print('***********************************************************************')
print('Question 3: Policy Iteration')
mdp = MDP(0.8)
policyIteration(mdp)

# Question 4: Change MDP is three different ways
# 1) change the discount factor
print('***********************************************************************')
print('Question 4.1: Change discount')
mdp_discount = MDP(0.99)
policyIteration(mdp_discount)

# 2) change the transition probabilities for a single action from a single state
class MDP2:
    def __init__(self,discount):
        self.discount = discount
    def actions(self,state):
        return ('drive', 'not drive')
    def step(self, state, action):
        # return list of (newState, prob, reward) triples
        # state = s, action = a, newState = s'
        # prob = T(s, a, s'), reward = Reward(s, a, s')
        result = []
        if state == 'bottom':
            if action == 'drive':
                result.append(('bottom', 0.4, 0))
                result.append(('top', 0.6, 2))
            else:
                result.append(('bottom',1,1))
        elif state == 'rolling':
            if action == 'drive':
                result.append(('bottom', 0.1, 0))
                result.append(('rolling', 0.1, 0))
                result.append(('top', 0.8, 2))
            else:
                result.append(('bottom',1,1))
        else: # for State.TOP
            if action == 'drive':
                result.append(('rolling', 0.1, 0))
                result.append(('top', 0.9, 2))
            else:
                result.append(('rolling', 0.3, 1))
                result.append(('top', 0.7, 3))
        return result
    def states(self):
        return ('bottom','rolling','top')

print('***********************************************************************')
print('Question 4.2: Change transition probability')
mdp_transition = MDP2(0.8)
policyIteration(mdp_transition)

# 3) change a reward for a single action at a single state
class MDP3:
    def __init__(self,discount):
        self.discount = discount
    def actions(self,state):
        return ('drive', 'not drive')
    def step(self, state, action):
        # return list of (newState, prob, reward) triples
        # state = s, action = a, newState = s'
        # prob = T(s, a, s'), reward = Reward(s, a, s')
        result = []
        if state == 'bottom':
            if action == 'drive':
                result.append(('bottom', 0.4, 0))
                result.append(('top', 0.6, 2))
            else:
                result.append(('bottom',1,1))
        elif state == 'rolling':
            if action == 'drive':
                result.append(('bottom', 0.1, 0))
                result.append(('rolling', 0.6, 0))
                result.append(('top', 0.3, 2))
            else:
                result.append(('bottom',1,1))
        else: # for State.TOP
            if action == 'drive':
                result.append(('rolling', 0.1, 0))
                result.append(('top', 0.9, 2))
            else:
                result.append(('rolling', 0.3, 3))
                result.append(('top', 0.7, 1))
        return result
    def states(self):
        return ('bottom','rolling','top')
    
print('***********************************************************************')
print('Question 4.3: Change reward')
mdp_reward = MDP3(0.8)
policyIteration(mdp_reward)