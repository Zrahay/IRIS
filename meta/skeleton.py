"""
This file contains the implementation of the Environment for RL. 

It will contain an Env class which in turn will consist of many functions like reset, step, Observe, Action, Rewards, Grader System, the function that the Agent will call like add_pass, compiler_and_measure, etc
"""

class Env:
    def __init__(self):
        """
        This is the default constructor. It consists of the initializations and all.
        """
        self.max_steps = 30 # Stops the agent from getting stuck in an infinite loop.

    
    def reset(self):
        pass
    
    def step(self):
        """
        The agent can take one step each time and call this function. Each time a step is taken, the 
        """

    def state(self):
        pass
    
    def observe(self):
        pass
    
    def _action(self):
        pass
    
    def _rewards(self):
        """
        All these private functions are going to be called inside the public APIs
        like step, reset, etc
        """
    
    def _grader_System(self):
        pass
    
    def _add_pass(self):
        pass
    
    def _compile_and_measure(self):
        pass