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
        self.action_space = ["add_pass", "compile_and_measure", "get_program_info", "list_passes", "get_current_sequence", "get_model_suggestion"]

    
    def reset(self):
        pass
    
    def step(self, action: str) -> str:
        """
        This public Api is going to:
        1. Match the action mentioned by the agent
        2. Dispatch it to the following function
        3. Calculate the reward based on the action
        4. The reward is calculated by dispatching it to the corresponding function
        5. Dispatch and get the next state using the private API _next_observation()
        6. Return the next state, reward, {} 
        """

        # Step 1 -Assert if the given function exists in the space
        assert action in self.action_space, f"Action {action} not found in action space"

        # Step 2 -Dispatch the action to the corresponding function
        if action == "add_pass":
            return self._add_pass()

        elif action == "compile_and_measure":
            return self._compile_and_measure()

        elif action == "get_program_info":
            return self._get_program_info()

        elif action == "list_passes":
            return self._list_passes()

        elif action == "get_current_sequence":
            return self._get_current_sequence()

        elif action == "get_model_suggestion":
            return self._get_model_suggestion()

        else:
            raise ValueError(f"Unknown action: {action}")

        # Step 3 - 
        # 






































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