"""
This file contains the implementation of the Environment for RL. 

It will contain an Env class which in turn will consist of many functions like reset, step, Observe, Action, Rewards, Grader System, the function that the Agent will call like add_pass, compiler_and_measure, etc
"""
from tools.pass_sequence_generator import PassSequenceGenerator
import os
import glob
import random

PROGRAMS_DIR = os.path.join(os.path.dirname(__file__),"..", "training_programs")
EVAL_PROGRAMS = {
    "easy": os.path.join(PROGRAMS_DIR, "01_insertion_sort.c"),
    "medium": os.path.join(PROGRAMS_DIR, "08_strassen_matrix.c"),
    "hard": os.path.join(PROGRAMS_DIR, "114_polynomial_multiply_fft.c"),
}
 
ALL_PROGRAMS = sorted(glob.glob(os.path.join(PROGRAMS_DIR, "*.c")))
TRAIN_PROGRAMS = [p for p in ALL_PROGRAMS if p not in EVAL_PROGRAMS.values()]

class Env:
    def __init__(self):
        """
        This is the default constructor. It consists of the initializations and all.
        """
        self.max_steps = 30 # Stops the agent from getting stuck in an infinite loop.
        self.action_space = ["add_pass", "compile_and_measure", "get_program_info", "list_passes", "get_current_sequence"]
        self.current_sequence = []
        self.valid_passes = PassSequenceGenerator().ALL_PASSES
        self.episode_reward = 0.0
        self.done = False
        self.eval_programs = EVAL_PROGRAMS
        self.train_programs = TRAIN_PROGRAMS
        self.current_program = None
        self.baseline_O0 = None
        self.baseline_O1 = None
        self.baseline_O2 = None
        self.baseline_O3 = None

    
    def reset(self, task_id=None):
        """
        The main goal of this API will be to pick a program from the given 50 C files
        Then it will also help in computing the baselines with all the different flags by running each on QEMU.
        The final task of this API will be to reset everything in the env like the current sequence, episode reward, step count and setting self.done = false
        """
        
        # First task is to choose the current program
        if task_id is not None:
            if task_id in self.eval_programs:
                self.current_program = self.eval_programs[task_id]
            else:
                raise ValueError(f"Task ID '{task_id}' not found. Valid IDs: {list(self.eval_programs.keys())}")
        else:
            self.current_program = random.choice(self.train_programs)

        # Second task is to reset the whole Environment
        self.current_sequence = []
        self.episode_reward = 0.0
        self.step_count = 0
        self.done = False

        # Third task is to compute the baselines by connecting this code to a private API
        self.baseline_O0 = self._compute_baseline(self.current_program, "O0")
        self.baseline_O1 = self._compute_baseline(self.current_program, "O1")
        self.baseline_O2 = self._compute_baseline(self.current_program, "O2")
        self.baseline_O3 = self._compute_baseline(self.current_program, "O3")

        # Return initial observation to the agent
        return {
            "program": os.path.basename(self.current_program),
            "baselines": {
                "O0": self.baseline_O0,
                "O1": self.baseline_O1,
                "O2": self.baseline_O2,
                "O3": self.baseline_O3,
            },
            "current_sequence": [],
            "steps_remaining": self.max_steps,
        }

    def step(self, action: str) -> dict:
        """
        This public Api is going to:
        1. Parse the action string (format: "action_name" or "action_name:argument")
        2. Dispatch it to the corresponding private method
        3. Track step count and check episode termination
        4. Return observation, reward, done, info
        """

        # Step 1 - Check if episode is already done
        if self.done:
            return {"error": "Episode is done. Call reset() to start a new episode."}, 0.0, True, {}

        # Step 2 - Parse action and optional argument
        action_name = action.split(":")[0]
        action_arg = action.split(":", 1)[1] if ":" in action else None

        # Step 3 - Validate the action
        if action_name not in self.action_space:
            return {"error": f"Action '{action_name}' not found in action space. Valid actions: {self.action_space}"}, -0.1, False, {}

        # Step 4 - Increment step count
        self.step_count += 1

        # Step 5 - Dispatch the action to the corresponding function
        if action_name == "add_pass":
            observation = self._add_pass(action_arg)

        elif action_name == "compile_and_measure":
            observation = self._compile_and_measure()

        elif action_name == "get_program_info":
            observation = self._get_program_info()

        elif action_name == "list_passes":
            observation = self._list_passes()

        elif action_name == "get_current_sequence":
            observation = self._get_current_sequence()

        else:
            raise ValueError(f"Unknown action: {action_name}")

        # Step 6 - Check if max steps reached
        if self.step_count >= self.max_steps and not self.done:
            self.done = True

        return observation, self.episode_reward, self.done, {"step_count": self.step_count}

    def _compute_baseline(self, program_path, flag:str) -> float:
        """
        The task of this function is to compute the baseline for each flag for the chosen program by connecting it to the Internal Logic
        """
        # The only task is to compute the baselines for each flag
        import subprocess, time, tempfile 

        exe_file = os.path.join(tempfile.mkdtemp(), "baseline.exe")

        clang_cmd = [
            "clang", f"--target=riscv64-unknown-linux-gnu",
            f"-{flag}", program_path, "-o", exe_file, "-static"
        ]

        subprocess.run(clang_cmd, check = True, capture_output = True, timeout = 30)
        start = time.perf_counter()

        subprocess.run(["qemu-riscv64", exe_file], check = True, capture_output = True, timeout = 10)
        end = time.perf_counter()

        return end - start

    def state(self):
        """
        Return a full snapshot of the current episode state.
        This includes everything needed to understand and evaluate the episode.
        """
        return {
            "program": os.path.basename(self.current_program) if self.current_program else None,
            "program_path": self.current_program,
            "baselines": {
                "O0": self.baseline_O0,
                "O1": self.baseline_O1,
                "O2": self.baseline_O2,
                "O3": self.baseline_O3,
            },
            "current_sequence": list(self.current_sequence),
            "episode_reward": self.episode_reward,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "steps_remaining": self.max_steps - self.step_count,
            "done": self.done,
        }

    def _grader_System(self, task_id):
        """
        Grade the agent's performance on a specific eval task.
        Returns a score between 0.0 and 1.0.
        """
        if not self.done:
            return 0.0

        # If compilation failed, score is 0
        if self.baseline_O3 is None or self.baseline_O0 is None:
            return 0.0

        # Run compile_and_measure to get the agent's result
        result = self.state()
        episode_reward = result["episode_reward"]

        # Normalize reward to 0.0-1.0 range
        # Max possible reward: 0.05 * max_steps + 1.0 (beat O3) = 2.5
        # Min possible reward: -0.2 * max_steps - 0.5 (all invalid + fail) = -6.5
        max_reward = 0.05 * self.max_steps + 1.0
        min_reward = -0.2 * self.max_steps - 0.5
        score = (episode_reward - min_reward) / (max_reward - min_reward)
        return max(0.0, min(1.0, score))

    def _add_pass(self, pass_name):
        """
        Add a pass to the current sequence. Validates the pass name
        and updates episode reward accordingly.
        """
        if pass_name is None:
            self.episode_reward -= 0.1
            return {
                "status": "error",
                "message": "No pass name provided. Use format: add_pass:pass_name",
                "valid_passes": self.valid_passes,
            }

        if pass_name not in self.valid_passes:
            self.episode_reward -= 0.2
            return {
                "status": "invalid_pass",
                "message": f"Pass '{pass_name}' is not valid.",
                "valid_passes": self.valid_passes,
            }

        self.current_sequence.append(pass_name)
        self.episode_reward += 0.05
        return {
            "status": "success",
            "message": f"Added '{pass_name}' to sequence.",
            "current_sequence": list(self.current_sequence),
            "sequence_length": len(self.current_sequence),
            "steps_remaining": self.max_steps - self.step_count,
        }

    def _get_program_info(self):
        """
        Return information about the current program and its baselines.
        """
        return {
            "program": os.path.basename(self.current_program),
            "baselines": {
                "O0": self.baseline_O0,
                "O1": self.baseline_O1,
                "O2": self.baseline_O2,
                "O3": self.baseline_O3,
            },
            "steps_remaining": self.max_steps - self.step_count,
        }

    def _list_passes(self):
        """
        Return the list of all valid LLVM optimization passes.
        """
        return {
            "valid_passes": self.valid_passes,
            "total_count": len(self.valid_passes),
        }

    def _get_current_sequence(self):
        """
        Return the current pass sequence built so far.
        """
        return {
            "current_sequence": list(self.current_sequence),
            "sequence_length": len(self.current_sequence),
            "steps_remaining": self.max_steps - self.step_count,
        }

    def _compile_and_measure(self):
        """
        Compile the current program with the agent's pass sequence, measure execution time,
        compare against baselines, compute reward, and end the episode.
        Pipeline: clang → opt (agent's passes) → llc → gcc → qemu
        """
        import subprocess, time, tempfile

        tmp_dir = tempfile.mkdtemp()
        bc_file = os.path.join(tmp_dir, "program.bc")
        opt_bc_file = os.path.join(tmp_dir, "program.opt.bc")
        asm_file = os.path.join(tmp_dir, "program.s")
        exe_file = os.path.join(tmp_dir, "program.exe")

        try:
            # Step 1: C → LLVM bitcode (compile with -O0 so no built-in opts interfere)
            subprocess.run(
                ["clang", "--target=riscv64-unknown-linux-gnu", "-O0",
                 "-emit-llvm", "-c", self.current_program, "-o", bc_file],
                check=True, capture_output=True, timeout=30
            )

            # Step 2: Apply agent's pass sequence via opt
            if self.current_sequence:
                pass_arg = f"-passes={','.join(self.current_sequence)}"
            else:
                pass_arg = "-passes=default<O0>"

            subprocess.run(
                ["opt", pass_arg, bc_file, "-o", opt_bc_file],
                check=True, capture_output=True, timeout=60
            )

            # Step 3: LLVM bitcode → RISC-V assembly via llc
            subprocess.run(
                ["llc", "-march=riscv64", opt_bc_file, "-o", asm_file],
                check=True, capture_output=True, timeout=30
            )

            # Step 4: Assembly → executable via RISC-V cross-compiler
            subprocess.run(
                ["riscv64-linux-gnu-gcc", asm_file, "-o", exe_file, "-static"],
                check=True, capture_output=True, timeout=30
            )

            # Step 5: Run on QEMU and measure execution time
            start = time.perf_counter()
            subprocess.run(
                ["qemu-riscv64", exe_file],
                check=True, capture_output=True, timeout=10
            )
            end = time.perf_counter()
            execution_time = end - start

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            # Compilation or execution failed
            self.episode_reward -= 0.5
            self.done = True
            return {
                "status": "failed",
                "error": str(e),
                "reward": -0.5,
                "episode_reward": self.episode_reward,
            }

        # Step 6: Compare against baselines and compute reward
        if execution_time < self.baseline_O3:
            reward = 1.0
        elif execution_time < self.baseline_O2:
            reward = 0.5
        elif execution_time < self.baseline_O1:
            reward = 0.3
        elif execution_time < self.baseline_O0:
            reward = 0.1
        else:
            reward = 0.0

        self.episode_reward += reward
        self.done = True

        return {
            "status": "success",
            "execution_time": execution_time,
            "baselines": {
                "O0": self.baseline_O0,
                "O1": self.baseline_O1,
                "O2": self.baseline_O2,
                "O3": self.baseline_O3,
            },
            "pass_sequence": self.current_sequence,
            "reward": reward,
            "episode_reward": self.episode_reward,
        }