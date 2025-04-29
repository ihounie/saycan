import warnings
warnings.filterwarnings('ignore')

import argparse
import datetime
import os
import time

import cv2  # Used by ViLD.
from flax import linen as nn
from flax.training import checkpoints
from flax.metrics import tensorboard
import imageio
from heapq import nlargest
# import IPython
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
# from moviepy.editor import ImageSequenceClip
import numpy as np
import optax
from PIL import Image
import tensorflow.compat.v1 as tf
from src.environment.robo_gripper import *
from src.models.vild import *
from src.environment.env import *
from src.utils.constants import *
from src.models.model import *
from src.models.llm import *
from src.utils.helper import *
import json
from config import Config, ModelConfig, EnvironmentConfig as EnvConfig, TaskConfig, TrainingConfig, create_config_from_args

def parse_args():
    parser = argparse.ArgumentParser(description='Run SayCan evaluation')
    
    # Model arguments
    parser.add_argument('--api_key', type=str, 
                      default=os.environ.get("OPENAI_API_KEY"),
                      help='OpenAI API key (default: OPENAI_API_KEY environment variable)')
    parser.add_argument('--engine', type=str, default='gpt-4o', 
                       choices=['gpt-4o', 'unsloth/Llama-3.2-3B-Instruct'], 
                       help='Model engine to use')
    parser.add_argument('--scoring_method', type=str, default='direct',
                       choices=['direct', 'sequence'], help='Scoring method to use')
    parser.add_argument('--termination_string', type=str, default="done()",
                       help='Termination string for the model')
    
    # Environment arguments
    parser.add_argument('--use_environment_description', action='store_true', 
                       help='Use environment description in prompts')
    parser.add_argument('--only_plan', action='store_true', 
                       help='Only generate plan without execution')
    parser.add_argument('--ignore_affordance', action='store_true', 
                       help='Ignore affordance scores')
    parser.add_argument('--plot_on', action='store_true', 
                       help='Enable plotting')
    parser.add_argument('--max_tasks', type=int, default=20,
                       help='Maximum number of tasks to execute from the task file')
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.2,
                       help='GPU memory fraction to use')
    parser.add_argument('--saved_model_dir', type=str, default="./image_path_v2",
                       help='Directory containing saved model')
    parser.add_argument('--image_path', type=str, default="./2db.png",
                       help='Path to save/load images')
    parser.add_argument('--max_steps', type=int, default=5,
                       help='Maximum number of steps to execute')
    
    # Task arguments
    parser.add_argument('--raw_input', type=str, 
                       default='put any blocks on their matched colored bowls.',
                       help='Raw input task description')
    parser.add_argument('--pick_items', nargs='+', 
                       default=['yellow block', 'green block', 'blue block'],
                       help='List of items that can be picked')
    parser.add_argument('--place_items', nargs='+', 
                       default=['yellow bowl', 'green bowl', 'blue bowl'],
                       help='List of items that can be placed')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for training')
    parser.add_argument('--checkpoint_step', type=int, default=40000,
                       help='Checkpoint step to load')
    
    # New task file argument
    parser.add_argument('--task_file', type=str, default='tasks.json',
                      help='Path to JSON file containing tasks')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Add validation for API key
    if not args.api_key:
        raise ValueError("OpenAI API key must be provided either through --api_key argument or OPENAI_API_KEY environment variable")

    # Use args instead of hardcoded values
    config = {'pick': args.pick_items, 'place': args.place_items}
    raw_input = args.raw_input
    
    # Initialize components using CLI args
    client = OpenAI(api_key=args.api_key)
    ENGINE = args.engine
    termination_string = args.termination_string
    scoring_method = args.scoring_method
    only_plan = args.only_plan
    ignore_affordance = args.ignore_affordance
    
    # Environment setup using CLI args
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
    saved_model_dir = args.saved_model_dir
    image_path = args.image_path
    
    # Model loading with CLI args
    # Initialize model parameters first
    rng = jax.random.PRNGKey(0)
    init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
    init_text = jnp.ones((4, 512), jnp.float32)
    init_pix = jnp.zeros((4, 2), np.int32)
    init_params = TransporterNets().init(rng, init_img, init_text, init_pix)['params']

    # Create empty_state using initialized params
    empty_state = {'target': init_params, 'state': {}}

    # Now load checkpoint
    loaded_opt = checkpoints.restore_checkpoint(f'ckpt_{args.checkpoint_step}', empty_state)

    # Show if JAX is using GPU.
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)

    # Initialize environment
    if 'env' in locals():
        # Safely exit gripper threading before re-initializing environment.
        env.gripper.running = False
        while env.gripper.constraints_thread.isAlive():
            time.sleep(0.01)
    env = PickPlaceEnv()

    #---------------------------------prompt---------------------------------------
    #@title Prompt

    gpt3_context = """
    objects = [red block, yellow block, blue block, green bowl]
    # move all the blocks to the top left corner.
    robot.pick_and_place(blue block, top left corner)
    robot.pick_and_place(red block, top left corner)
    robot.pick_and_place(yellow block, top left corner)
    done()

    objects = [red block, yellow block, blue block, green bowl]
    # put the yellow one the green thing.
    robot.pick_and_place(yellow block, green bowl)
    done()

    objects = [yellow block, blue block, red block]
    # move the light colored block to the middle.
    robot.pick_and_place(yellow block, middle)
    done()

    objects = [blue block, green bowl, red block, yellow bowl, green block]
    # stack the blocks.
    robot.pick_and_place(green block, blue block)
    robot.pick_and_place(red block, green block)
    done()

    objects = [red block, blue block, green bowl, blue bowl, yellow block, green block]
    # group the blue objects together.
    robot.pick_and_place(blue block, blue bowl)
    done()

    objects = [green bowl, red block, green block, red bowl, yellow bowl, yellow block]
    # sort all the blocks into their matching color bowls.
    robot.pick_and_place(green block, green bowl)
    robot.pick_and_place(red block, red bowl)
    robot.pick_and_place(yellow block, yellow bowl)
    done()
    """

    use_environment_description = args.use_environment_description
    gpt3_context_lines = gpt3_context.split("\n")
    gpt3_context_lines_keep = []
    for gpt3_context_line in gpt3_context_lines:
        if "objects =" in gpt3_context_line and not use_environment_description:
            continue
        gpt3_context_lines_keep.append(gpt3_context_line)

    gpt3_context = "\n".join(gpt3_context_lines_keep)
    print(gpt3_context)

    #--------------------------------task and config---------------------------------
    #@title Task and Config
    only_plan = False
    ignore_affordance = False  # Set to True to ignore affordance scores
    scoring_method = "direct"#"sequence"  # Options: "sequence" (default) or "direct"

    #-----------------------------------Setup scene-----------------------------------------
    #@title Setup Scene
    np.random.seed(2)

    obs = env.reset(config)

    img_top = env.get_camera_image_top()
    img_top_rgb = cv2.cvtColor(img_top, cv2.COLOR_BGR2RGB)
    plt.imshow(img_top)

    imageio.imsave(image_path, img_top)
    #-----------------------------------------runner-------------------------------------------
    #@ load model
    if not only_plan:
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
        init_text = jnp.ones((4, 512), jnp.float32)
        init_pix = jnp.zeros((4, 2), np.int32)
        init_params = TransporterNets().init(key, init_img, init_text, init_pix)['params']
        print(f'Model parameters: {n_params(init_params):,}')
        
        # Use the original flax.optim format for loading the checkpoint
        from flax.training import train_state
        
        # First load the checkpoint with a compatible structure
        empty_state = {'target': init_params, 'state': {}}
        loaded_opt = checkpoints.restore_checkpoint(f'ckpt_{40000}', empty_state)
        
        # Then create our optimizer with the loaded parameters
        tx = optax.adam(learning_rate=1e-4)
        optim = {'target': loaded_opt['target'], 'opt_state': tx.init(loaded_opt['target'])}


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    session = tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options))
    saved_model_dir = "./image_path_v2"
    _ = tf.saved_model.loader.load(session, ["serve"], saved_model_dir)


    # Add before update_task_status function
    def update_environment_description(env_description, object_moved, new_location):
        """
        Updates the environment description after an object has been moved.
        
        Args:
            env_description: Current environment description string
            object_moved: The object that was moved
            new_location: The new location of the object
        
        Returns:
            Updated environment description string
        """
        # Parse the current environment description
        lines = env_description.strip().split('\n')
        objects_line = ""
        
        # Find the objects line
        for line in lines:
            if line.startswith("objects ="):
                objects_line = line
                break
        
        if not objects_line:
            return env_description  # No objects line found
        
        # Get the objects list
        objects_str = objects_line.replace("objects = [", "").replace("]", "")
        objects = [obj.strip() for obj in objects_str.split(",")]
        
        # Create a new description that reflects the updated state
        # In a real implementation, you would need to actually update object positions
        # This is a simplified version that just indicates the move happened
        updated_desc = f"objects = ["
        obj_strs = []
        for obj in objects:
            if obj == object_moved:
                # Mark that this object is now at the new location
                obj_strs.append(f"{obj} at {new_location}")
            else:
                obj_strs.append(obj)
        
        updated_desc += ", ".join(obj_strs) + "]"
        
        # Replace the objects line in the description
        new_lines = []
        for line in lines:
            if line.startswith("objects ="):
                new_lines.append(updated_desc)
            else:
                new_lines.append(line)
        
        return "\n".join(new_lines)

    # Add a function to track task progress
    def update_task_status(env_description, selected_tasks, raw_input):
        """
        Basic task status tracking without task-specific logic
        Returns:
            env_description: Current environment state
            task_status: Simple summary of completed actions
        """
        # Parse environment description
        objects = []
        if "objects =" in env_description:
            objects_line = [line for line in env_description.split('\n') if "objects =" in line][0]
            objects = [obj.strip() for obj in objects_line.replace("objects = [", "").replace("]", "").split(",")]
        
        # Track completed actions
        completed_actions = []
        for task in selected_tasks:
            if "pick_and_place" in task:
                parts = task.replace("robot.pick_and_place(", "").replace(")", "").split(", ")
                if len(parts) == 2:
                    completed_actions.append(f"Moved {parts[0]} to {parts[1]}")
        
        # Simple status summary
        task_status = f"Original task: {raw_input}\n"
        if completed_actions:
            task_status += "Completed actions:\n" + "\n".join([f"- {action}" for action in completed_actions])
        
        # Add basic remaining objects count
        moved_objects = [action.split(" to ")[0].replace("Moved ", "") for action in completed_actions]
        remaining_objects = [obj for obj in objects if obj not in moved_objects]
        if remaining_objects:
            task_status += f"\n\nRemaining objects: {len(remaining_objects)}"
        
        # Add pending action count
        remaining_actions = len([t for t in selected_tasks if t != "done()"])
        task_status += f"\nPending actions: {remaining_actions}"
        
        return task_status

    # Add after the update_task_status function
    def evaluate_task_success(initial_env_description, final_env_description, task_description, plan):
        """
        Use GPT-4.1 to evaluate if the task was successfully completed based on text descriptions only.
        This function doesn't run the simulator, but only evaluates the text-based environment descriptions.
        
        Args:
            initial_env_description: Initial environment description string
            final_env_description: Final environment description string (after all actions)
            task_description: The original task description
            plan: List of actions taken
        
        Returns:
            Boolean indicating success and explanation text
        """
        # Create the plan text
        plan_text = "\n".join([f"{i+1}. {action}" for i, action in enumerate(plan)])
        
        # Generate task-specific evaluation criteria
        task_criteria = ""
        if "correspond" in task_description.lower() or "matching" in task_description.lower():
            task_criteria = """
            Evaluation criteria:
            - Each colored block should be placed in a bowl of the same color
            - All blocks should be placed
            """
        elif "blue stuff together" in task_description.lower():
            task_criteria = """
            Evaluation criteria:
            - All blue objects (blocks) should be placed with other blue objects (blue bowl)
            """
        elif "stack" in task_description.lower() and "aren't red" in task_description.lower():
            task_criteria = """
            Evaluation criteria:
            - All non-red blocks should be stacked (one on top of another)
            - Red blocks should not be part of the stack
            """
        elif "corners" in task_description.lower() and "middle" in task_description.lower():
            task_criteria = """
            Evaluation criteria:
            - Blocks should fill all corners first
            - If there are more blocks than corners, the extra blocks go to the middle
            """
        elif "non-matching colors" in task_description.lower():
            task_criteria = """
            Evaluation criteria:
            - Each block should be placed in a bowl of a different color than the block
            - All blocks should be placed in bowls
            """
        elif "half" in task_description.lower() and "bowls" in task_description.lower() and "corners" in task_description.lower():
            task_criteria = """
            Evaluation criteria:
            - Half of the blocks should be in bowls
            - The other half should be in corners
            """
        elif "green block in the middle" in task_description.lower():
            task_criteria = """
            Evaluation criteria:
            - All green blocks should be placed in the middle
            - All non-green blocks should be placed in corners
            """
        elif "exactly one block in the middle" in task_description.lower():
            task_criteria = """
            Evaluation criteria:
            - Exactly one block should be placed in the middle
            - All other blocks should be placed in bowls
            """
        elif "odd number" in task_description.lower() and "blocks" in task_description.lower() and "bowl" in task_description.lower():
            task_criteria = """
            Evaluation criteria:
            - Each bowl should contain either an odd number of blocks (1, 3, 5, etc.) or no blocks at all
            - No bowl should contain an even number of blocks
            """
        
        # Prepare the messages for GPT-4.1
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant evaluating robotic task completion. Compare the initial and final environment descriptions to determine if the task was successfully completed. Look at how objects moved according to the plan and environment descriptions. Provide a clear YES or NO answer followed by a detailed explanation."
            },
            {
                "role": "user",
                "content": f"Task: {task_description}\n\n{task_criteria}\n\nActions executed:\n{plan_text}\n\nInitial environment state:\n{initial_env_description}\n\nFinal environment state:\n{final_env_description}\n\nBased on the initial and final environment descriptions and the specific task criteria, was the task successfully completed? Answer with YES or NO, followed by a detailed explanation of your reasoning. Don't focus on the physical execution, but on whether the final arrangement of objects matches what was required by the task."
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            max_tokens=400
        )
        
        # Extract the response
        evaluation_text = response.choices[0].message.content
        
        # Determine success based on response
        success = "YES" in evaluation_text.upper()
        
        return success, evaluation_text

    # Create a timestamped results directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = raw_input.replace(" ", "_").replace(".", "")[:30]
    base_dir = os.path.join("output", ENGINE)
    os.makedirs(base_dir, exist_ok=True)
    results_dir = os.path.join(base_dir, f"eval_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    def run_task(
        raw_input,
        config,
        env,
        session,
        optim,
        gpt3_context,
        use_environment_description,
        termination_string,
        category_name_string,
        vild_params,
        ENGINE,
        scoring_method,
        ignore_affordance,
        only_plan=False,
        description="",
        max_steps=args.max_steps
    ):
        """Run a single task and return the results."""
        # Reset environment with new config
        try:
            obs = env.reset(config)
        except Exception as e:
            print(f"Error resetting environment for task '{raw_input}': {e}")
            # Return a failure result if env reset fails
            return {
                "task": raw_input, "description": description, "plan": [], "success": False,
                "evaluation": f"Failed: Environment reset error: {e}",
                "initial_state": "N/A", "final_state": "N/A", "task_dir": "error",
                "initial_image": "error", "final_image": "error"
            }

        # Create a timestamped directory for this specific task run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name_safe = "".join(c if c.isalnum() else "_" for c in raw_input)[:20]
        task_dir = f"task_{timestamp}_{task_name_safe}"
        os.makedirs(task_dir, exist_ok=True)

        # Define image paths within the task directory
        initial_image_path = os.path.join(task_dir, "initial_state.jpg")
        vild_temp_image_path = os.path.join(task_dir, "vild_input_temp.jpg") # Task-specific path for VILD

        # Save initial image
        try:
            img_top = env.get_camera_image_top()
            imageio.imsave(initial_image_path, img_top)
            # Use the task-specific temp path for initial VILD input
            imageio.imsave(vild_temp_image_path, img_top)
        except Exception as e:
            print(f"Error saving initial image for task '{raw_input}': {e}")
            # Return failure if image saving fails
            return {
                "task": raw_input, "description": description, "plan": [], "success": False,
                "evaluation": f"Failed: Initial image saving error: {e}",
                "initial_state": "N/A", "final_state": "N/A", "task_dir": task_dir,
                "initial_image": "error", "final_image": "error"
            }

        # Get options and initial state description
        options = make_options(PICK_TARGETS, PLACE_TARGETS, termination_string=termination_string)
        try:
            found_objects = vild(session, vild_temp_image_path, category_name_string, vild_params, plot_on=False)
        except Exception as e:
            print(f"Error running VILD for task '{raw_input}': {e}")
            return {
                "task": raw_input, "description": description, "plan": [], "success": False,
                "evaluation": f"Failed: VILD error: {e}",
                "initial_state": "N/A", "final_state": "N/A", "task_dir": task_dir,
                "initial_image": initial_image_path, "final_image": "error"
            }

        scene_description = build_scene_description(found_objects)
        env_description = scene_description
        initial_env_description = env_description

        # Initialize task status and prompt
        task_status = update_task_status(env_description, [], raw_input)
        gpt3_prompt = gpt3_context
        if use_environment_description:
            gpt3_prompt += "\n" + env_description
        gpt3_prompt += "\n# " + raw_input + "\n"
        gpt3_prompt += f"\nTask Status:\n{task_status}\n"

        # Run task planning/execution loop
        num_steps = 0 # Renamed from num_tasks to avoid confusion
        selected_task = ""
        steps_text = []
        affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False)

        while not selected_task == termination_string:
            num_steps += 1
            if num_steps > args.max_steps:
                print(f"Task '{raw_input}' exceeded max steps ({args.max_steps}). Stopping.")
                break # Stop if max steps reached

            valid_options = [opt for opt in options if opt not in steps_text and opt != termination_string]
            if not valid_options:
                # If only termination_string is left, select it. Otherwise, break.
                if termination_string in options and termination_string not in steps_text:
                    selected_task = termination_string
                    steps_text.append(selected_task)
                    print(f"Step {num_steps}: Selecting {selected_task} as no other valid options remain.")
                    gpt3_prompt += selected_task + "\n" # Add to prompt history
                    break # End the loop after selecting done()
                else:
                    print(f"Task '{raw_input}' has no valid options left. Stopping.")
                    break # No valid moves left

            try:
                # Get LLM scores for valid options
                llm_scores, _ = gpt3_scoring(gpt3_prompt, valid_options + [termination_string], verbose=False, engine=ENGINE, print_tokens=False, scoring_method=scoring_method)

                # Combine scores (ensure termination string has a score, default to low if not present)
                if termination_string not in llm_scores:
                    llm_scores[termination_string] = -float('inf') # Give termination a very low score unless favored

                current_combined_scores = {}
                if ignore_affordance:
                    current_combined_scores = {option: np.exp(llm_scores[option]) for option in valid_options + [termination_string]}
                else:
                    valid_affordance_scores = {opt: affordance_scores.get(opt, 1.0) for opt in valid_options + [termination_string] if opt in affordance_scores or opt == termination_string}
                    # Ensure termination_string has affordance 1.0 if not present
                    if termination_string not in valid_affordance_scores:
                        valid_affordance_scores[termination_string] = 1.0

                    current_combined_scores = {
                        option: np.exp(llm_scores[option]) * valid_affordance_scores.get(option, 1.0)
                        for option in valid_options + [termination_string]
                    }

                if not current_combined_scores:
                    print(f"Task '{raw_input}' - No combined scores generated. Stopping.")
                    break
                
                selected_task = max(current_combined_scores, key=current_combined_scores.get)
                steps_text.append(selected_task)
                print(f"Step {num_steps}: Selecting '{selected_task}' (Score: {current_combined_scores[selected_task]:.3f})")


                # Update state based on selected task
                if selected_task == termination_string:
                    gpt3_prompt += selected_task + "\n"
                    break # Task finished

                if "pick_and_place" in selected_task:
                    parts = selected_task.strip().replace("robot.pick_and_place(", "").replace(")", "").split(", ")
                    if len(parts) == 2:
                        object_moved, new_location = parts
                        # Update text-based state description
                        env_description = update_environment_description(env_description, object_moved, new_location)
                        task_status = update_task_status(env_description, steps_text, raw_input)

                        # Rebuild prompt for next step
                        gpt3_prompt = gpt3_context
                        if use_environment_description:
                            gpt3_prompt += "\n" + env_description
                        gpt3_prompt += "\n# " + raw_input + "\n"
                        gpt3_prompt += f"\nTask Status:\n{task_status}\n"
                        for task_step in steps_text: # Append history
                            gpt3_prompt += task_step + "\n"

                        # Execute in simulator if not just planning
                        if not only_plan:
                            nlp_step = step_to_nlp(selected_task)
                            print(f"  Executing: {nlp_step}")
                            try:
                                obs = run_cliport(optim, env, obs, nlp_step, None, show_state=False)

                                # Update visual perception for next step's affordance
                                img_top = env.get_camera_image_top()
                                imageio.imsave(vild_temp_image_path, img_top) # Use temp path
                                found_objects = vild(session, vild_temp_image_path, category_name_string, vild_params, plot_on=False)
                                affordance_scores = affordance_scoring(options, found_objects, block_name="box", bowl_name="circle", verbose=False)
                            except Exception as sim_error:
                                print(f"  Simulation error during '{nlp_step}': {sim_error}")
                                # Decide how to handle sim error - maybe stop the task?
                                # For now, let's record the error and stop this task.
                                return {
                                    "task": raw_input, "description": description, "plan": steps_text, "success": False,
                                    "evaluation": f"Failed: Simulation error during '{nlp_step}': {sim_error}",
                                    "initial_state": initial_env_description, "final_state": env_description, "task_dir": task_dir,
                                    "initial_image": "initial_state.jpg", "final_image": "error"
                                }
                    else:
                        print(f"  Warning: Could not parse action '{selected_task}'")
                        # Append raw task to prompt if not parsable pick/place
                        gpt3_prompt += selected_task + "\n"
                else:
                    # If not pick/place (e.g., unexpected action or done()), just append
                    gpt3_prompt += selected_task + "\n"

            except Exception as loop_error:
                print(f"Error during task loop for '{raw_input}' at step {num_steps}: {loop_error}")
                # Record error and potentially stop
                return {
                    "task": raw_input, "description": description, "plan": steps_text, "success": False,
                    "evaluation": f"Failed: Error in planning loop: {loop_error}",
                    "initial_state": initial_env_description, "final_state": env_description, "task_dir": task_dir,
                    "initial_image": "initial_state.jpg", "final_image": "error"
                }

        # After task completion loop
        final_image_path = os.path.join(task_dir, "final_state.jpg")
        try:
            # Save final image if simulation ran
            if not only_plan:
                final_img = env.get_camera_image_top()
                imageio.imsave(final_image_path, final_img)
            else:
                # If only planning, copy initial image as final for consistency? Or mark as N/A?
                # shutil.copy2(initial_image_path, final_image_path)
                final_image_path = "N/A (planning only)" # Mark as not applicable
        except Exception as e:
            print(f"Error saving final image for task '{raw_input}': {e}")
            final_image_path = "error" # Mark as error

        # Evaluate task success using the final text-based environment description
        # Note: This evaluation uses the *text description* changes, not the final simulator state directly.
        success, evaluation_text = evaluate_task_success(
            initial_env_description,
            env_description, # Final text state description after simulated steps
            raw_input,
            steps_text
        )

        # Clean up the temporary VILD image
        if os.path.exists(vild_temp_image_path):
            try:
                os.remove(vild_temp_image_path)
            except OSError as e:
                print(f"Warning: Could not remove temp file {vild_temp_image_path}: {e}")


        return {
            "task": raw_input,
            "description": description,
            "plan": steps_text,
            "success": success,
            "evaluation": evaluation_text,
            "initial_state": initial_env_description,
            "final_state": env_description, # Return the final text state
            "task_dir": task_dir,
            "initial_image": "initial_state.jpg", # Relative path within task_dir
            "final_image": os.path.basename(final_image_path) if final_image_path not in ["error", "N/A (planning only)"] else final_image_path
        }

    # Load tasks from JSON file
    try:
        with open(args.task_file) as f:
            tasks = json.load(f)
            
        # Validate task structure
        required_keys = {'raw_input', 'config', 'description'}
        for task in tasks:
            if not required_keys.issubset(task.keys()):
                raise ValueError(f"Task missing required keys: {task}")
            if 'pick' not in task['config'] or 'place' not in task['config']:
                raise ValueError(f"Invalid config in task: {task['raw_input']}")
                
    except Exception as e:
        print(f"Error loading tasks from {args.task_file}: {e}")
        raise

    # Run all tasks and collect results
    results = []
    task_types = {}

    # Main sequential execution loop
    for i, task_config in enumerate(tasks):
        if i >= args.max_tasks:  # Add this condition
            print(f"Stopping after {args.max_tasks} tasks as configured")
            break
        
        task_input = task_config['raw_input']
        if task_input not in task_types:
            task_types[task_input] = []

        print(f"\nRunning task {i+1}/{len(tasks)}: {task_input}")
        print(f"Configuration: {task_config.get('description', 'No description')}")
        # Progress calculation remains the same

        # Call the refactored run_task function
        # Pass all necessary global or shared variables explicitly
        result = run_task(
            raw_input=task_input,
            config=task_config['config'],
            env=env,                 # Pass shared environment
            session=session,           # Pass shared TF session
            optim=optim,             # Pass shared optimizer state
            gpt3_context=gpt3_context,
            use_environment_description=use_environment_description,
            termination_string=termination_string,
            category_name_string=category_name_string,
            vild_params=vild_params,
            ENGINE=ENGINE,
            scoring_method=scoring_method,
            ignore_affordance=ignore_affordance,
            only_plan=only_plan,
            description=task_config.get('description', ''),
            max_steps=args.max_steps # Use the argument for max steps
        )
        results.append(result)
        task_types[task_input].append(result)

        # Print results for this task
        print(f"\nTask: {result['task']}")
        print(f"Configuration: {result['description']}")
        print(f"Success: {result['success']}")
        print(f"Evaluation: {result['evaluation']}")
        if result['task_dir'] != 'error':
            print(f"Files saved to: {result['task_dir']}")
        else:
            print(f"Task failed before directory creation.")
        print("\nPlan:")
        # Ensure plan is iterable even if it's None or failed early
        plan_steps = result.get('plan', [])
        if plan_steps:
            for step_idx, step in enumerate(plan_steps):
                print(f"Step {step_idx+1}: {step}")
        else:
            print("No plan generated or task failed early.")
        print("\n" + "="*80)


    # Calculate and print overall accuracy
    successful_tasks = sum(1 for r in results if r.get('success', False)) # Use .get for safety
    total_tasks_executed = len(results)
    if total_tasks_executed > 0:
        accuracy = successful_tasks / total_tasks_executed * 100
        print(f"\nOverall Accuracy: {accuracy:.2f}% ({successful_tasks}/{total_tasks_executed} tasks successful)")
    else:
        print("\nNo tasks were executed.")


    # Calculate accuracy per task type
    print("\nAccuracy by Task Type:")
    print("-"*50)
    for task_input, task_results in task_types.items():
        successful = sum(1 for r in task_results if r.get('success', False))
        total = len(task_results)
        if total > 0:
            task_accuracy = successful / total * 100
            print(f"{task_input}:")
            print(f"  Success rate: {task_accuracy:.2f}% ({successful}/{total})")

            # Summary of evaluations for failed tasks
            if successful < total:
                print("  Failed configurations:")
                for result in task_results:
                    if not result.get('success', False):
                        desc = result.get('description', 'N/A')
                        eval_summary = result.get('evaluation', 'N/A').split('\n')[0] # First line of evaluation
                        print(f"    - {desc}")
                        print(f"      Reason: {eval_summary}")
        else:
            print(f"{task_input}: No results")
        print("-"*50)

    # Save all results to a JSON file and create a consolidated results directory
    if total_tasks_executed > 0: # Only create report if tasks ran
        # Create subdirectories for tasks by type and copy images
        import shutil
        for task_input, task_results in task_types.items():
            task_type_dir_name = "".join(c if c.isalnum() else "_" for c in task_input)[:30]
            task_type_dir_path = os.path.join(results_dir, task_type_dir_name)
            os.makedirs(task_type_dir_path, exist_ok=True)

            for i, result in enumerate(task_results):
                if result.get('task_dir') != 'error' and os.path.isdir(result['task_dir']):
                    config_dir = os.path.join(task_type_dir_path, f"config_{i+1}_{result['description'].replace(' ','_')[:20]}")
                    os.makedirs(config_dir, exist_ok=True)

                    # Copy images if they exist and are not marked as error/N/A
                    for img_key, img_filename in [('initial_image', 'initial.jpg'), ('final_image', 'final.jpg')]:
                        img_rel_path = result.get(img_key)
                        if img_rel_path and img_rel_path not in ['error', 'N/A (planning only)']:
                            src_img_path = os.path.join(result['task_dir'], img_rel_path)
                            dst_img_path = os.path.join(config_dir, img_filename)
                            if os.path.exists(src_img_path):
                                try:
                                    shutil.copy2(src_img_path, dst_img_path)
                                except Exception as copy_err:
                                    print(f"Warning: Could not copy image {src_img_path} to {dst_img_path}: {copy_err}")
                            else:
                                print(f"Warning: Source image not found: {src_img_path}")

        # Create consolidated report with relative image links
        report_content = f"# SayCan Task Evaluation Results\n\n"
        report_content += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report_content += f"## Overall Accuracy: {accuracy:.2f}% ({successful_tasks}/{total_tasks_executed})\n\n"

        for task_input, task_results in task_types.items():
            successful = sum(1 for r in task_results if r.get('success', False))
            total = len(task_results)
            if total > 0:
                task_accuracy = successful / total * 100
                task_type_dir_name = "".join(c if c.isalnum() else "_" for c in task_input)[:30]

                report_content += f"## Task: {task_input}\n\n"
                report_content += f"Success rate: {task_accuracy:.2f}% ({successful}/{total})\n\n"

                for i, result in enumerate(task_results):
                    config_dir_name = f"config_{i+1}_{result['description'].replace(' ','_')[:20]}"
                    relative_config_dir = os.path.join(task_type_dir_name, config_dir_name)

                    report_content += f"### Configuration {i+1}: {result.get('description', 'N/A')}\n\n"
                    report_content += f"**Success:** {'Yes' if result.get('success', False) else 'No'}\n\n"
                    report_content += f"**Plan:**\n"
                    plan_steps = result.get('plan', [])
                    if plan_steps:
                        for j, step in enumerate(plan_steps):
                            report_content += f"{j+1}. {step}\n"
                    else:
                        report_content += "No plan generated or task failed early.\n"
                    report_content += f"\n**Evaluation:**\n{result.get('evaluation', 'N/A')}\n\n"

                    # Add relative image links if images were copied
                    initial_img_rel = os.path.join(relative_config_dir, 'initial.jpg')
                    final_img_rel = os.path.join(relative_config_dir, 'final.jpg')

                    if os.path.exists(os.path.join(results_dir, initial_img_rel)):
                        report_content += f"**Initial State:**\n\n![Initial State]({initial_img_rel})\n\n"
                    else:
                        report_content += f"**Initial State:** (Image not available)\n\n"

                    if os.path.exists(os.path.join(results_dir, final_img_rel)):
                        report_content += f"**Final State:**\n\n![Final State]({final_img_rel})\n\n"
                    else:
                        report_content += f"**Final State:** (Image not available or planning only)\n\n"

                    report_content += "---\n\n"

            report_content += "\n\n"

        # Write the report to a markdown file
        report_path = os.path.join(results_dir, "report.md")
        with open(report_path, "w") as f:
            f.write(report_content)

        # Save the full results data
        results_json_path = os.path.join(results_dir, "results.json")
        # Prepare data for JSON serialization (convert numpy types if any)
        serializable_results = []
        for r in results:
            # Basic check, can be expanded if complex types are used
            serializable_results.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()})

        serializable_task_type_accuracy = {}
        for task_input, task_data in task_types.items():
            if len(task_data) > 0:
                successful = sum(1 for r in task_data if r.get('success', False))
                total = len(task_data)
                serializable_task_type_accuracy[task_input] = {
                    "success_rate": successful / total * 100,
                    "successful": successful,
                    "total": total
                }


        with open(results_json_path, "w") as f:
            json.dump({
                "tasks": serializable_results,
                "overall_accuracy": accuracy,
                "task_type_accuracy": serializable_task_type_accuracy,
                "timestamp": timestamp
            }, f, indent=2)

        print(f"\nDetailed results saved to {results_dir}/")
        print(f"Report available at {report_path}")
        print(f"Full results data at {results_json_path}")
    else:
        print("\nNo tasks executed, skipping report generation.")

if __name__ == "__main__":
    main()