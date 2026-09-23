"""
OpenManus Rollout - Staged rollout with VERL compatibility.
Uses modular stage processing system.
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from openmanus_rl.multi_turn_rollout.rollout_loop import TrajectoryCollector
from openmanus_rl.multi_turn_rollout.modular_stages import ModularStageProcessor


class OpenmanusRollout(TrajectoryCollector):
    """
    Staged rollout that extends TrajectoryCollector for VERL compatibility.
    Supports: 
    
    Planning stage
    <plan> xxx</plan>
    [opt] 
    <memory query> xxx </memory query><memory result> xxx </memory result>
    Action stage
    <action>
    which tool use, parameters</action>
    <action results>
    link to out tools to get the results
    </action results>
    Memory Stage
    <memory store>xxx</memory store>
    Reflection Stage
    <reflection > 
    xxX
    <memory query> xxx </memory query>
    <memory result> xxx </memory result>
    xxx 
    </reflection>
    
    Can be used standalone or as drop-in replacement for TrajectoryCollector.
    """
    
    def __init__(self, config, tokenizer, processor=None):
        # Initialize parent for VERL compatibility
        super().__init__(config, tokenizer, processor)
        
        # Initialize modular stage processor
        memory_file = getattr(config, 'memory_file', 'memory.md')
        self.stage_processor = ModularStageProcessor(memory_file)
        
        # Register default tools
        for name, func in DEFAULT_TOOLS.items():
            self.stage_processor.register_tool(name, func)
        
        # Enable staged format by default
        self.use_staged = getattr(config, 'use_staged_format', True)
    
    def parse_staged(self, text: str) -> Dict[str, Any]:
        """
        Parse staged format according to exact spec:
        Planning: <plan> with optional <memory query> inside
        Action: <action> followed by <action results>
        Memory: <memory store>
        Reflection: <reflection> with optional <memory query> inside
        """
        result = {
            'plan': None,
            'plan_memory_queries': [],
            'action': None,
            'action_results': None,
            'memory_store': None,
            'reflection': None,
            'reflection_memory_queries': [],
            'think': None,  # Backward compatibility
        }
        
        # Extract main stages
        patterns = {
            'plan': r'<plan>(.*?)</plan>',
            'action': r'<action>(.*?)</action>',
            'action_results': r'<action results>(.*?)</action results>',  # Note: space in tag
            'memory_store': r'<memory store>(.*?)</memory store>',  # Note: space in tag
            'reflection': r'<reflection\s*>(.*?)</reflection>',  # Allow space after reflection
            'think': r'<think>(.*?)</think>',  # Backward compatibility
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
        
        # Extract memory queries from within plan
        if result['plan']:
            result['plan_memory_queries'] = re.findall(
                r'<memory query>(.*?)</memory query>', 
                result['plan'], 
                re.DOTALL
            )
            # Clean plan text (remove memory tags for cleaner storage)
            clean_plan = re.sub(r'<memory query>.*?</memory query>', '', result['plan'], flags=re.DOTALL)
            clean_plan = re.sub(r'<memory result>.*?</memory result>', '', clean_plan, flags=re.DOTALL)
            result['plan_clean'] = clean_plan.strip()
        
        # Extract memory queries from within reflection
        if result['reflection']:
            result['reflection_memory_queries'] = re.findall(
                r'<memory query>(.*?)</memory query>', 
                result['reflection'], 
                re.DOTALL
            )
            # Clean reflection text
            clean_reflection = re.sub(r'<memory query>.*?</memory query>', '', result['reflection'], flags=re.DOTALL)
            clean_reflection = re.sub(r'<memory result>.*?</memory result>', '', clean_reflection, flags=re.DOTALL)
            result['reflection_clean'] = clean_reflection.strip()
        
        return result
    
    def query_memory(self, query: str, top_k: int = 3) -> str:
        """Simple memory search - delegate to stage processor."""
        return self.stage_processor.query_memory(query, top_k)
    
    def store_memory(self, content: str, episode: str = "", step: int = 0):
        """Store memory to file and RAM - delegate to stage processor."""
        self.stage_processor.store_memory(content, episode, step)
    
    def execute_tool(self, action: str) -> Tuple[str, Optional[str]]:
        """
        Parse and execute tool from action.
        Returns (env_action, tool_result).
        """
        if not action:
            return action, None
        
        # Check for tool format
        if 'tool:' in action.lower():
            tool_name = None
            params = {}
            
            for line in action.split('\n'):
                if line.lower().startswith('tool:'):
                    tool_name = line.split(':', 1)[1].strip()
                elif line.lower().startswith('parameters:'):
                    try:
                        params = json.loads(line.split(':', 1)[1].strip())
                    except:
                        params = {'query': line.split(':', 1)[1].strip()}
            
            if tool_name and tool_name in self.stage_processor.action.tools:
                result = self.stage_processor.action.tools[tool_name](params)
                return "", result  # Empty action for env, return tool result
        
        return action, None
    
    def process_response(self, response: str, episode_id: str, step_id: int) -> Tuple[str, Dict]:
        """
        Process response using modular stage processor.
        Returns (env_action, processed_data).
        """
        # Use modular processor
        result = self.stage_processor.process_response(response, episode_id, step_id)
        
        # Extract environment action
        env_action = result.get('env_action', '')
        
        # For backward compatibility - if no staged format, try simple parse
        if not env_action:
            simple = self.stage_processor.parse_simple(response)
            if simple.get('think') or simple.get('action'):
                env_action = simple.get('action', '')
        
        return env_action, result
    
    def multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        envs,
        is_train: bool = True
    ) -> DataProto:
        """
        VERL-compatible rollout loop.
        Can handle both staged and simple formats.
        """
        # If not using staged format, delegate to parent
        if not self.use_staged:
            return super().multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train)
        
        # Reset environments
        obs, infos = envs.reset()
        
        # Handle batch size adjustment (VERL compatibility)
        batch_size = len(gen_batch)
        length_obs = len(obs['text']) if obs.get('text') is not None else len(obs.get('image', []))
        
        if batch_size != length_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            batch_size = len(gen_batch)
        
        # Initialize storage
        trajectories = []
        episode_rewards = np.zeros(batch_size)
        episode_lengths = np.zeros(batch_size, dtype=int)
        is_done = np.zeros(batch_size, dtype=bool)
        trajectory_uids = [f"traj_{i}" for i in range(batch_size)]  # Consistent UIDs
        
        # Main rollout loop
        for step in range(self.config.env.max_steps):
            if is_done.all():
                break
            
            active_masks = ~is_done
            
            # Preprocess observations (VERL style)
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            
            # Prepare input for actor
            batch_keys = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_keys = ["raw_prompt_ids"]
            
            batch_input = batch.pop(
                batch_keys=batch_keys,
                non_tensor_batch_keys=non_tensor_keys
            )
            batch_input.meta_info = gen_batch.meta_info
            
            # Generate sequences
            batch_output = actor_rollout_wg.generate_sequences(batch_input)
            
            # Decode responses
            responses = self.tokenizer.batch_decode(
                batch_output.batch['responses'],
                skip_special_tokens=True
            )
            
            # Process responses
            actions = []
            parsed_responses = []
            
            for i, response in enumerate(responses):
                if not active_masks[i]:
                    actions.append("")
                    parsed_responses.append({})
                    continue
                
                # Process with staged format
                action, parsed = self.process_response(
                    response,
                    episode_id=f"ep_{i}",
                    step_id=step
                )
                
                actions.append(action)
                parsed_responses.append(parsed)
            
            # Step environment
            next_obs, rewards, dones, infos = envs.step(actions)
            
            # Handle reward shapes
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)
            
            # Update tracking
            episode_rewards += rewards * active_masks
            episode_lengths[active_masks] += 1
            is_done = is_done | dones
            
            # Store trajectory data (VERL format)
            batch.non_tensor_batch['rewards'] = rewards
            batch.non_tensor_batch['active_masks'] = active_masks
            batch.non_tensor_batch['parsed_responses'] = parsed_responses
            batch.non_tensor_batch['traj_uid'] = trajectory_uids
            
            batch = batch.union(batch_output)
            trajectories.append(batch)
            
            # Update observations
            obs = next_obs
        
        # Use parent's gather_rollout_data for VERL compatibility
        if hasattr(self, 'gather_rollout_data'):
            # Convert to expected format
            total_batch_list = [[] for _ in range(batch_size)]
            for batch_data in trajectories:
                batch_list = self._to_list_of_dict(batch_data)
                for i, item in enumerate(batch_list):
                    if i < batch_size:
                        total_batch_list[i].append(item)
            
            # Generate success metrics
            success = envs.success_evaluator(
                total_infos=[[{}] * len(total_batch_list[0]) for _ in range(batch_size)],
                total_batch_list=total_batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths
            )
            
            # Use parent's gathering method
            import uuid
            traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
            
            return self.gather_rollout_data(
                total_batch_list=total_batch_list,
                episode_rewards=episode_rewards,
                episode_lengths=episode_lengths,
                success=success,
                traj_uid=traj_uid
            )
        else:
            # Fallback to simple packaging
            return self._simple_package(trajectories, episode_rewards, episode_lengths)
    
    def _to_list_of_dict(self, batch: DataProto) -> List[Dict]:
        """Convert DataProto batch to list of dicts."""
        result = []
        if 'responses' in batch.batch:
            batch_size = len(batch.batch['responses'])
        else:
            batch_size = len(batch.batch.get('input_ids', []))
        
        for i in range(batch_size):
            item = {}
            for key, value in batch.batch.items():
                if hasattr(value, '__len__') and len(value) > i:
                    item[key] = value[i]
            for key, value in batch.non_tensor_batch.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > i:
                    item[key] = value[i]
                else:
                    item[key] = value
            result.append(item)
        
        return result
    
    def _simple_package(self, trajectories, rewards, lengths) -> DataProto:
        """Simple packaging when parent methods not available."""
        all_data = []
        for traj_batch in trajectories:
            if hasattr(traj_batch, 'batch'):
                all_data.append(traj_batch.batch)
        
        if all_data:
            batch = collate_fn(all_data)
        else:
            batch = {}
        
        return DataProto.from_single_dict(
            data=batch,
            meta_info={
                'mean_reward': float(np.mean(rewards)),
                'mean_length': float(np.mean(lengths)),
                'success_rate': float(np.mean(rewards > 0))
            }
        )
    
    def register_tool(self, name: str, func):
        """Register a tool function."""
        self.stage_processor.register_tool(name, func)
    
    # Alias for simpler API
    rollout = multi_turn_loop


# Simple default tools
def search_tool(params):
    return f"Found: {params.get('query', 'nothing')}"

def calculate_tool(params):
    try:
        result = eval(params.get('expression', '0'), {"__builtins__": {}})
        return f"Result: {result}"
    except:
        return "Error: Invalid expression"

DEFAULT_TOOLS = {
    'search': search_tool,
    'calculate': calculate_tool
}