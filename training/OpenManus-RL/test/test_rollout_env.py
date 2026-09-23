"""
Training-free test suite with REAL environments (WebShop/AlfWorld).
This allows you to test the rollout system with actual environments without training.
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openmanus_rl.multi_turn_rollout.openmanus_rollout import OpenmanusRollout
from openmanus_rl.environments.env_manager import make_envs
from verl import DataProto
import torch


class SimpleConfig:
    """Configuration for test runs."""
    def __init__(self, env_name="webshop", batch_size=2):
        self.env = EnvConfig(env_name)
        self.data = DataConfig(batch_size)
        self.memory_file = f'test_memory_{env_name}.md'
        self.use_staged_format = True
        
class EnvConfig:
    def __init__(self, env_name):
        self.env_name = env_name
        self.seed = 42
        self.max_steps = 10
        self.history_length = 3
        self.rollout = RolloutConfig()
        
        # WebShop specific
        self.webshop = WebShopConfig()
        
class WebShopConfig:
    def __init__(self):
        self.use_small = True  # Use smaller dataset for testing
        self.human_goals = True
        
class RolloutConfig:
    def __init__(self):
        self.n = 0  # No repetition for testing
        
class DataConfig:
    def __init__(self, batch_size):
        self.train_batch_size = batch_size
        self.val_batch_size = 1


class SimpleTokenizer:
    """Simple tokenizer for testing."""
    def __init__(self):
        self.pad_token_id = 0
        
    def encode(self, text, return_tensors=None):
        # Simple character-level encoding
        tokens = [ord(c) % 256 for c in text[:512]]  # Limit length
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([tokens])}
        return tokens
    
    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return ''.join([chr(t) for t in tokens if t < 128])
    
    def batch_decode(self, sequences, skip_special_tokens=True):
        return [self.decode(seq, skip_special_tokens) for seq in sequences]


class SimpleActor:
    """Simple actor that generates responses based on environment state."""
    def __init__(self, use_staged=True, model_name="mock"):
        self.use_staged = use_staged
        self.model_name = model_name
        self.step_count = 0
        
    def generate_sequences(self, batch_input):
        """Generate responses based on input."""
        self.step_count += 1
        
        # Extract batch size from input
        if hasattr(batch_input, 'batch') and batch_input.batch:
            if 'input_ids' in batch_input.batch:
                batch_size = len(batch_input.batch['input_ids'])
            else:
                batch_size = 1
        else:
            batch_size = 2  # Default
        
        responses = []
        for i in range(batch_size):
            if self.use_staged:
                response = self._generate_staged_response(i)
            else:
                response = self._generate_simple_response(i)
            responses.append(response)
        
        return DataProto.from_single_dict({
            'responses': np.array(responses)
        })
    
    def _generate_staged_response(self, batch_idx):
        """Generate staged response for environment interaction."""
        step = self.step_count
        
        # WebShop-style responses
        if step == 1:
            return """<plan>
I need to search for the requested product.
<memory query>previous searches</memory query>
</plan>

<action>
search[red shirt cotton medium]
</action>

<memory store>
Searching for red cotton shirt in medium size
</memory store>"""
        
        elif step == 2:
            return """<plan>
Found search results, need to select appropriate item
</plan>

<action>
click[Red Cotton T-Shirt - Medium]
</action>

<memory store>
Selected red cotton t-shirt product page
</memory store>"""
        
        elif step == 3:
            return """<action>
click[Buy Now]
</action>

<reflection>
Successfully found and purchased the requested item
</reflection>"""
        
        else:
            return """<action>
click[Back to Search]
</action>"""
    
    def _generate_simple_response(self, batch_idx):
        """Generate simple action."""
        actions = ["search[product]", "click[item 1]", "click[buy]", "click[back]"]
        return actions[self.step_count % len(actions)]


def test_with_webshop():
    """Test rollout with real WebShop environment."""
    print("\n" + "="*60)
    print("Testing with Real WebShop Environment")
    print("="*60)
    
    # Setup configuration
    config = SimpleConfig(env_name="webshop", batch_size=2)
    
    # Create real environments
    print("\n1. Creating WebShop environments...")
    try:
        envs, val_envs = make_envs(config)
        print("✓ WebShop environments created successfully")
    except Exception as e:
        print(f"✗ Failed to create WebShop: {e}")
        print("  Make sure WebShop data files are available")
        return
    
    # Setup rollout system
    tokenizer = SimpleTokenizer()
    rollout = OpenmanusRollout(config, tokenizer, None)
    actor = SimpleActor(use_staged=True)
    
    # Reset environment
    print("\n2. Environment Reset:")
    print("-" * 40)
    obs, infos = envs.reset()
    print(f"  Observations: {obs['text'][0][:100]}...")
    
    # Run a few steps
    print("\n3. Running Interaction Steps:")
    print("-" * 40)
    
    for step in range(3):
        print(f"\n  Step {step + 1}:")
        
        # Generate mock batch for actor
        batch = DataProto.from_single_dict({
            'input_ids': np.array([[1, 2, 3]] * config.data.train_batch_size),
            'attention_mask': np.array([[1, 1, 1]] * config.data.train_batch_size)
        })
        
        # Generate responses
        response_batch = actor.generate_sequences(batch)
        if hasattr(response_batch, 'batch') and response_batch.batch:
            responses = response_batch.batch['responses']
        else:
            responses = response_batch.non_tensor_batch.get('responses', [])
        
        # Process responses and get actions
        actions = []
        for i, response in enumerate(responses):
            action, parsed = rollout.process_response(
                response,
                episode_id=f"webshop_ep_{i}",
                step_id=step
            )
            
            # Show what's happening
            if i == 0:  # Show first agent only
                if isinstance(parsed, dict):
                    if parsed.get('plan'):
                        plan_text = parsed['plan'].get('plan', '') if isinstance(parsed['plan'], dict) else str(parsed.get('plan', ''))
                        if plan_text:
                            print(f"    Plan: {plan_text[:80]}...")
                    if parsed.get('action'):
                        action_text = parsed['action'].get('action', '') if isinstance(parsed['action'], dict) else str(parsed.get('action', ''))
                        if action_text:
                            print(f"    Action: {action_text}")
            
            # Use the action or a default
            actions.append(action if action else "search[test]")
        
        # Step environment
        next_obs, rewards, dones, infos = envs.step(actions)
        print(f"    Rewards: {rewards[:2]}")  # Show first 2
        print(f"    Done: {dones[:2]}")
        
        if dones.all():
            print("\n  Episodes completed!")
            break
    
    print("\n✓ WebShop test completed!")


def test_with_alfworld():
    """Test rollout with real AlfWorld environment."""
    print("\n" + "="*60)
    print("Testing with Real AlfWorld Environment")
    print("="*60)
    
    # Setup configuration
    config = SimpleConfig(env_name="alfworld/AlfredTWEnv", batch_size=1)
    
    # Create real environments
    print("\n1. Creating AlfWorld environments...")
    try:
        envs, val_envs = make_envs(config)
        print("✓ AlfWorld environments created successfully")
    except Exception as e:
        print(f"✗ Failed to create AlfWorld: {e}")
        print("  Make sure AlfWorld is properly installed")
        return
    
    # Setup rollout system
    tokenizer = SimpleTokenizer()
    rollout = OpenmanusRollout(config, tokenizer, None)
    
    # Create AlfWorld-specific actor
    class AlfWorldActor(SimpleActor):
        def _generate_staged_response(self, batch_idx):
            """Generate AlfWorld-specific responses."""
            step = self.step_count
            
            if step == 1:
                return """<plan>
I need to explore the room and understand the task.
</plan>

<action>
look
</action>"""
            elif step == 2:
                return """<action>
go to desk 1
</action>

<memory store>
Moved to desk 1
</memory store>"""
            elif step == 3:
                return """<action>
take pencil 1 from desk 1
</action>

<reflection>
Successfully picked up the pencil
</reflection>"""
            else:
                return """<action>
inventory
</action>"""
    
    actor = AlfWorldActor(use_staged=True)
    
    # Reset environment
    print("\n2. Environment Reset:")
    print("-" * 40)
    obs, infos = envs.reset()
    print(f"  Task: {obs['text'][0][:200]}...")
    
    # Run a few steps
    print("\n3. Running Interaction Steps:")
    print("-" * 40)
    
    for step in range(3):
        print(f"\n  Step {step + 1}:")
        
        # Generate batch
        batch = DataProto.from_single_dict({
            'input_ids': np.array([[1, 2, 3]] * config.data.train_batch_size),
            'attention_mask': np.array([[1, 1, 1]] * config.data.train_batch_size)
        })
        
        # Generate responses
        response_batch = actor.generate_sequences(batch)
        if hasattr(response_batch, 'batch') and response_batch.batch:
            responses = response_batch.batch['responses']
        else:
            responses = response_batch.non_tensor_batch.get('responses', [])
        
        # Process responses
        actions = []
        for i, response in enumerate(responses):
            action, parsed = rollout.process_response(
                response,
                episode_id=f"alfworld_ep_{i}",
                step_id=step
            )
            
            # Show the action
            if i == 0 and action:
                print(f"    Action: {action}")
            
            actions.append(action if action else "look")
        
        # Step environment
        next_obs, rewards, dones, infos = envs.step(actions)
        print(f"    Observation: {next_obs['text'][0][:100]}...")
        print(f"    Reward: {rewards[0]}")
        
        if dones.all():
            print("\n  Task completed!")
            break
    
    print("\n✓ AlfWorld test completed!")


def main():
    """Main test runner."""
    print("\n" + "="*60)
    print("OpenManus Rollout - Real Environment Test")
    print("="*60)
    print("\nThis test demonstrates rollout with REAL environments")
    print("(WebShop and AlfWorld) without training.")
    
    # Test with WebShop
    try:
        test_with_webshop()
    except Exception as e:
        print(f"\nWebShop test failed: {e}")
        print("This is expected if WebShop data files are not available")
    
    # Test with AlfWorld
    try:
        test_with_alfworld()
    except Exception as e:
        print(f"\nAlfWorld test failed: {e}")
        print("This is expected if AlfWorld is not installed")
    
    print("\n" + "="*60)
    print("✓ Real environment tests completed!")
    print("="*60)
    print("\nNote: These tests use simple mock actors.")
    print("You can replace the actor with a real LLM for actual testing:")
    print("1. Replace SimpleActor with your LLM-based actor")
    print("2. The actor should generate staged responses based on observations")
    print("3. The rollout system will parse stages and execute tools/actions")
    
    # Clean up
    for f in ['test_memory_webshop.md', 'test_memory_alfworld/AlfredTWEnv.md']:
        if os.path.exists(f):
            os.remove(f)
            print(f"\nCleaned up {f}")


if __name__ == "__main__":
    main()