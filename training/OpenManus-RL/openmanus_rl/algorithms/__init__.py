"""
Algorithms module for OpenManus-RL.

This module contains various RL algorithms implementations including:
- PPO (Proximal Policy Optimization) 
- GRPO (Group Relative Policy Optimization)
- GiGPO (Group-in-Group Policy Optimization)
- and other future algorithms
"""

from .gigpo import *

__all__ = [
    # GiGPO functions
    'compute_gigpo_outcome_advantage',
    'compute_step_discounted_returns',
    # Add other algorithms as they are implemented
]