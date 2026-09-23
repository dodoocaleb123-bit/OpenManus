"""
Multi-turn rollout module.
Modular stage processing with memory.md integration.
"""

from .openmanus_rollout import OpenmanusRollout
from .modular_stages import ModularStageProcessor, DEFAULT_TOOLS
from .rollout_loop import TrajectoryCollector
from .tool_integration import GLOBAL_TOOL_REGISTRY, ToolRegistry, create_simple_tool_wrappers

__all__ = [
    'OpenmanusRollout',          # VERL-compatible rollout with modular stages
    'ModularStageProcessor', # Standalone modular processor
    'DEFAULT_TOOLS',         # Simple tool functions  
    'TrajectoryCollector',   # Legacy, kept for compatibility
    'GLOBAL_TOOL_REGISTRY',  # Global tool registry instance
    'ToolRegistry',          # Tool registry class
    'create_simple_tool_wrappers'  # Helper for tool wrappers
]