"""
Integration layer between the new tools system and modular rollout.
Wraps octotools-style tools for use in staged rollout.
"""

import os
import importlib
from typing import Dict, Any, Optional, List
from openmanus_rl.tools.base import BaseTool


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.tool_instances: Dict[str, Any] = {}
    
    def discover_tools(self, tools_dir: str = "openmanus_rl/tools"):
        """Auto-discover all available tools in the tools directory."""
        tools_found = []
        
        # List all subdirectories in tools
        for item in os.listdir(tools_dir):
            tool_path = os.path.join(tools_dir, item)
            if os.path.isdir(tool_path) and not item.startswith('_'):
                # Check if it has a tool.py file
                tool_module_path = os.path.join(tool_path, 'tool.py')
                if os.path.exists(tool_module_path):
                    tools_found.append(item)
        
        print(f"Discovered tools: {tools_found}")
        return tools_found
    
    def load_tool(self, tool_name: str, model_string: Optional[str] = None) -> Optional[BaseTool]:
        """Load a specific tool by name."""
        try:
            # Import the tool module
            module_path = f"openmanus_rl.tools.{tool_name}.tool"
            module = importlib.import_module(module_path)
            
            # Find the tool class (usually named after the tool)
            # Convert snake_case to CamelCase
            class_name = ''.join(word.capitalize() for word in tool_name.split('_'))
            if hasattr(module, class_name):
                tool_class = getattr(module, class_name)
            else:
                # Try to find any class that inherits from BaseTool
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, BaseTool) and obj != BaseTool:
                        tool_class = obj
                        break
                else:
                    print(f"No tool class found in {module_path}")
                    return None
            
            # Instantiate the tool
            if tool_class.require_llm_engine and model_string:
                tool_instance = tool_class(model_string=model_string)
            else:
                tool_instance = tool_class()
            
            self.tools[tool_name] = tool_instance
            return tool_instance
            
        except Exception as e:
            print(f"Failed to load tool {tool_name}: {e}")
            return None
    
    def register_tool(self, name: str, tool: BaseTool):
        """Register a tool instance."""
        self.tools[name] = tool
    
    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> str:
        """Execute a tool with given parameters."""
        if tool_name not in self.tools:
            # Try to load it
            tool = self.load_tool(tool_name)
            if not tool:
                return f"Error: Tool '{tool_name}' not found"
        else:
            tool = self.tools[tool_name]
        
        try:
            # Execute the tool
            # Map common parameter names
            if 'query' in params and 'text' in tool.input_types:
                params['text'] = params.pop('query')
            elif 'expression' in params and 'code' in tool.input_types:
                params['code'] = params.pop('expression')
            
            result = tool.execute(**params)
            
            # Format result as string
            if isinstance(result, dict):
                return str(result.get('output', result))
            else:
                return str(result)
                
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self.tools.keys())
    
    def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get metadata for a specific tool."""
        if tool_name in self.tools:
            return self.tools[tool_name].get_metadata()
        return {}


def create_tool_wrapper(registry: ToolRegistry):
    """
    Create a wrapper function that matches the simple tool interface.
    This allows octotools to work with our modular system.
    """
    def tool_wrapper(params: Dict[str, Any]) -> str:
        # Extract tool name from params if specified
        tool_name = params.pop('tool', None)
        if not tool_name:
            # Try to infer from params
            if 'image' in params:
                tool_name = 'object_detector'
            elif 'url' in params:
                tool_name = 'url_text_extractor'
            elif 'arxiv' in str(params.get('query', '')).lower():
                tool_name = 'arxiv_paper_searcher'
            else:
                tool_name = 'google_search'  # Default
        
        return registry.execute_tool(tool_name, params)
    
    return tool_wrapper


# Quick tool wrappers for common tools
def create_simple_tool_wrappers(registry: ToolRegistry) -> Dict[str, Any]:
    """Create simple wrapper functions for common tools."""
    
    def search_wrapper(params: dict) -> str:
        """Google search wrapper."""
        return registry.execute_tool('google_search', params)
    
    def arxiv_wrapper(params: dict) -> str:
        """ArXiv search wrapper."""
        return registry.execute_tool('arxiv_paper_searcher', params)
    
    def wikipedia_wrapper(params: dict) -> str:
        """Wikipedia search wrapper."""
        return registry.execute_tool('wikipedia_knowledge_searcher', params)
    
    def python_wrapper(params: dict) -> str:
        """Python code generator wrapper."""
        return registry.execute_tool('python_code_generator', params)
    
    def image_caption_wrapper(params: dict) -> str:
        """Image captioner wrapper."""
        return registry.execute_tool('image_captioner', params)
    
    def object_detect_wrapper(params: dict) -> str:
        """Object detector wrapper."""
        return registry.execute_tool('object_detector', params)
    
    def url_extract_wrapper(params: dict) -> str:
        """URL text extractor wrapper."""
        return registry.execute_tool('url_text_extractor', params)
    
    return {
        'search': search_wrapper,
        'arxiv': arxiv_wrapper,
        'wikipedia': wikipedia_wrapper,
        'python': python_wrapper,
        'caption': image_caption_wrapper,
        'detect': object_detect_wrapper,
        'url': url_extract_wrapper,
    }


# Global registry instance
GLOBAL_TOOL_REGISTRY = ToolRegistry()

# Auto-discover tools on import
import os
tools_path = os.path.join(os.path.dirname(__file__), '..', 'tools')
if os.path.exists(tools_path):
    GLOBAL_TOOL_REGISTRY.discover_tools(tools_path)