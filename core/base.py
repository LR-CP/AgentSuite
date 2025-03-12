# agent_manager/base.py
import os
import re
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

# ------------------------------------------------------------------------------
# Base Classes
# ------------------------------------------------------------------------------

class BaseTool(ABC):
    """
    Generic base class for tools.
    """
    tool_type: str = "base_tool"
    
    def __init__(self, name: str, description: str, settings: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.settings = settings or {}

    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        Executes the tool with given input data.
        """
        pass
    
    @classmethod
    def get_required_settings(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary of settings required for this tool.
        Each setting is described with a dict containing default value and description.
        """
        return {}
    
    @classmethod
    def from_settings(cls, name: str, description: str, settings: Dict[str, Any]) -> 'BaseTool':
        """
        Factory method to create a tool instance from settings.
        """
        return cls(name, description, settings)


class BaseAgent(ABC):
    """
    Generic base class for agents.
    """
    agent_type: str = "base_agent"
    required_tools: List[str] = []
    
    def __init__(self, name: str, description: str, tools: Dict[str, BaseTool] = None, settings: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.tools = tools or {}
        self.settings = settings or {}
        self._validate_tools()
    
    def _validate_tools(self):
        """
        Validates that all required tools are provided.
        """
        for tool_name in self.required_tools:
            if tool_name not in self.tools:
                raise ValueError(f"Agent {self.name} requires tool {tool_name}, but it was not provided.")

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """
        Runs the agent with given input data.
        """
        pass
    
    @classmethod
    def get_required_settings(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary of settings required for this agent.
        Each setting is described with a dict containing default value and description.
        """
        return {}
    
    @classmethod
    def from_settings(cls, name: str, description: str, tools: Dict[str, BaseTool], settings: Dict[str, Any]) -> 'BaseAgent':
        """
        Factory method to create an agent instance from settings.
        """
        return cls(name, description, tools, settings)


# ------------------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------------------

class Registry:
    """
    Registry to store available agent and tool classes.
    """
    def __init__(self):
        self.agent_classes: Dict[str, Type[BaseAgent]] = {}
        self.tool_classes: Dict[str, Type[BaseTool]] = {}
    
    def register_agent(self, agent_class: Type[BaseAgent]):
        """
        Register an agent class.
        """
        self.agent_classes[agent_class.agent_type] = agent_class
        return agent_class  # Return the class to allow decorator usage
    
    def register_tool(self, tool_class: Type[BaseTool]):
        """
        Register a tool class.
        """
        self.tool_classes[tool_class.tool_type] = tool_class
        return tool_class  # Return the class to allow decorator usage
    
    def get_agent_class(self, agent_type: str) -> Type[BaseAgent]:
        """
        Get an agent class by type.
        """
        return self.agent_classes.get(agent_type)
    
    def get_tool_class(self, tool_type: str) -> Type[BaseTool]:
        """
        Get a tool class by type.
        """
        return self.tool_classes.get(tool_type)
    
    def get_agent_types(self) -> List[str]:
        """
        Get a list of registered agent types.
        """
        return list(self.agent_classes.keys())
    
    def get_tool_types(self) -> List[str]:
        """
        Get a list of registered tool types.
        """
        return list(self.tool_classes.keys())


# Create a global registry instance
registry = Registry()


# ------------------------------------------------------------------------------
# Agent Manager
# ------------------------------------------------------------------------------

class AgentManager:
    """
    A registry to add, delete, and retrieve agent instances.
    """
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.tools: Dict[str, BaseTool] = {}
    
    def add_tool(self, tool_id: str, tool: BaseTool):
        """
        Add a tool to the manager.
        """
        self.tools[tool_id] = tool
    
    def delete_tool(self, tool_id: str):
        """
        Delete a tool from the manager.
        """
        if tool_id in self.tools:
            # Check if the tool is used by any agent
            for agent_id, agent in self.agents.items():
                if tool_id in agent.tools:
                    raise ValueError(f"Cannot delete tool {tool_id} as it is used by agent {agent_id}")
            del self.tools[tool_id]
    
    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """
        Get a tool by ID.
        """
        return self.tools.get(tool_id)
    
    def list_tools(self) -> List[str]:
        """
        List all tool IDs.
        """
        return list(self.tools.keys())
    
    def add_agent(self, agent_id: str, agent: BaseAgent):
        """
        Add an agent to the manager.
        """
        self.agents[agent_id] = agent
    
    def delete_agent(self, agent_id: str):
        """
        Delete an agent from the manager.
        """
        if agent_id in self.agents:
            del self.agents[agent_id]
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an agent by ID.
        """
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """
        List all agent IDs.
        """
        return list(self.agents.keys())
    
    def create_agent(self, agent_id: str, agent_type: str, name: str, description: str, 
                     tool_mapping: Dict[str, str], settings: Dict[str, Any]) -> BaseAgent:
        """
        Create and register a new agent using the registry.
        
        Args:
            agent_id: The ID to register the agent under
            agent_type: The type of agent to create
            name: The name of the agent
            description: The description of the agent
            tool_mapping: A mapping from required tool names to tool IDs
            settings: Settings for the agent
            
        Returns:
            The created agent
        """
        agent_class = registry.get_agent_class(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Collect the tools required by the agent
        tools = {}
        for tool_name in agent_class.required_tools:
            if tool_name not in tool_mapping:
                raise ValueError(f"Required tool {tool_name} not specified for agent {name}")
            
            tool_id = tool_mapping[tool_name]
            tool = self.get_tool(tool_id)
            if not tool:
                raise ValueError(f"Tool {tool_id} not found")
            
            tools[tool_name] = tool
        
        # Create the agent
        agent = agent_class.from_settings(name, description, tools, settings)
        self.add_agent(agent_id, agent)
        return agent
    
    def create_tool(self, tool_id: str, tool_type: str, name: str, description: str, 
                   settings: Dict[str, Any]) -> BaseTool:
        """
        Create and register a new tool using the registry.
        
        Args:
            tool_id: The ID to register the tool under
            tool_type: The type of tool to create
            name: The name of the tool
            description: The description of the tool
            settings: Settings for the tool
            
        Returns:
            The created tool
        """
        tool_class = registry.get_tool_class(tool_type)
        if not tool_class:
            raise ValueError(f"Unknown tool type: {tool_type}")
        
        # Create the tool
        tool = tool_class.from_settings(name, description, settings)
        self.add_tool(tool_id, tool)
        return tool