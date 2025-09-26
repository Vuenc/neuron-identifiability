"""
Component registration system inspired by GraphGym.
Allows for easy registration and instantiation of models, datasets, optimizers, etc.
"""

from typing import Any, Dict, Type, Callable, Optional
import inspect
from abc import ABC, abstractmethod


class Registry:
    """Central registry for all components in the framework."""
    
    def __init__(self):
        self._components: Dict[str, Dict[str, Any]] = {}
    
    def register(self, 
                 component_type: str, 
                 name: str, 
                 component: Type[Any], 
                 **kwargs) -> None:
        """Register a component with the given type and name.
        
        Args:
            component_type: Type of component (e.g., 'model', 'dataset', 'optimizer')
            name: Name to register the component under
            component: The component class or function
            **kwargs: Additional metadata for the component
        """
        if component_type not in self._components:
            self._components[component_type] = {}
        
        self._components[component_type][name] = {
            'component': component,
            'metadata': kwargs
        }
    
    def get(self, component_type: str, name: str) -> Any:
        """Get a registered component.
        
        Args:
            component_type: Type of component
            name: Name of the component
            
        Returns:
            The registered component
        """
        if component_type not in self._components:
            raise KeyError(f"Component type '{component_type}' not found in registry")
        
        if name not in self._components[component_type]:
            available = list(self._components[component_type].keys())
            raise KeyError(f"Component '{name}' not found in '{component_type}'. Available: {available}")
        
        return self._components[component_type][name]['component']
    
    def list_components(self, component_type: str) -> list:
        """List all components of a given type."""
        if component_type not in self._components:
            return []
        return list(self._components[component_type].keys())
    
    def build(self, component_type: str, name: str, **kwargs) -> Any:
        """Build a component instance with the given parameters.
        
        Args:
            component_type: Type of component
            name: Name of the component
            **kwargs: Parameters to pass to the component constructor
            
        Returns:
            Instantiated component
        """
        component = self.get(component_type, name)
        
        # Handle both classes and functions
        if inspect.isclass(component):
            return component(**kwargs)
        elif callable(component):
            return component(**kwargs)
        else:
            raise ValueError(f"Component '{name}' is not callable")
    
    def get_metadata(self, component_type: str, name: str) -> Dict[str, Any]:
        """Get metadata for a component."""
        if component_type not in self._components:
            raise KeyError(f"Component type '{component_type}' not found in registry")
        
        if name not in self._components[component_type]:
            raise KeyError(f"Component '{name}' not found in '{component_type}'")
        
        return self._components[component_type][name]['metadata']


# Global registry instance
REGISTRY = Registry()


def register(component_type: str, name: str, **kwargs):
    """Decorator for registering components."""
    def decorator(component):
        REGISTRY.register(component_type, name, component, **kwargs)
        return component
    return decorator


def build_component(component_type: str, name: str, **kwargs) -> Any:
    """Convenience function to build a component from the global registry."""
    return REGISTRY.build(component_type, name, **kwargs)


class BaseComponent(ABC):
    """Base class for all components in the framework."""
    
    @abstractmethod
    def __init__(self, **kwargs):
        pass
