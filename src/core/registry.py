from typing import Any, Dict, Type
import inspect
from abc import ABC, abstractmethod


class Registry:
    
    def __init__(self):
        self._components: Dict[str, Dict[str, Any]] = {}
    
    def register(self, 
                 component_type: str, 
                 name: str, 
                 component: Type[Any], 
                 **kwargs) -> None:
        if component_type not in self._components:
            self._components[component_type] = {}
        self._components[component_type][name] = {
            'component': component,
            'metadata': kwargs
        }
    
    def get(self, component_type: str, name: str) -> Any:
        if component_type not in self._components:
            raise KeyError(f"Component type '{component_type}' not found in registry")
        
        if name not in self._components[component_type]:
            available = list(self._components[component_type].keys())
            raise KeyError(f"Component '{name}' not found in '{component_type}'. Available: {available}")
        
        return self._components[component_type][name]['component']
    
    def list_components(self, component_type: str) -> list:
        if component_type not in self._components:
            return []
        return list(self._components[component_type].keys())
    
    def build(self, component_type: str, name: str, **kwargs) -> Any:
        component = self.get(component_type, name)
        if inspect.isclass(component):
            return component(**kwargs)
        elif callable(component):
            return component(**kwargs)
        else:
            raise ValueError(f"Component '{name}' is not callable")
    
    def get_metadata(self, component_type: str, name: str) -> Dict[str, Any]:
        if component_type not in self._components:
            raise KeyError(f"Component type '{component_type}' not found in registry")
        
        if name not in self._components[component_type]:
            raise KeyError(f"Component '{name}' not found in '{component_type}'")
        
        return self._components[component_type][name]['metadata']


REGISTRY = Registry()


def register(component_type: str, name: str, **kwargs):
    def decorator(component):
        REGISTRY.register(component_type, name, component, **kwargs)
        return component
    return decorator


def build_component(component_type: str, name: str, **kwargs) -> Any:
    return REGISTRY.build(component_type, name, **kwargs)


class BaseComponent(ABC):
    
    @abstractmethod
    def __init__(self, **kwargs):
        pass
