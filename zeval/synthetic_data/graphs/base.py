"""Base class for graph builders"""

from abc import ABC, abstractmethod
import networkx as nx
from ...schemas.base import BaseUnit


class GraphBuilder(ABC):
    """
    Base class for graph construction strategies
    
    Core logic:
    Input = list of BaseUnit
    Output = NetworkX DiGraph with nodes + edges
    
    Each builder implements a different strategy for adding edges
    based on unit properties (embedding, keyphrases, entities, etc.)
    """
    
    @abstractmethod
    def build(self, units: list[BaseUnit]) -> nx.DiGraph:
        """
        Build graph from units
        
        Args:
            units: List of units to build graph from
            
        Returns:
            NetworkX directed graph with nodes and edges
        """
        pass
    
    def _create_base_graph(self, units: list[BaseUnit]) -> nx.DiGraph:
        """
        Create base graph with nodes only
        
        Each node stores the unit object for easy access
        """
        G = nx.DiGraph()
        for unit in units:
            G.add_node(unit.unit_id, unit=unit)
        return G
