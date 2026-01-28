"""Overlap-based graph builders using Jaccard similarity with fuzzy matching"""

import networkx as nx
from collections import Counter
from .base import GraphBuilder
from ...schemas.unit import BaseUnit


class EntityOverlapBuilder(GraphBuilder):
    """
    Build graph based on entity overlap
    
    Connects units that share common entities (people, organizations, products, etc.).
    Uses Jaccard similarity with fuzzy string matching for robustness.
    
    Algorithm:
    ----------
    1. Noise Filtering:
       - Identify top 5% most frequent entities across all units
       - Remove these "noisy" entities (e.g., "AI", "system") from comparison
       - This focuses on distinctive entities rather than common terms
    
    2. Fuzzy Jaccard Similarity:
       For each pair of units:
       - Normalize entity strings to lowercase
       - For each entity in unit1, find best match in unit2 using:
         * Exact match (case-insensitive), OR
         * Fuzzy match via Jaro-Winkler distance >= 0.9
           (handles variants like "Google" vs "Google Cloud")
       - Track matched pairs to avoid double-counting
       - Calculate: matched_count / total_unique_count
    
    3. Edge Creation:
       - If similarity >= threshold, create bidirectional edge
       - Edge weight = Jaccard similarity score
    
    Threshold Rationale:
    -------------------
    Default: 0.2 (20% overlap)
    
    Examples:
    - High relevance (0.4-0.6): Both discuss "TensorFlow applications"
      Shared: TensorFlow, Google, Python, ML → 4/7 = 0.57
    
    - Medium relevance (0.2-0.4): Related but different focus
      Shared: Python, neural networks → 2/8 = 0.25
    
    - Low relevance (<0.2): Tangential connection
      Shared: AI → 1/9 = 0.11 (filtered out)
    
    Args:
        threshold: Minimum Jaccard similarity to create edge (default: 0.2)
                  Lower = more connections, Higher = stricter relevance
        fuzzy_threshold: Minimum fuzzy match score for strings (default: 0.9)
                        Higher = require closer string match
        noise_cutoff: Filter top X% most common entities as noise (default: 0.05)
                     Higher = filter more common entities
    
    Example:
        Unit A: entities = ["TensorFlow", "Google", "Python"]
        Unit B: entities = ["PyTorch", "Meta", "Python"]
        
        Common entities: {"Python"}
        Jaccard = 1 / 5 = 0.2 → Edge created (at threshold=0.2)
        
        Unit C: entities = ["TensorFlow", "Google"]
        Unit D: entities = ["TensorFlow", "Google Cloud"]
        
        With fuzzy matching: "Google" ≈ "Google Cloud" (Jaro-Winkler = 0.95)
        Jaccard = 2 / 3 = 0.67 → Strong edge
    """
    
    def __init__(
        self,
        threshold: float = 0.2,
        fuzzy_threshold: float = 0.9,
        noise_cutoff: float = 0.05
    ):
        self.threshold = threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.noise_cutoff = noise_cutoff
        
        # Lazy import rapidfuzz
        try:
            from rapidfuzz import distance
            self.distance = distance
        except ImportError:
            raise ImportError(
                "rapidfuzz is required for fuzzy matching. "
                "Install with: pip install rapidfuzz"
            )
    
    def build(self, units: list[BaseUnit]) -> nx.DiGraph:
        """Build graph with entity overlap edges"""
        G = self._create_base_graph(units)
        
        # Filter units with entities
        units_with_entities = [u for u in units if u.entities]
        
        if len(units_with_entities) < 2:
            return G
        
        # Identify noisy entities (too common)
        noisy_entities = self._get_noisy_items(units_with_entities)
        
        # Compare all pairs
        for i, unit1 in enumerate(units_with_entities):
            for unit2 in units_with_entities[i+1:]:
                # Filter out noisy entities
                entities1 = [
                    e for e in unit1.entities 
                    if e.lower() not in noisy_entities
                ]
                entities2 = [
                    e for e in unit2.entities 
                    if e.lower() not in noisy_entities
                ]
                
                if not entities1 or not entities2:
                    continue
                
                # Calculate overlap with fuzzy matching
                similarity = self._fuzzy_jaccard_similarity(entities1, entities2)
                
                if similarity >= self.threshold:
                    G.add_edge(
                        unit1.unit_id,
                        unit2.unit_id,
                        type="entity_overlap",
                        weight=similarity
                    )
                    # Bidirectional
                    G.add_edge(
                        unit2.unit_id,
                        unit1.unit_id,
                        type="entity_overlap",
                        weight=similarity
                    )
        
        return G
    
    def _get_noisy_items(self, units: list[BaseUnit]) -> set[str]:
        """
        Get noisy entities that appear too frequently
        
        Args:
            units: Units to analyze
            
        Returns:
            Set of lowercased noisy entity strings
        """
        all_entities = []
        for unit in units:
            if unit.entities:
                all_entities.extend([e.lower() for e in unit.entities])
        
        if not all_entities:
            return set()
        
        # Count frequencies
        entity_counts = Counter(all_entities)
        
        # Get top X% as noise
        num_unique = len(entity_counts)
        num_noisy = max(1, int(num_unique * self.noise_cutoff))
        
        noisy_list = [
            entity for entity, _ in entity_counts.most_common(num_noisy)
        ]
        
        return set(noisy_list)
    
    def _fuzzy_jaccard_similarity(
        self,
        list1: list[str],
        list2: list[str]
    ) -> float:
        """
        Calculate Jaccard similarity with fuzzy string matching
        
        Args:
            list1: First list of strings
            list2: Second list of strings
            
        Returns:
            Jaccard similarity in [0, 1]
        """
        if not list1 or not list2:
            return 0.0
        
        # Normalize to lowercase
        set1 = [s.lower() for s in list1]
        set2 = [s.lower() for s in list2]
        
        # Track matched items to avoid double counting
        matched_in_set2 = set()
        overlaps = []
        
        for item1 in set1:
            best_match = False
            for item2 in set2:
                if item2 in matched_in_set2:
                    continue
                
                # Exact match or fuzzy match
                if item1 == item2:
                    best_match = True
                    matched_in_set2.add(item2)
                    overlaps.append(True)
                    break
                else:
                    # Fuzzy match using Jaro-Winkler
                    similarity = 1 - self.distance.JaroWinkler.distance(item1, item2)
                    if similarity >= self.fuzzy_threshold:
                        best_match = True
                        matched_in_set2.add(item2)
                        overlaps.append(True)
                        break
            
            if not best_match:
                overlaps.append(False)
        
        # Add unmatched items from set2
        for item2 in set2:
            if item2 not in matched_in_set2:
                overlaps.append(False)
        
        # Jaccard = matches / total_unique
        if not overlaps:
            return 0.0
        
        return sum(overlaps) / len(overlaps)


class KeyphraseOverlapBuilder(GraphBuilder):
    """
    Build graph based on keyphrase overlap
    
    Connects units that share common keyphrases (topics, concepts, themes).
    Uses Jaccard similarity with fuzzy string matching for robustness.
    
    Algorithm:
    ----------
    1. Noise Filtering:
       - Identify top 5% most frequent keyphrases across all units
       - Remove these "noisy" keyphrases (e.g., "technology", "system") from comparison
       - This focuses on specific topics rather than generic terms
    
    2. Fuzzy Jaccard Similarity:
       For each pair of units:
       - Normalize keyphrase strings to lowercase
       - For each keyphrase in unit1, find best match in unit2 using:
         * Exact match (case-insensitive), OR
         * Fuzzy match via Jaro-Winkler distance >= 0.9
           (handles variants like "machine learning" vs "Machine Learning")
       - Track matched pairs to avoid double-counting
       - Calculate: matched_count / total_unique_count
    
    3. Edge Creation:
       - If similarity >= threshold, create bidirectional edge
       - Edge weight = Jaccard similarity score
    
    Threshold Rationale:
    -------------------
    Default: 0.15 (15% overlap)
    
    Note: Keyphrases are more abstract than entities, so we use a lower
    threshold. A 15% conceptual overlap often indicates related topics.
    
    Examples:
    - High relevance (0.3-0.5): Both discuss same core topic
      Shared: machine learning, neural networks, deep learning → 3/7 = 0.43
    
    - Medium relevance (0.15-0.3): Related but different angles
      Shared: AI, algorithms → 2/8 = 0.25
    
    - Low relevance (<0.15): Weak topical connection
      Shared: technology → 1/9 = 0.11 (filtered out)
    
    Args:
        threshold: Minimum Jaccard similarity to create edge (default: 0.15)
                  Lower = more connections, Higher = stricter topical relevance
        fuzzy_threshold: Minimum fuzzy match score for strings (default: 0.9)
                        Higher = require closer string match
        noise_cutoff: Filter top X% most common keyphrases as noise (default: 0.05)
                     Higher = filter more common keyphrases
    
    Example:
        Unit A: keyphrases = ["machine learning", "neural networks", "AI"]
        Unit B: keyphrases = ["deep learning", "neural networks", "AI"]
        
        Common: {"neural networks", "AI"}
        Jaccard = 2 / 4 = 0.5 → Strong topical connection
    """
    
    def __init__(
        self,
        threshold: float = 0.15,
        fuzzy_threshold: float = 0.9,
        noise_cutoff: float = 0.05
    ):
        self.threshold = threshold
        self.fuzzy_threshold = fuzzy_threshold
        self.noise_cutoff = noise_cutoff
        
        # Lazy import rapidfuzz
        try:
            from rapidfuzz import distance
            self.distance = distance
        except ImportError:
            raise ImportError(
                "rapidfuzz is required for fuzzy matching. "
                "Install with: pip install rapidfuzz"
            )
    
    def build(self, units: list[BaseUnit]) -> nx.DiGraph:
        """Build graph with keyphrase overlap edges"""
        G = self._create_base_graph(units)
        
        # Filter units with keyphrases
        units_with_kp = [u for u in units if u.keyphrases]
        
        if len(units_with_kp) < 2:
            return G
        
        # Identify noisy keyphrases
        noisy_keyphrases = self._get_noisy_items(units_with_kp)
        
        # Compare all pairs
        for i, unit1 in enumerate(units_with_kp):
            for unit2 in units_with_kp[i+1:]:
                # Filter out noisy keyphrases
                kp1 = [
                    kp for kp in unit1.keyphrases 
                    if kp.lower() not in noisy_keyphrases
                ]
                kp2 = [
                    kp for kp in unit2.keyphrases 
                    if kp.lower() not in noisy_keyphrases
                ]
                
                if not kp1 or not kp2:
                    continue
                
                # Calculate overlap with fuzzy matching
                similarity = self._fuzzy_jaccard_similarity(kp1, kp2)
                
                if similarity >= self.threshold:
                    G.add_edge(
                        unit1.unit_id,
                        unit2.unit_id,
                        type="keyphrase_overlap",
                        weight=similarity
                    )
                    # Bidirectional
                    G.add_edge(
                        unit2.unit_id,
                        unit1.unit_id,
                        type="keyphrase_overlap",
                        weight=similarity
                    )
        
        return G
    
    def _get_noisy_items(self, units: list[BaseUnit]) -> set[str]:
        """Get noisy keyphrases that appear too frequently"""
        all_keyphrases = []
        for unit in units:
            if unit.keyphrases:
                all_keyphrases.extend([kp.lower() for kp in unit.keyphrases])
        
        if not all_keyphrases:
            return set()
        
        kp_counts = Counter(all_keyphrases)
        num_unique = len(kp_counts)
        num_noisy = max(1, int(num_unique * self.noise_cutoff))
        
        noisy_list = [
            kp for kp, _ in kp_counts.most_common(num_noisy)
        ]
        
        return set(noisy_list)
    
    def _fuzzy_jaccard_similarity(
        self,
        list1: list[str],
        list2: list[str]
    ) -> float:
        """Calculate Jaccard similarity with fuzzy string matching"""
        if not list1 or not list2:
            return 0.0
        
        # Normalize to lowercase
        set1 = [s.lower() for s in list1]
        set2 = [s.lower() for s in list2]
        
        # Track matched items
        matched_in_set2 = set()
        overlaps = []
        
        for item1 in set1:
            best_match = False
            for item2 in set2:
                if item2 in matched_in_set2:
                    continue
                
                # Exact match or fuzzy match
                if item1 == item2:
                    best_match = True
                    matched_in_set2.add(item2)
                    overlaps.append(True)
                    break
                else:
                    # Fuzzy match using Jaro-Winkler
                    similarity = 1 - self.distance.JaroWinkler.distance(item1, item2)
                    if similarity >= self.fuzzy_threshold:
                        best_match = True
                        matched_in_set2.add(item2)
                        overlaps.append(True)
                        break
            
            if not best_match:
                overlaps.append(False)
        
        # Add unmatched items from set2
        for item2 in set2:
            if item2 not in matched_in_set2:
                overlaps.append(False)
        
        if not overlaps:
            return 0.0
        
        return sum(overlaps) / len(overlaps)
