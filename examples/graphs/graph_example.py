"""
Test graph builders
"""

import asyncio
import os
from dotenv import load_dotenv
from zeval.schemas.markdown import Markdown
from zeval.schemas.metadata import DocumentMetadata, UnitMetadata
from zeval.schemas.unit import BaseUnit
from zeval.synthetic_data.transforms.extractors import (
    SummaryExtractor,
    KeyphrasesExtractor,
    EntitiesExtractor
)
from zeval.synthetic_data.graphs import (
    EntityOverlapBuilder,
    KeyphraseOverlapBuilder,
)

# Load environment variables
load_dotenv()


def create_test_units():
    """Create test units for graph building"""
    
    # Create test markdown document
    markdown_content = """# Artificial Intelligence

Artificial intelligence (AI) is transforming various industries by automating tasks that previously required human intelligence. Machine learning and deep learning are key subfields of AI.

## Machine Learning

Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.

## Deep Learning

Deep learning uses neural networks with multiple layers to process complex patterns in data. It has revolutionized computer vision, natural language processing, and speech recognition.

## Natural Language Processing

Natural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. Modern NLP uses deep learning techniques like transformers.
"""
    
    doc = Markdown(
        content=markdown_content,
        metadata=DocumentMetadata(
            source="test_ai.md",
            source_type="local",
            file_type="markdown",
            content_length=len(markdown_content)
        )
    )
    
    # Manually create units (simulating splitter output)
    units = [
        BaseUnit(
            unit_id="unit_1",
            content="# Artificial Intelligence\n\nArtificial intelligence (AI) is transforming various industries by automating tasks that previously required human intelligence. Machine learning and deep learning are key subfields of AI.",
            source_doc_id=doc.doc_id,
            metadata=UnitMetadata(context_path="Artificial Intelligence")
        ),
        BaseUnit(
            unit_id="unit_2",
            content="## Machine Learning\n\nMachine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.",
            source_doc_id=doc.doc_id,
            metadata=UnitMetadata(context_path="Artificial Intelligence/Machine Learning")
        ),
        BaseUnit(
            unit_id="unit_3",
            content="## Deep Learning\n\nDeep learning uses neural networks with multiple layers to process complex patterns in data. It has revolutionized computer vision, natural language processing, and speech recognition.",
            source_doc_id=doc.doc_id,
            metadata=UnitMetadata(context_path="Artificial Intelligence/Deep Learning")
        ),
        BaseUnit(
            unit_id="unit_4",
            content="## Natural Language Processing\n\nNatural language processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language. Modern NLP uses deep learning techniques like transformers.",
            source_doc_id=doc.doc_id,
            metadata=UnitMetadata(context_path="Artificial Intelligence/Natural Language Processing")
        )
    ]
    
    return units


async def main():
    """Test graph building"""
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    print("\n" + "="*60)
    print("Testing Graph Builders")
    print("="*60)
    
    # 1. Create and enrich units
    units = create_test_units()
    print(f"\n✓ Created {len(units)} test units")
    
    # 2. Extract properties (keyphrases and entities)
    extractors = [
        KeyphrasesExtractor(
            model_uri="openai/gpt-4o-mini",
            api_key=api_key,
            max_num=5
        ),
        EntitiesExtractor(
            model_uri="openai/gpt-4o-mini",
            api_key=api_key,
            max_num=5
        )
    ]
    
    print("\n⏳ Extracting keyphrases and entities...")
    extractor = (
        KeyphrasesExtractor(model_uri="openai/gpt-4o-mini", api_key=api_key, max_num=5)
        | EntitiesExtractor(model_uri="openai/gpt-4o-mini", api_key=api_key, max_num=5)
    )
    enriched_units = await extractor.transform(units, max_concurrency=5)
    print("✓ Extraction completed")
    
    # Display extracted properties
    print("\n" + "="*60)
    print("Extracted Properties")
    print("="*60)
    for i, unit in enumerate(enriched_units, 1):
        print(f"\n[Unit {i}] {unit.metadata.context_path}")
        if unit.keyphrases:
            print(f"  Keyphrases: {unit.keyphrases}")
        if unit.entities:
            print(f"  Entities: {unit.entities}")
    
    # 3. Build graphs with different strategies
    print("\n" + "="*60)
    print("Building Graphs")
    print("="*60)
    
    # Strategy 1: Keyphrase similarity
    print("\n[1] Keyphrase Overlap Graph (Jaccard >= 0.2)")
    kp_builder = KeyphraseOverlapBuilder(threshold=0.2)  # Lower threshold
    kp_graph = kp_builder.build(enriched_units)
    print(f"  Nodes: {kp_graph.number_of_nodes()}")
    print(f"  Edges: {kp_graph.number_of_edges()}")
    
    # Show edges
    if kp_graph.number_of_edges() > 0:
        print("\n  Edges:")
        for u, v, data in kp_graph.edges(data=True):
            unit_u = kp_graph.nodes[u]['unit']
            unit_v = kp_graph.nodes[v]['unit']
            print(f"    {unit_u.metadata.context_path} <-> {unit_v.metadata.context_path}")
            print(f"      weight: {data.get('weight', 0):.3f}")
            # Show keyphrases for analysis
            kp_u = [kp.lower() for kp in (unit_u.keyphrases or [])]
            kp_v = [kp.lower() for kp in (unit_v.keyphrases or [])]
            print(f"      unit_u keyphrases: {unit_u.keyphrases}")
            print(f"      unit_v keyphrases: {unit_v.keyphrases}")
            # Simple exact match check
            common_exact = set(kp_u) & set(kp_v)
            if common_exact:
                print(f"      ✓ common (exact): {common_exact}")
    else:
        print("\n  No edges found (units don't have enough keyphrase overlap)")
    
    # Strategy 2: Entity similarity
    print(f"\n[2] Entity Overlap Graph (Jaccard >= 0.2)")
    entity_builder = EntityOverlapBuilder(threshold=0.2)  # Lower threshold
    entity_graph = entity_builder.build(enriched_units)
    print(f"  Nodes: {entity_graph.number_of_nodes()}")
    print(f"  Edges: {entity_graph.number_of_edges()}")
    
    # Show edges
    if entity_graph.number_of_edges() > 0:
        print("\n  Edges:")
        for u, v, data in entity_graph.edges(data=True):
            unit_u = entity_graph.nodes[u]['unit']
            unit_v = entity_graph.nodes[v]['unit']
            print(f"    {unit_u.metadata.context_path} <-> {unit_v.metadata.context_path}")
            print(f"      weight: {data.get('weight', 0):.3f}")
            # Show entities for analysis
            ent_u = [e.lower() for e in (unit_u.entities or [])]
            ent_v = [e.lower() for e in (unit_v.entities or [])]
            print(f"      unit_u entities: {unit_u.entities}")
            print(f"      unit_v entities: {unit_v.entities}")
            # Simple exact match check
            common_exact = set(ent_u) & set(ent_v)
            if common_exact:
                print(f"      ✓ common (exact): {common_exact}")
    else:
        print("\n  No edges found (units don't have enough entity overlap)")
    
    # 4. Demo: Find paths for multi-hop reasoning
    print("\n" + "="*60)
    print("Multi-hop Path Analysis")
    print("="*60)
    
    import networkx as nx
    
    # Use keyphrase graph for demo
    if kp_graph.number_of_edges() > 0:
        # Convert to undirected for path finding
        G_undirected = kp_graph.to_undirected()
        
        # Find all simple paths up to length 3
        print("\nFinding paths (max length 3):")
        path_count = 0
        for source in list(G_undirected.nodes())[:2]:  # Limit source nodes
            for target in G_undirected.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(
                            G_undirected, source, target, cutoff=2
                        ))
                        for path in paths:
                            if len(path) >= 3:  # Only show multi-hop
                                path_count += 1
                                path_names = [
                                    G_undirected.nodes[nid]['unit'].metadata.context_path.split('/')[-1]
                                    for nid in path
                                ]
                                print(f"  {' -> '.join(path_names)}")
                    except nx.NetworkXNoPath:
                        continue
        
        if path_count == 0:
            print("  No multi-hop paths found (graph might be too sparse)")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
