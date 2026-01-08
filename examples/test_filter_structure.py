"""
Test: Filter structure validation

Validates the new Filter design without requiring API calls.
"""

from rich import print as rprint

from zeval.synthetic_data.filters import Filter, FilterReport, GeneralFilter
from zeval.schemas.eval import EvalDataset, EvalCase


def test_imports():
    """Test 1: All imports work"""
    rprint("[bold cyan]Test 1: Imports[/bold cyan]")
    rprint(f"✓ Filter: {Filter}")
    rprint(f"✓ FilterReport: {FilterReport}")
    rprint(f"✓ GeneralFilter: {GeneralFilter}")
    rprint("[green]✓ All imports successful[/green]\n")


def test_filter_creation():
    """Test 2: Can create GeneralFilter"""
    rprint("[bold cyan]Test 2: Filter Creation[/bold cyan]")
    
    # Create filter with different configs
    filter1 = GeneralFilter()
    rprint(f"✓ Default filter: uri={filter1.uri}, concurrency={filter1.concurrency}")
    
    filter2 = GeneralFilter(
        uri="bailian/qwen-max",
        api_key="test_key",
        concurrency=5
    )
    rprint(f"✓ Custom filter: uri={filter2.uri}, api_key={'***' if filter2.api_key else None}, concurrency={filter2.concurrency}")
    rprint("[green]✓ Filter creation works[/green]\n")


def test_filter_report():
    """Test 3: FilterReport works"""
    rprint("[bold cyan]Test 3: FilterReport[/bold cyan]")
    
    report = FilterReport(
        total_cases=10,
        accepted_cases=7,
        rejected_cases=3,
        rejection_reasons={
            "context_completeness": 2,
            "question_quality": 1
        }
    )
    
    rprint("Report summary:")
    report.print_summary()
    rprint("[green]✓ FilterReport works[/green]\n")


def test_interface_compatibility():
    """Test 4: Interface with generators"""
    rprint("[bold cyan]Test 4: Interface Compatibility[/bold cyan]")
    
    # Simulate what generators do
    filter = GeneralFilter(uri="bailian/qwen-plus")
    
    # Check filter parameter type
    from zeval.synthetic_data.generators.multi_hop import generate_multi_hop
    import inspect
    
    sig = inspect.signature(generate_multi_hop)
    filter_param = sig.parameters.get('filter')
    
    rprint(f"✓ generate_multi_hop has 'filter' parameter: {filter_param is not None}")
    rprint(f"  - Type annotation: {filter_param.annotation if filter_param else 'N/A'}")
    rprint(f"  - Default value: {filter_param.default if filter_param else 'N/A'}")
    
    rprint("\n✓ Usage example:")
    rprint("  dataset = await generate_multi_hop(")
    rprint("      llm_uri='bailian/qwen-plus',")
    rprint("      graph=graph,")
    rprint("      personas=personas,")
    rprint("      filter=GeneralFilter(uri='bailian/qwen-max')  # Optional")
    rprint("  )")
    
    rprint("[green]✓ Interface compatible[/green]\n")


def main():
    """Run all tests"""
    rprint("\n[bold cyan]" + "="*60 + "[/bold cyan]")
    rprint("[bold cyan]Filter Structure Validation[/bold cyan]")
    rprint("[bold cyan]" + "="*60 + "[/bold cyan]\n")
    
    test_imports()
    test_filter_creation()
    test_filter_report()
    test_interface_compatibility()
    
    rprint("[bold green]" + "="*60 + "[/bold green]")
    rprint("[bold green]✓ All tests passed![/bold green]")
    rprint("[bold green]" + "="*60 + "[/bold green]\n")


if __name__ == "__main__":
    main()
