#!/usr/bin/env python3
"""Functional interpretation and report generation.

This module analyzes the functional implications of simplicial structure
and generates a summary report.
"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

# Default paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def load_all_results() -> dict:
    """Load all analysis results."""
    results = {}

    # Simplex counts
    simplex_path = RESULTS_DIR / "simplex_counts.csv"
    if simplex_path.exists():
        results['simplex'] = pd.read_csv(simplex_path).to_dict('records')

    # Null model comparison
    null_path = RESULTS_DIR / "null_model_comparison.csv"
    if null_path.exists():
        results['null_comparison'] = pd.read_csv(null_path).to_dict('records')

    # Cavity analysis
    cavity_path = RESULTS_DIR / "cavity_analysis.json"
    if cavity_path.exists():
        results['cavity'] = json.loads(cavity_path.read_text())

    # Fractal analysis
    fractal_path = RESULTS_DIR / "fractal_analysis.json"
    if fractal_path.exists():
        results['fractal'] = json.loads(fractal_path.read_text())

    return results


def generate_report(results: dict, output_path: Path) -> Path:
    """Generate summary report.

    Args:
        results: Dictionary of all analysis results
        output_path: Output markdown path

    Returns:
        Path to saved report
    """
    report = []
    report.append("# exp0418_brainSimplex — Analysis Report")
    report.append("")
    report.append(f"**Generated**: {datetime.now().isoformat()}")
    report.append("")

    # Executive summary
    report.append("## Executive Summary")
    report.append("")
    report.append("This report presents the topological analysis of neural connectome data,")
    report.append("focusing on directed simplicial structure, cavities, and fractal properties.")
    report.append("")

    # Simplex analysis
    report.append("## 1. Simplex Analysis")
    report.append("")
    if 'simplex' in results:
        df = pd.DataFrame(results['simplex'])
        report.append("### Simplex Distribution")
        report.append("")
        report.append("| Dimension | Count |")
        report.append("|-----------|-------|")
        for _, row in df.iterrows():
            report.append(f"| {row['dimension']} | {row['count']:,.0f} |")
        report.append("")

    # Null model comparison
    if 'null_comparison' in results:
        report.append("### Comparison with Null Models")
        report.append("")
        df = pd.DataFrame(results['null_comparison'])
        report.append("| Dim | Original | Null Mean | Z-score | Excess Ratio |")
        report.append("|-----|----------|-----------|---------|--------------|")
        for _, row in df.iterrows():
            if pd.notna(row['z_score']):
                report.append(f"| {int(row['dimension'])} | {row['original']:,.0f} | {row['null_mean']:.1f} | {row['z_score']:.2f} | {row['excess_ratio']:.2f}x |")
        report.append("")

    # Cavity analysis
    report.append("## 2. Cavity Analysis")
    report.append("")
    if 'cavity' in results:
        cavity = results['cavity']
        report.append(f"- **Euler characteristic**: χ = {cavity.get('euler_characteristic', 'N/A'):,}")
        report.append(f"- **Total simplices**: {cavity.get('total_simplices', 'N/A'):,}")
        report.append(f"- **Maximum simplex dimension**: {cavity.get('max_simplex_dim', 'N/A')}")
        report.append("")

        if 'betti_numbers' in cavity:
            report.append("### Betti Numbers (Approximate)")
            report.append("")
            for k, v in cavity['betti_numbers'].items():
                report.append(f"- β_{k} = {v:,}")
            report.append("")

    # Fractal analysis
    report.append("## 3. Fractal Analysis")
    report.append("")
    if 'fractal' in results:
        fractal = results['fractal']
        report.append(f"- **Box-counting dimension**: {fractal.get('box_counting_dimension', 'N/A'):.3f}")
        report.append(f"- **Correlation dimension**: {fractal.get('correlation_dimension', 'N/A')}")

        if 'power_law_test' in fractal:
            pl = fractal['power_law_test']
            if pl.get('valid'):
                report.append(f"- **Power-law fit**: R² = {pl.get('r_squared', np.nan):.3f}")
                report.append(f"- **Scaling exponent**: {pl.get('scaling_exponent', np.nan):.3f}")
                report.append(f"- **Interpretation**: {pl.get('interpretation', 'N/A')}")
        report.append("")

    # Interpretation
    report.append("## 4. Functional Interpretation")
    report.append("")
    report.append("### Implications for Information Processing")
    report.append("")
    report.append("The presence of high-dimensional directed simplices suggests:")
    report.append("")
    report.append("1. **Hierarchical information flow**: Directed simplices enforce ordered activation sequences")
    report.append("2. **Parallel processing paths**: Multiple simplices provide redundant computation pathways")
    report.append("3. **Temporal binding**: Clique structure supports synchronized activity patterns")
    report.append("")
    report.append("### Comparison with exp0414_simplexNet")
    report.append("")
    report.append("The simplex architecture in exp0414_simplexNet was inspired by these biological findings.")
    report.append("Key parallels:")
    report.append("")
    report.append("- Both use simplicial structure to organize computation")
    report.append("- Both have directed information flow (input → output)")
    report.append("- Both decompose into backbone and buffer components")
    report.append("")

    # Methods
    report.append("## 5. Methods")
    report.append("")
    report.append("### Data")
    report.append("")
    report.append("- Sample: 100-node random graph (for testing)")
    report.append("- Full analysis pending FlyWire connectome download")
    report.append("")
    report.append("### Algorithms")
    report.append("")
    report.append("- Directed simplex detection: Iterative clique enumeration")
    report.append("- Euler characteristic: Alternating sum of simplex counts")
    report.append("- Betti numbers: Heuristic approximation via cycle basis")
    report.append("- Fractal dimensions: Box-counting and correlation methods")
    report.append("")

    # Next steps
    report.append("## 6. Next Steps")
    report.append("")
    report.append("1. Download full FlyWire connectome (~130,000 neurons)")
    report.append("2. Run simplex detection on complete data")
    report.append("3. Compare with biological null models (degree-preserving rewiring)")
    report.append("4. Analyze regional variations (mushroom body, optic lobes, etc.)")
    report.append("5. Investigate cell-type-specific simplicial structure")
    report.append("")

    # Write report
    content = "\n".join(report)
    output_path.write_text(content)

    print(f"Saved report: {output_path}")
    return output_path


def main():
    """Main entry point."""
    print("=" * 60)
    print("Report Generation")
    print("=" * 60)
    print()

    results = load_all_results()

    if not results:
        print("No results found. Run analysis scripts first.")
        return

    print(f"Loaded results from {len(results)} analyses")
    for key in results:
        print(f"  - {key}")
    print()

    output_path = PROJECT_ROOT / "report.md"
    generate_report(results, output_path)

    print()
    print("=" * 60)
    print("Report generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
