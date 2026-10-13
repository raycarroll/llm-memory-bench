"""Generate HTML visualizations of value benchmark results."""

import json
from pathlib import Path


def generate_html_visualization(result_path: Path, output_path: Path | None = None):
    """Generate interactive HTML visualization from value benchmark results."""

    # Load results
    with open(result_path) as f:
        data = json.load(f)

    # Load template
    template_path = Path(__file__).parent / "visualize_template.html"
    with open(template_path) as f:
        template = f.read()

    # Inject data
    html = template.replace(
        "{{DATA_PLACEHOLDER}}",
        json.dumps(data, indent=2)
    )

    # Write output
    if output_path is None:
        output_path = result_path.with_suffix('.html')

    output_path = Path(output_path)
    output_path.write_text(html)

    return output_path
