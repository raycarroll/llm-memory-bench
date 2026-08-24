#!/usr/bin/env bash
set -euo pipefail

# Dump all stored facts and pages as JSON.
# Output format: {"facts": [...], "pages": [...]}

facts=$(gbrain recall --json 2>/dev/null || echo '{"facts":[]}')
pages=$(gbrain list-pages --json --sort updated_desc 2>/dev/null || echo '{"pages":[]}')

python3 -c "
import json, sys
facts = json.loads('''$facts''').get('facts', [])
pages = json.loads('''$pages''').get('pages', [])

stored = []
for f in facts:
    stored.append({
        'type': 'fact',
        'content': f.get('fact', ''),
        'entity': f.get('entity_slug'),
        'kind': f.get('kind', 'fact'),
        'provenance': f.get('provenance', ''),
    })
for p in pages:
    stored.append({
        'type': 'page',
        'slug': p.get('slug', ''),
        'title': p.get('title', ''),
    })

json.dump({'stored': stored}, sys.stdout, indent=2)
"
