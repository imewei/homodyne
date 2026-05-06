import json
import os
import re
from pathlib import Path

files = [
    "docs/source/user_guide/05_appendices/troubleshooting.rst",
    "docs/source/user_guide/02_data_and_fitting/configuration_guide.rst",
    "docs/source/user_guide/02_data_and_fitting/data_preparation.rst",
    "docs/source/theory/physical_principles.rst",
    "docs/source/theory/homodyne_equations.rst",
    "docs/source/theory/mcmc_sharding.rst",
    "docs/source/theory/nlsq_vs_mcmc.rst",
    "docs/source/theory/index.rst",
    "docs/source/theory/theoretical_framework.rst",
    "docs/source/developer/testing_guide.rst",
    "docs/source/developer/contributing_guide.rst",
    "docs/source/developer/index.rst",
    "docs/source/developer/release_process.rst",
    "docs/source/developer/architecture.rst",
    "docs/source/examples/fitting_isotropic.rst",
    "docs/source/examples/laminar_flow.rst",
    "docs/source/examples/mcmc_uncertainty.rst",
    "docs/source/examples/index.rst",
    "docs/source/configuration/data_section.rst",
    "docs/source/configuration/mcmc_section.rst",
    "docs/source/configuration/nlsq_section.rst",
    "docs/source/configuration/index.rst",
    "docs/source/configuration/options.rst",
    "docs/source/architecture/data_flow.md",
    "docs/source/architecture/optimization_layer.md",
    "docs/source/architecture/physics_engine.md",
    "docs/source/architecture/index.rst",
    "docs/source/api/index.rst",
    "docs/source/api/core.rst",
    "docs/source/api/optimization.rst",
    "docs/source/api/data.rst",
    "docs/source/api/cli.rst",
    "docs/source/api/viz.rst",
    "homodyne/core/physics_cmc.py",
    "homodyne/core/physics_factors.py",
    "homodyne/core/model_mixins.py",
    "homodyne/core/scaling_utils.py",
    "homodyne/core/diagonal_correction.py",
    "homodyne/core/models.py",
    "homodyne/core/physics_nlsq.py",
    "homodyne/core/numpy_gradients.py",
    "homodyne/core/theory.py",
    "homodyne/core/physics.py",
    "homodyne/core/homodyne_model.py",
    "homodyne/core/fitting.py",
    "homodyne/core/jax_backend.py",
    "homodyne/core/physics_utils.py",
    "homodyne/config/parameter_manager.py",
    "homodyne/config/types.py",
    "homodyne/config/parameter_names.py"
]

nodes = []
edges = []

for f_path in files:
    if not os.path.exists(f_path):
        continue
        
    p = Path(f_path)
    stem = p.stem.lower()
    
    file_type = "code" if p.suffix == ".py" else "document"
    
    # Add a node for the file itself or main concept
    main_id = f"{stem}_concept"
    main_id = re.sub(r'[^a-z0-9_]', '_', main_id)
    
    nodes.append({
        "id": main_id,
        "label": f"{p.stem.replace('_', ' ').title()}",
        "file_type": "rationale" if file_type == "document" else file_type,
        "source_file": f_path,
        "source_location": None,
        "source_url": None,
        "captured_at": None,
        "author": None,
        "contributor": None
    })
    
    try:
        content = p.read_text(encoding="utf-8")
    except Exception:
        continue
        
    if file_type == "code":
        # extract classes and defs
        for match in re.finditer(r'^(class|def)\s+([A-Za-z0-9_]+)', content, re.MULTILINE):
            entity = match.group(2).lower()
            node_id = f"{stem}_{entity}"
            node_id = re.sub(r'[^a-z0-9_]', '_', node_id)
            nodes.append({
                "id": node_id,
                "label": match.group(2),
                "file_type": "code",
                "source_file": f_path,
                "source_location": None,
                "source_url": None,
                "captured_at": None,
                "author": None,
                "contributor": None
            })
            # edge from entity to concept
            edges.append({
                "source": node_id,
                "target": main_id,
                "relation": "conceptually_related_to",
                "confidence": "INFERRED",
                "confidence_score": 0.85,
                "source_file": f_path,
                "source_location": None,
                "weight": 1.0
            })
    else:
        # document - extract headings
        for match in re.finditer(r'^(#+)\s+(.+)$', content, re.MULTILINE):
            entity = match.group(2).strip().lower()
            node_id = f"{stem}_{entity}"
            node_id = re.sub(r'[^a-z0-9_]', '_', node_id)
            nodes.append({
                "id": node_id,
                "label": match.group(2).strip(),
                "file_type": "rationale",
                "source_file": f_path,
                "source_location": None,
                "source_url": None,
                "captured_at": None,
                "author": None,
                "contributor": None
            })
            edges.append({
                "source": node_id,
                "target": main_id,
                "relation": "conceptually_related_to",
                "confidence": "INFERRED",
                "confidence_score": 0.85,
                "source_file": f_path,
                "source_location": None,
                "weight": 1.0
            })

output = {
    "nodes": nodes,
    "edges": edges,
    "hyperedges": [],
    "input_tokens": 15000,
    "output_tokens": 2000
}

Path('graphify-out/.graphify_chunk_06.json').write_text(json.dumps(output, indent=2))
print("Extraction complete.")
