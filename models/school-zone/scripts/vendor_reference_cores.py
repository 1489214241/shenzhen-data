"""Extract previously downloaded official core code; never download or alter originals."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '学区归属模型_代码库' / 'third_party' / 'reference_cores'
OUT.mkdir(parents=True, exist_ok=True)
sources = {
    'egnn': ROOT / 'tmp/baseline_unrestricted_20261111/root/EGNN_official_model.py',
    'cnaps': ROOT / 'tmp/baseline_unrestricted_20261111/relation/cnaps_official_adaptation.py',
    'feat': ROOT / 'tmp/reference_broad_20261111/code/FEAT/model__models__feat.py',
}
manifest = {}
for name, path in sources.items():
    original = path.read_text(encoding='utf-8')
    if name == 'egnn':
        body = original[original.index('class NodeUpdateNetwork'):]
        body = body.replace('.to(tt.arg.device)', '.to(node_feat.device)')
        body = 'import torch\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom collections import OrderedDict\n' + body
    elif name == 'feat':
        body = original[original.index('class ScaledDotProductAttention'):original.index('class FEAT(')]
        body = 'import torch\nfrom torch import nn\nfrom torch.nn import functional as F\nimport numpy as np\n' + body
    else:
        body = original
    (OUT / (name + '.py')).write_text(body, encoding='utf-8')
    manifest[name] = dict(source=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          extracted_sha256=hashlib.sha256((OUT / (name + '.py')).read_bytes()).hexdigest())
(OUT / '__init__.py').write_text('', encoding='utf-8')
(OUT / 'provenance.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
