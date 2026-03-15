# medguard

Healthcare-specific LLM guardrails middleware for clinical safety.

Wraps any LLM (Anthropic, OpenAI, Ollama) with five safety layers before and after each inference call:

| Layer | What it does |
|-------|-------------|
| **PHI Detection** | Detects and redacts SSN, DOB, phone, email, MRN, and labeled names |
| **Clinical Scope** | Blocks or warns when queries fall outside medical scope (legal, financial) |
| **Drug Safety** | Checks drug-drug interactions and contraindications via OpenFDA + RxNorm |
| **Hallucination Detection** | Flags fake drug names, impossible dosages, and overconfident claims |
| **Output Annotation** | Inlines `[WARNING: ...]` markers on flagged response spans |

---

## Install

```bash
pip install medguard                          # core only
pip install "medguard[anthropic]"             # + Anthropic LLM
pip install "medguard[anthropic,openai]"      # + OpenAI-compatible
pip install "medguard[anthropic,openai,nlp]"  # + Presidio PHI engine
```

## Quick start

```python
from medguard import MedGuard

mg = MedGuard()

# Check text without calling an LLM
result = mg.check("Patient SSN: 123-45-6789 is taking warfarin and aspirin.")
print(result.phi_result.phi_detected)      # True
print(result.phi_result.processed)         # "Patient SSN: [REDACTED] is taking ..."
print(result.drug_result.highest_severity) # InteractionSeverity.HIGH

# Full guardrailed LLM chat
import asyncio
response = asyncio.run(mg.achat("What are side effects of metformin?"))
```

## Run the API server

```bash
medguard serve               # default port 8080
medguard serve --port 9090   # custom port
python -m medguard serve     # alternative
```

Endpoints:

```
GET  /v1/health                     dependency status (RxNorm, OpenFDA)
POST /v1/chat                       guardrailed LLM chat (stream: true supported)
POST /v1/check/phi                  standalone PHI detection / redaction
POST /v1/check/drug-interactions    standalone drug interaction check
GET  /docs                          Swagger UI
```

### Example requests

```bash
# PHI redaction
curl -X POST http://localhost:8080/v1/check/phi \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient John Smith, SSN: 123-45-6789", "mode": "redact"}'

# Drug interaction check
curl -X POST http://localhost:8080/v1/check/drug-interactions \
  -H "Content-Type: application/json" \
  -d '{"drugs": ["warfarin", "aspirin"]}'

# Guardrailed chat
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the max dose of ibuprofen?"}]}'
```

## Docker

```bash
ANTHROPIC_API_KEY=sk-... docker compose -f docker/docker-compose.yml up
```

All guardrails are configurable via environment variables:

```bash
MEDGUARD_GUARDRAILS__PHI_DETECTION__MODE=block          # redact | flag | block
MEDGUARD_GUARDRAILS__DRUG_SAFETY__SEVERITY_THRESHOLD=high
MEDGUARD_GUARDRAILS__SCOPE_ENFORCEMENT__ACTION=block    # warn | block
MEDGUARD_LLM__PROVIDER=openai
MEDGUARD_LLM__MODEL=gpt-4o
```

## Configuration

Config is loaded from `~/.medguard/config.json` (auto-created on first run).

```json
{
  "guardrails": {
    "phi_detection":          { "enabled": true, "mode": "redact", "engine": "regex" },
    "drug_safety":            { "enabled": true, "severity_threshold": "moderate" },
    "scope_enforcement":      { "enabled": true, "action": "warn" },
    "hallucination_detection":{ "enabled": true, "confidence_threshold": 0.7 }
  },
  "llm": {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "api_key_env": "ANTHROPIC_API_KEY"
  },
  "api": { "host": "0.0.0.0", "port": 8080 }
}
```

PHI engine options:
- `"regex"` — zero-dependency, ships with the library (default)
- `"presidio"` — Microsoft Presidio + spaCy, higher recall (`pip install "medguard[nlp]"`)

## Architecture

```
User input
    │
    ▼
[PHI Detection] ──── redact / block
    │
[Scope Enforcement] ── warn / block
    │
[Drug Safety Check] ── warn / block  ◄── OpenFDA API + RxNorm + static table
    │
    ▼
   LLM
    │
    ▼
[Hallucination Detection] ── flag / block  ◄── RxNorm + SNOMED bundle
    │
    ▼
Annotated response
```

Each guardrail runs in an isolated `try/except` — an API timeout in drug safety never blocks the full request.

## NeMo Guardrails integration

```python
from nemoguardrails import RailsConfig, LLMRails
from medguard import MedGuard
from medguard.integrations.nemo import MEDICAL_COLANG_CONFIG

mg = MedGuard()
config = RailsConfig.from_content(colang_content=MEDICAL_COLANG_CONFIG)
rails = LLMRails(config)

for name, action in mg.as_nemo_actions().items():
    rails.register_action(action, name=name)
```

## Extending medguard

Every guardrail implements a Protocol interface. Register custom engines via entry points:

```toml
# In your package's pyproject.toml
[project.entry-points."medguard.phi_engines"]
my_engine = "my_package:MyPHIEngine"

[project.entry-points."medguard.interaction_sources"]
drugbank = "medguard_drugbank:DrugBankClient"
```

Then select it in config: `"phi_detection": {"engine": "my_engine"}`.

**Contribution targets:**
- New drug interaction sources (`InteractionSourceProtocol`)
- New PHI engines (`PHIEngineProtocol`)
- Curated drug interaction data (`medguard/knowledge/data/drug_interactions.csv`)
- Clinical Colang flows (`medguard/integrations/nemo.py`)
- Language-specific medical fact checkers

## Development

```bash
git clone https://github.com/sarvanithin/Medguard
cd Medguard
pip install -e ".[anthropic,dev]"
pytest tests/ -m "not integration"   # unit tests (no network)
pytest tests/ -m integration         # hits real OpenFDA / RxNorm APIs
```

## Data sources

| Source | Used for |
|--------|----------|
| [RxNorm API](https://rxnav.nlm.nih.gov/) | Drug name normalization |
| [OpenFDA Drug Labels](https://open.fda.gov/apis/drug/label/) | Interaction + contraindication text |
| [OpenFDA Adverse Events](https://open.fda.gov/apis/drug/event/) | Adverse event counts |
| Bundled SNOMED-CT subset | Medical terminology validation |
| Curated static table | Highest-risk drug pairs (offline fallback) |

## License

Apache 2.0
