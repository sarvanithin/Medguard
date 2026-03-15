# Medguard

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

# Ollama (local, no API key needed)
MEDGUARD_LLM__PROVIDER=ollama
MEDGUARD_LLM__MODEL=llama3.2
MEDGUARD_LLM__BASE_URL=http://localhost:11434/v1
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

LLM provider options:

| Provider | Config |
|----------|--------|
| Anthropic (default) | `"provider": "anthropic", "model": "claude-haiku-4-5-20251001"` |
| OpenAI | `"provider": "openai", "model": "gpt-4o"` |
| **Ollama (local)** | `"provider": "ollama", "model": "llama3.2"` |
| Azure / custom | `"provider": "openai", "base_url": "https://..."` |

### Ollama (fully local, no API key)

```bash
# 1. Install and start Ollama
ollama pull llama3.2         # or mistral, phi3, gemma2, etc.
ollama serve                 # runs on localhost:11434

# 2. Run medguard with Ollama
MEDGUARD_LLM__PROVIDER=ollama \
MEDGUARD_LLM__MODEL=llama3.2 \
medguard serve
```

Or in Python:

```python
from medguard import MedGuard
from medguard.config import MedGuardConfig, LLMConfig

mg = MedGuard(config=MedGuardConfig(
    llm=LLMConfig(provider="ollama", model="llama3.2")
))

import asyncio
response = asyncio.run(mg.achat("What are the side effects of metformin?"))
```

PHI engine options:
- `"regex"` — zero-dependency, ships with the library (default)
- `"presidio"` — Microsoft Presidio + spaCy, higher recall (`pip install "medguard[nlp]"`)

## Architecture

```
User Input
    │
    ▼
MedGuard.achat("I take warfarin and aspirin, SSN 123-45-6789")
    │
    ▼
GuardrailPipeline.run(text)
    │
    ├─── Stage 1: INPUT (runs sequentially)
    │         │
    │         ├─ PHIDetector.detect(text)
    │         │       RegexPHIEngine scans with compiled regexes
    │         │       → matches: [{SSN, "123-45-6789"}]
    │         │       → processed: "I take warfarin and aspirin, SSN [REDACTED]"
    │         │       mode=redact → stores redacted text in ctx.processed_input
    │         │       mode=block  → sets ctx.blocked=True, stops here
    │         │
    │         └─ ScopeEnforcer.check(text)
    │                 KeywordScopeClassifier scans for clinical vs out-of-scope keywords
    │                 → in_scope=True (has "take", drug names)
    │                 action=warn  → appends to ctx.warnings
    │                 action=block → sets ctx.blocked=True, stops here
    │
    ├─── Stage 2: PRE-LLM (runs if not blocked)
    │         │
    │         └─ DrugSafetyChecker.check(ctx.processed_input)
    │                 DrugMentionExtractor pulls drug names via regex
    │                 → ["warfarin", "aspirin"]
    │                 RxNormClient.get_rxcui("warfarin") → "202421"  (disk cached)
    │                 RxNormClient.get_rxcui("aspirin")  → "1191"    (disk cached)
    │                 OpenFDAClient.get_drug_interactions("warfarin", "aspirin")
    │                 → fetches label text, _parse_severity() → HIGH
    │                 StaticTable.lookup("warfarin", "aspirin") → HIGH (offline fallback)
    │                 → ctx.warnings.append("Drug interaction: warfarin+aspirin HIGH")
    │                 severity >= threshold → ctx.blocked=True (if block mode)
    │
    ├─── Stage 3: LLM CALL (runs if not blocked)
    │         │
    │         └─ LLMCaller.call(ctx.processed_input)
    │                 AnthropicCaller  → anthropic SDK  → claude-haiku / claude-sonnet
    │                 OpenAICaller     → openai SDK     → gpt-4o
    │                 OpenAICaller     → httpx raw      → ollama:11434/v1 (local)
    │                 → ctx.llm_response = "Warfarin and aspirin together..."
    │
    └─── Stage 4: POST-LLM (runs on LLM output)
              │
              ├─ 4a. HallucinationDetector.check(ctx.llm_response)
              │         asyncio.gather() runs 3 sub-checks concurrently:
              │         ├─ _check_drug_names()
              │         │     _DRUG_MENTION_RE finds drug-like tokens
              │         │     RxNormClient.validate_drug_exists("xyzitol") → False
              │         │     → flag: FAKE_DRUG_NAME
              │         ├─ _check_dosages()
              │         │     _DOSAGE_RE finds "ibuprofen 10000mg"
              │         │     MAX_DOSES["ibuprofen"] = 3200
              │         │     10000 > 3200 * 1.5 → flag: IMPOSSIBLE_DOSAGE
              │         └─ _check_confident_claims()
              │               _CONFIDENT_RE finds "definitely", "always"
              │               → flag: CONFIDENT_UNSUPPORTED_CLAIM
              │         score = weighted sum of flags
              │         score >= threshold → ctx.blocked=True
              │         _annotate_text() → inlines [WARNING:...] in response
              │
              └─ 4b. FactVerifier.verify(ctx.llm_response)  [opt-in]
                        _extract_claims() → regex extracts falsifiable claims
                        → ["max dose of ibuprofen is 3000mg", "aspirin inhibits COX-2"]
                        PubMedClient.search("ibuprofen max dose") → [PMIDs]
                        PubMedClient.fetch_abstracts([PMIDs]) → abstracts
                        _score_evidence() → supporting / contradicting
                        → ctx.processed_output += "[FACT-CHECK: See PMIDs: 12345678]"
```

```
PipelineContext returned to caller:
{
  original_input:        "I take warfarin and aspirin, SSN 123-45-6789"
  processed_input:       "I take warfarin and aspirin, SSN [REDACTED]"
  phi_result:            { phi_detected: True, matches: [SSN] }
  drug_result:           { interactions: [warfarin+aspirin HIGH], blocked: False }
  llm_response:          "Warfarin and aspirin together can increase bleeding risk..."
  hallucination_result:  { flags: [], score: 0.0 }
  fact_check_result:     { claims_checked: 2, verified: 2, confidence: 0.85 }
  processed_output:      same as llm_response (clean — no flags triggered)
  warnings:              ["Drug interaction: warfarin + aspirin (high)"]
  blocked:               False
  processing_time_ms:    312.4
}
```

**Key design decisions:**

- **Each stage is isolated in `try/except`** — if OpenFDA times out, drug safety fails silently and the request continues. Nothing crashes the full pipeline.
- **`PipelineContext` is the single mutable state object** — passed through all stages, each stage writes its results onto it. No hidden globals.
- **Streaming path** — stages 1+2 run synchronously before the first token, LLM streams to a buffer, stages 4a+4b run on the complete buffer before the final SSE flush.
- **Caching inside knowledge clients** — RxNorm and OpenFDA responses are cached on disk via `diskcache`. Repeated drug lookups are instant.
- **`MedGuard` is only a wiring layer** — it instantiates all components at `__init__` time and hands them to `GuardrailPipeline`. The pipeline itself has no knowledge of config or components — it just calls whatever it receives.
- **PubMed fact verification is opt-in** — enabled via `MEDGUARD_GUARDRAILS__FACT_CHECKING__ENABLED=true`. Requires network; disabled by default to keep latency low.

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
| [PubMed / NCBI E-utilities](https://www.ncbi.nlm.nih.gov/home/develop/api/) | Fact verification against peer-reviewed literature |
| Bundled SNOMED-CT subset | Medical terminology validation |
| Curated static table | Highest-risk drug pairs (offline fallback) |

## License

Apache 2.0
