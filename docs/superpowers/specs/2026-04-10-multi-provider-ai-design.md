# Multi-Provider AI Configuration — Design Spec

## Overview

Replace the hardcoded Gemini-only AI layer with a user-configurable multi-provider system. Users can either use the platform's default Gemini models or bring their own API key (BYOK) from Google Gemini, OpenAI, Anthropic, or any OpenAI-compatible provider.

## Goals

- Let users choose their preferred LLM provider and model
- Support BYOK with encrypted key storage
- Preserve current platform default behavior for users who don't configure anything
- Keep structured output reliable across providers with native JSON mode or prompt-based fallback

## Non-Goals

- Per-purpose model selection (e.g. different models for structuring vs tailoring)
- Cross-provider fallback for BYOK users
- Encryption key rotation mechanism (v1 uses a single key)
- LiteLLM or any third-party abstraction library

---

## Data Model

### New table: `user_ai_config`

| Column | Type | Notes |
|--------|------|-------|
| `id` | int PK | Auto-increment |
| `user_id` | str FK → users, unique | One config per user |
| `provider` | enum | `PLATFORM_GEMINI`, `GEMINI`, `OPENAI`, `ANTHROPIC`, `CUSTOM_OPENAI_COMPATIBLE` |
| `api_key_encrypted` | text, nullable | Fernet-encrypted. Null for `PLATFORM_GEMINI` |
| `api_host` | text, nullable | Only for `CUSTOM_OPENAI_COMPATIBLE` |
| `model_id` | str | e.g. `gemini-3-flash-preview`, `gpt-4o`, `claude-sonnet-4-6` |
| `created_at` | datetime | |
| `updated_at` | datetime | |

### Rules

- `PLATFORM_GEMINI`: uses platform's `GEMINI_API_KEY` env var + user-selected model. No user API key stored.
- All other providers require `api_key_encrypted`.
- `api_host` only required/used for `CUSTOM_OPENAI_COMPATIBLE`.
- When no row exists for a user, system uses platform defaults (flash for structuring/OCR/roast, pro for tailoring) — zero impact on existing users.

### Encryption

- New env var: `ENCRYPTION_KEY` (Fernet key, generated via `cryptography.fernet.Fernet.generate_key()`)
- New utility module: `app/services/crypto.py` with `encrypt(plaintext) -> str` and `decrypt(ciphertext) -> str`
- API key encrypted before DB write, decrypted on read when constructing provider

---

## Provider Abstraction Layer

### Module structure

```
app/services/ai/
  providers/
    __init__.py          # exports get_provider() factory
    base.py              # abstract LLMProvider + ModelInfo + provider exceptions
    gemini.py            # GeminiProvider (google-genai SDK)
    openai.py            # OpenAIProvider (openai SDK)
    anthropic.py         # AnthropicProvider (anthropic SDK)
    openai_compatible.py # OpenAICompatibleProvider (openai SDK + custom base_url)
  inference.py           # refactored to delegate to providers
```

### Abstract base: `LLMProvider`

```python
class LLMProvider(ABC):
    async def list_models(self) -> list[ModelInfo]
    async def generate(
        self,
        system_prompt: str,
        inputs: list[str | dict],
        structured_output_schema: type[BaseModel] | None,
        temperature: float,
        timeout: int | None,
    ) -> str  # raw text response
```

- `ModelInfo`: dataclass with `id: str`, `name: str`, `supports_structured_output: bool`
- `generate()` returns raw text; structured output enforcement is provider-internal:
  - **Native** (Gemini via `response_mime_type`, OpenAI via `response_format`): provider passes schema to API
  - **Prompt-based** (Anthropic, some custom): provider injects JSON schema into system prompt as instructions
- Common parsing/validation layer stays in `inference.py`

### Provider implementations

**GeminiProvider:**
- Uses `google.genai.Client` (existing SDK)
- `list_models()`: calls genai list models API, filters to generative models
- `generate()`: preserves current behavior — thinking_config, vision inputs, native structured output
- Constructed with either platform key or user's BYOK key

**OpenAIProvider:**
- Uses `openai.AsyncOpenAI` client
- `list_models()`: calls `client.models.list()`, filters to chat models
- `generate()`: uses `client.chat.completions.create()` with `response_format: { type: "json_schema" }` for structured output

**AnthropicProvider:**
- Uses `anthropic.AsyncAnthropic` client
- `list_models()`: returns hardcoded curated list (claude-sonnet-4-6, claude-haiku-4-5, etc.) — Anthropic has no list models endpoint
- `generate()`: uses `client.messages.create()`, structured output via prompt engineering (schema injected into system prompt)

**OpenAICompatibleProvider:**
- Extends OpenAI SDK with custom `base_url`
- `list_models()`: attempts `GET {api_host}/v1/models`. Returns empty list on failure (frontend falls back to manual text input).
- `generate()`: same as OpenAIProvider but structured output support not guaranteed — falls back to prompt-based if native mode fails

### Factory function

```python
def get_provider(config: UserAIConfig | None, purpose: str) -> LLMProvider:
```

- `config is None` → `GeminiProvider` with platform key + per-purpose env default model (flash for structuring/OCR/roast, pro for tailoring). This is the legacy behavior.
- `config.provider == PLATFORM_GEMINI` → `GeminiProvider` with platform key + user's selected `model_id` for all purposes.
- Other providers → appropriate class with decrypted user key + user's selected `model_id` for all purposes.

### Refactored `InferenceService` (renamed from `GeminiInference`)

- `run_inference()` orchestration unchanged: retry, schema validation, fallback
- `parse_output()` JSON parsing + Pydantic validation unchanged
- `_log_request()` token usage tracking unchanged (already captures model_name)
- Instead of calling `google.genai` directly, calls `self.provider.generate()`
- Fallback for BYOK: retries with same provider/key only. No cross-provider fallback.

---

## API Endpoints

### New endpoints under `/settings/ai`

| Method | Path | Description |
|--------|------|-------------|
| GET | `/settings/ai` | Get current AI config (provider, model, host, `key_configured: bool`, last 4 chars of key). Never returns full API key. |
| PUT | `/settings/ai` | Save/update config. Validates key via `list_models()` call before persisting. |
| DELETE | `/settings/ai` | Remove custom config, revert to platform default. |
| POST | `/settings/ai/models` | Fetch available models. Body: `{ provider, api_key?, api_host? }`. Returns model list. |

### Design decisions

- `POST /settings/ai/models` uses POST (not GET) because the API key is in the request body, avoiding log leakage in query params.
- This endpoint supports pre-save model fetching — user can browse models before committing their config.
- Rate limited: 10 calls/minute per user on `/settings/ai/models` to prevent abuse.

### Validation on save (`PUT`)

| Provider | Required fields |
|----------|----------------|
| `PLATFORM_GEMINI` | `model_id` |
| `GEMINI`, `OPENAI`, `ANTHROPIC` | `api_key` + `model_id` |
| `CUSTOM_OPENAI_COMPATIBLE` | `api_key` + `api_host` + `model_id` |

On save, a lightweight `list_models()` call verifies the key works. If it fails, return error — don't persist broken config.

---

## Frontend — AI Settings Page

### New page: `/#/settings/ai`

Accessible from a "Settings" link in the nav/dashboard.

### Layout

```
Provider:       [dropdown: Platform Default / Google Gemini / OpenAI / Anthropic / Custom]

API Key:        [password field]          ← hidden for Platform Default
API Host:       [text field]              ← only for Custom
                [Fetch Models button]
Model:          [dropdown]                ← populated after fetch; text input if fetch fails

[Save]  [Reset to Default]

Info: "Your API key is encrypted at rest."
```

### Behavior

- **Platform Default** selected: hides API Key/Host fields, Fetch Models fetches from platform key
- **BYOK provider** selected: shows API Key (+ Host for custom)
- **Fetch Models**: calls `POST /settings/ai/models`, populates dropdown
- **Fetch failure (custom)**: model dropdown switches to text input for manual entry
- **Save**: calls `PUT /settings/ai` with validation
- **Reset to Default**: calls `DELETE /settings/ai`
- **Page load**: `GET /settings/ai` populates current config. Shows `key_configured: true` + masked key (e.g. `••••sk-abcd`)

### Existing flow impact

- No changes to job creation, profile upload, or roast flows
- Optional dashboard indicator: "Using: GPT-4o via OpenAI" or "Using: Platform Default"

---

## Service Integration

### How provider flows through existing services

1. Request handler queries `UserAIConfig` for current user (or None)
2. Config passed to service methods
3. Service methods pass config to `InferenceService` (formerly `GeminiInference`)
4. `InferenceService` uses factory to get provider, then orchestrates as before

### Per-service changes

- **ProfileService** (`process_profile`, `enhance_profile`): currently uses flash model. BYOK uses user's configured model.
- **JobService** (`generate_custom_resume`): currently uses pro model. BYOK uses user's configured model.
- **RoastService** (`process_roast`): currently uses flash model. BYOK uses user's configured model.
- **OCR/Extractor** (`extract_and_structure_via_vision`): vision input (base64 images) is Gemini-specific. Non-Gemini BYOK providers use **text-only path** (pdfplumber → text → LLM structuring). Vision path remains Gemini-only.

### Platform default preserved

When no `user_ai_config` row exists, behavior is identical to today. Flash for structuring/OCR/roast, pro for tailoring. Existing users unaffected.

### Credit system

No changes. Credits deducted per operation regardless of provider. BYOK users still consume credits — they pay for the platform service, not just LLM cost.

### Token logging

No changes needed. `_log_request()` already captures `model_name` — it will log whatever model the provider used.

---

## Error Handling

### Provider exception hierarchy

```
ProviderError (base)
  ├── ProviderAuthError        → HTTP 401, "Check your AI settings"
  ├── ProviderRateLimitError   → HTTP 429
  ├── ProviderModelError       → HTTP 400, "Model not found or unavailable"
  └── ProviderError            → HTTP 502, catch-all
```

Each provider maps its SDK-specific errors to these common exceptions.

### BYOK failure behavior

- BYOK errors surface directly to the user — no platform fallback
- Job status flow unchanged: provider error → `FAILED` status + credit refund
- `PLATFORM_GEMINI` users: fallback works as today (primary model → fallback model)

---

## Security

- **API key never returned**: `GET /settings/ai` returns `key_configured: bool` + last 4 chars only
- **Key validation on save**: lightweight `list_models()` call verifies key before persisting
- **Rate limiting**: 10 calls/min per user on `/settings/ai/models`
- **Fernet encryption**: `ENCRYPTION_KEY` env var, application-level encrypt/decrypt
- **No key rotation in v1**: document that rotating requires re-encryption migration

---

## New Dependencies

Added to `pyproject.toml`:

- `openai` — OpenAI + custom compatible providers
- `anthropic` — Anthropic provider
- `cryptography` — Fernet encryption (make explicit even if transitive)

---

## Migration

- Alembic migration to create `user_ai_config` table
- New `ENCRYPTION_KEY` env var added to `.env.example`
- Existing behavior completely preserved — no data migration needed
