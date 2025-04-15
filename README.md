# AkashChat Plugin

This plugin provides integration with Akash Chat's models through the ElizaOS v2 platform.

## Usage

Add the plugin to your character configuration:

```json
"plugins": ["@elizaos/plugin-akashchat"]
```

## Configuration

The plugin requires these environment variables (can be set in .env file or character settings):

```json
"settings": {
  "AKASH_CHAT_API_KEY": "your_akashchat_api_key",
  "AKASH_CHAT_BASE_URL": "optional_custom_endpoint",
  "AKASH_CHAT_SMALL_MODEL": "Meta-Llama-3-1-8B-Instruct-FP8",
  "AKASH_CHAT_LARGE_MODEL": "Meta-Llama-3-3-70B-Instruct",
  "AKASH_CHAT_EMBEDDING_MODEL": "BAAI-bge-large-en-v1-5",
  "AKASH_CHAT_EMBEDDING_DIMENSIONS": "1024"
}
```

Or in `.env` file:

```env
AKASH_CHAT_API_KEY=your_akashchat_api_key
# Optional overrides:
AKASH_CHAT_BASE_URL=optional_custom_endpoint
AKASH_CHAT_SMALL_MODEL=Meta-Llama-3-1-8B-Instruct-FP8
AKASH_CHAT_LARGE_MODEL=Meta-Llama-3-3-70B-Instruct
AKASH_CHAT_EMBEDDING_MODEL=BAAI-bge-large-en-v1-5
AKASH_CHAT_EMBEDDING_DIMENSIONS=1024
```

### Configuration Options

- `AKASH_CHAT_API_KEY` (required): Your Akash Chat API credentials
- `AKASH_CHAT_BASE_URL`: Custom API endpoint (default: https://chatapi.akash.network/api/v1)
- `AKASH_CHAT_SMALL_MODEL`: Defaults to Llama 3.1 ("Meta-Llama-3-1-8B-Instruct-FP8")
- `AKASH_CHAT_LARGE_MODEL`: Defaults to Llama 3.3 ("Meta-Llama-3-3-70B-Instruct")
- `AKASH_CHAT_EMBEDDING_MODEL`: Defaults to BAAI-bge-large-en-v1-5 ("BAAI-bge-large-en-v1-5")
- `AKASH_CHAT_EMBEDDING_DIMENSIONS`: Defaults to 1024 (1024)

The plugin provides these model classes:

- `TEXT_SMALL`: Optimized for fast, cost-effective responses
- `TEXT_LARGE`: For complex tasks requiring deeper reasoning
- `IMAGE`: AkashGen image generation
- `TEXT_TOKENIZER_ENCODE`: Text tokenization
- `TEXT_TOKENIZER_DECODE`: Token decoding

## Additional Features

### Image Generation

```js
await runtime.useModel(ModelType.IMAGE, {
  prompt: 'A sunset over mountains',
  negative: "",
  sampler: "dpmpp_2m",
  scheduler: "sgm_uniform",
  preferred_gpu: [ "RTX4090", "A10", "A100", "V100-32Gi", "H100"]
});
```

### Text Embeddings

```js
const embedding = await runtime.useModel(ModelType.TEXT_EMBEDDING, 'text to embed');
```
