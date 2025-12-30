"""
Model Serving Abstraction

Provides a unified interface for running ML model inference across
different backends (local models, API services, custom implementations).
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Union, AsyncIterator
import uuid


class ModelBackend(Enum):
    """Supported model backends."""
    LOCAL = "local"           # Local model (transformers, etc.)
    OPENAI = "openai"         # OpenAI API
    ANTHROPIC = "anthropic"   # Anthropic API
    OLLAMA = "ollama"         # Ollama local server
    CUSTOM = "custom"         # Custom implementation


class ModelType(Enum):
    """Types of models."""
    TEXT_GENERATION = "text_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class InferenceRequest:
    """Request for model inference."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    images: Optional[List[str]] = None  # Base64 encoded
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Common parameters
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


@dataclass
class InferenceResponse:
    """Response from model inference."""
    request_id: str
    model_id: str
    success: bool
    content: Optional[str] = None
    embeddings: Optional[List[float]] = None
    classification: Optional[Dict[str, float]] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_error(self) -> bool:
        return not self.success


@dataclass
class StreamChunk:
    """A chunk of streaming response."""
    request_id: str
    content: str
    is_final: bool = False
    finish_reason: Optional[str] = None


class ModelProvider(ABC):
    """
    Abstract base class for model providers.

    Implement this to add support for new backends.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    @property
    @abstractmethod
    def backend(self) -> ModelBackend:
        """Return the backend type."""
        pass

    @property
    @abstractmethod
    def supported_types(self) -> List[ModelType]:
        """Return supported model types."""
        pass

    @abstractmethod
    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a request."""
        pass

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        """Stream inference results. Override for streaming support."""
        # Default: yield single chunk
        response = await self.infer(request)
        yield StreamChunk(
            request_id=request.request_id,
            content=response.content or "",
            is_final=True,
        )

    async def batch_infer(
        self,
        requests: List[InferenceRequest]
    ) -> List[InferenceResponse]:
        """Run batch inference. Override for optimized batching."""
        return await asyncio.gather(*[self.infer(r) for r in requests])


class MockModelProvider(ModelProvider):
    """
    Mock model provider for testing.

    Returns configurable responses without calling any external APIs.
    """

    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.CUSTOM

    @property
    def supported_types(self) -> List[ModelType]:
        return [ModelType.TEXT_GENERATION, ModelType.EMBEDDING]

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        # Simulate latency
        latency = self.config.get("latency_ms", 100)
        await asyncio.sleep(latency / 1000)

        # Generate mock response
        if request.messages:
            last_msg = request.messages[-1].get("content", "")
            content = f"Mock response to: {last_msg[:50]}..."
        elif request.prompt:
            content = f"Mock response to: {request.prompt[:50]}..."
        else:
            content = "Mock response"

        return InferenceResponse(
            request_id=request.request_id,
            model_id=request.model_id or "mock-model",
            success=True,
            content=content,
            prompt_tokens=len(content.split()) * 2,
            completion_tokens=len(content.split()),
            total_tokens=len(content.split()) * 3,
            latency_ms=latency,
        )

    async def stream(self, request: InferenceRequest) -> AsyncIterator[StreamChunk]:
        response = await self.infer(request)
        words = (response.content or "").split()

        for i, word in enumerate(words):
            yield StreamChunk(
                request_id=request.request_id,
                content=word + " ",
                is_final=(i == len(words) - 1),
            )
            await asyncio.sleep(0.05)  # Simulate streaming delay


class AnthropicProvider(ModelProvider):
    """
    Anthropic Claude model provider.

    Requires: pip install anthropic
    """

    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.ANTHROPIC

    @property
    def supported_types(self) -> List[ModelType]:
        return [ModelType.TEXT_GENERATION, ModelType.VISION]

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                success=False,
                error="anthropic package not installed",
            )

        client = AsyncAnthropic(api_key=self.config.get("api_key"))
        model = request.model_id or self.config.get("default_model", "claude-3-sonnet-20240229")

        start = datetime.utcnow()

        try:
            # Build messages
            if request.messages:
                messages = request.messages
            elif request.prompt:
                messages = [{"role": "user", "content": request.prompt}]
            else:
                messages = []

            response = await client.messages.create(
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=messages,
            )

            latency = (datetime.utcnow() - start).total_seconds() * 1000

            return InferenceResponse(
                request_id=request.request_id,
                model_id=model,
                success=True,
                content=response.content[0].text if response.content else None,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=latency,
                raw_response={"stop_reason": response.stop_reason},
            )

        except Exception as e:
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return InferenceResponse(
                request_id=request.request_id,
                model_id=model,
                success=False,
                error=str(e),
                latency_ms=latency,
            )


class OpenAIProvider(ModelProvider):
    """
    OpenAI model provider.

    Requires: pip install openai
    """

    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.OPENAI

    @property
    def supported_types(self) -> List[ModelType]:
        return [
            ModelType.TEXT_GENERATION,
            ModelType.EMBEDDING,
            ModelType.VISION,
        ]

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                success=False,
                error="openai package not installed",
            )

        client = AsyncOpenAI(api_key=self.config.get("api_key"))
        model = request.model_id or self.config.get("default_model", "gpt-4")

        start = datetime.utcnow()

        try:
            # Build messages
            if request.messages:
                messages = request.messages
            elif request.prompt:
                messages = [{"role": "user", "content": request.prompt}]
            else:
                messages = []

            response = await client.chat.completions.create(
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                messages=messages,
                stop=request.stop_sequences,
            )

            latency = (datetime.utcnow() - start).total_seconds() * 1000
            choice = response.choices[0]

            return InferenceResponse(
                request_id=request.request_id,
                model_id=model,
                success=True,
                content=choice.message.content,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                latency_ms=latency,
                raw_response={"finish_reason": choice.finish_reason},
            )

        except Exception as e:
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return InferenceResponse(
                request_id=request.request_id,
                model_id=model,
                success=False,
                error=str(e),
                latency_ms=latency,
            )


class OllamaProvider(ModelProvider):
    """
    Ollama local model provider.

    Requires: Ollama running locally (default: http://localhost:11434)
    """

    @property
    def backend(self) -> ModelBackend:
        return ModelBackend.OLLAMA

    @property
    def supported_types(self) -> List[ModelType]:
        return [ModelType.TEXT_GENERATION, ModelType.EMBEDDING]

    async def infer(self, request: InferenceRequest) -> InferenceResponse:
        try:
            import httpx
        except ImportError:
            return InferenceResponse(
                request_id=request.request_id,
                model_id=request.model_id,
                success=False,
                error="httpx package not installed",
            )

        base_url = self.config.get("base_url", "http://localhost:11434")
        model = request.model_id or self.config.get("default_model", "llama2")

        start = datetime.utcnow()

        try:
            # Build prompt
            if request.messages:
                prompt = "\n".join(
                    f"{m['role']}: {m['content']}"
                    for m in request.messages
                )
            else:
                prompt = request.prompt or ""

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": request.temperature,
                            "top_p": request.top_p,
                            "num_predict": request.max_tokens,
                        }
                    },
                    timeout=120.0,
                )
                response.raise_for_status()
                data = response.json()

            latency = (datetime.utcnow() - start).total_seconds() * 1000

            return InferenceResponse(
                request_id=request.request_id,
                model_id=model,
                success=True,
                content=data.get("response", ""),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                latency_ms=latency,
                raw_response=data,
            )

        except Exception as e:
            latency = (datetime.utcnow() - start).total_seconds() * 1000
            return InferenceResponse(
                request_id=request.request_id,
                model_id=model,
                success=False,
                error=str(e),
                latency_ms=latency,
            )


# ============================================================================
# Model Server
# ============================================================================

class ModelServer:
    """
    Central model serving manager.

    Manages multiple model providers and routes inference requests
    to the appropriate backend.
    """

    def __init__(self):
        self._providers: Dict[str, ModelProvider] = {}
        self._default_provider: Optional[str] = None

    def register_provider(
        self,
        name: str,
        provider: ModelProvider,
        default: bool = False
    ):
        """Register a model provider."""
        self._providers[name] = provider
        if default or self._default_provider is None:
            self._default_provider = name

    def get_provider(self, name: Optional[str] = None) -> ModelProvider:
        """Get a provider by name, or the default."""
        name = name or self._default_provider
        if not name or name not in self._providers:
            raise ValueError(f"Provider not found: {name}")
        return self._providers[name]

    async def infer(
        self,
        request: InferenceRequest,
        provider: Optional[str] = None
    ) -> InferenceResponse:
        """Run inference using specified or default provider."""
        p = self.get_provider(provider)
        return await p.infer(request)

    async def stream(
        self,
        request: InferenceRequest,
        provider: Optional[str] = None
    ) -> AsyncIterator[StreamChunk]:
        """Stream inference using specified or default provider."""
        p = self.get_provider(provider)
        async for chunk in p.stream(request):
            yield chunk

    async def batch_infer(
        self,
        requests: List[InferenceRequest],
        provider: Optional[str] = None
    ) -> List[InferenceResponse]:
        """Run batch inference."""
        p = self.get_provider(provider)
        return await p.batch_infer(requests)

    def list_providers(self) -> List[Dict[str, Any]]:
        """List all registered providers."""
        return [
            {
                "name": name,
                "backend": p.backend.value,
                "supported_types": [t.value for t in p.supported_types],
                "is_default": name == self._default_provider,
            }
            for name, p in self._providers.items()
        ]


# ============================================================================
# Factory Functions
# ============================================================================

def create_model_server(
    providers: Optional[Dict[str, Dict[str, Any]]] = None,
    default: Optional[str] = None,
) -> ModelServer:
    """
    Create a model server with configured providers.

    Example:
        server = create_model_server({
            "anthropic": {"api_key": "sk-..."},
            "openai": {"api_key": "sk-..."},
            "mock": {},
        }, default="anthropic")
    """
    server = ModelServer()

    provider_classes = {
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "mock": MockModelProvider,
    }

    if providers:
        for name, config in providers.items():
            if name in provider_classes:
                provider = provider_classes[name](config)
                is_default = (name == default) if default else False
                server.register_provider(name, provider, default=is_default)

    return server


# Global model server instance
global_model_server = ModelServer()
global_model_server.register_provider("mock", MockModelProvider(), default=True)
