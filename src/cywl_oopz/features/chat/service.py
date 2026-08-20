"""Use cases for serialised, expiring LLM conversations."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

from cywl_oopz.core.errors import (
    AuthorizationError,
    FeatureDisabledError,
    ProviderError,
    UserRequestError,
)
from cywl_oopz.core.health import HealthRegistry, HealthState
from cywl_oopz.core.observability import exception_kind, opaque_ref
from cywl_oopz.features.agent.input import AgentUserInput
from cywl_oopz.settings import ChatSettings

from .history import HistoryTrimmer
from .locks import ConversationLockPool
from .models import (
    ChatInvocation,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatRole,
    ChatStatus,
    ConversationKey,
    ConversationSession,
)
from .progress import ConversationProgressEvent, ProgressKind, ProgressSink, emit_progress
from .provider import ChatProvider
from .rate_limit import RateLimitService
from .repository import ConversationRepository
from .streaming import StreamResponseAssembler

logger = logging.getLogger(__name__)


class ChatService:
    """Coordinates provider requests, history persistence, and request ownership."""

    def __init__(
        self,
        settings: ChatSettings,
        provider: ChatProvider,
        repository: ConversationRepository,
        rate_limits: RateLimitService | None = None,
        locks: ConversationLockPool | None = None,
        health: HealthRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._repository = repository
        self._rate_limits = rate_limits or RateLimitService(settings)
        self._locks = locks or ConversationLockPool()
        self._history = HistoryTrimmer(
            max_messages=settings.max_history_messages,
            max_characters=settings.max_history_characters,
        )
        self._stream_assembler = StreamResponseAssembler()
        self._health = health

    @property
    def enabled(self) -> bool:
        """Expose the feature flag without exposing provider credentials."""
        return self._settings.enabled

    async def ask(
        self,
        key: ConversationKey,
        prompt: str,
        *,
        user_input: AgentUserInput | None = None,
        invocation: ChatInvocation | None = None,
        progress: ProgressSink | None = None,
    ) -> ChatResponse:
        """Answer one prompt while keeping the same session strictly ordered."""
        del invocation
        if user_input is not None and user_input.has_images:
            raise UserRequestError(
                "image_model_unavailable",
                "当前文字对话模型不支持图片，请切换到支持图片的模型后再试。",
            )
        self._ensure_enabled()
        content = prompt.strip()
        if not content:
            raise ValueError("Chat prompt must not be empty")
        self._history.validate_input(content)
        started_at = time.perf_counter()
        conversation = self._conversation_ref(key)
        logger.info(
            "Chat request received: conversation=%s prompt_characters=%s streaming=%s",
            conversation,
            len(content),
            self._settings.stream_responses,
        )
        await emit_progress(progress, ConversationProgressEvent(ProgressKind.ACCEPTED))

        async with self._locks.hold(key):
            async with await self._rate_limits.acquire(key):
                now = datetime.now(UTC)
                existing = await self._load_active(key, now)
                history = existing.messages if existing is not None else ()
                selected_model = existing.selected_model if existing is not None else None
                model = selected_model or self._settings.model
                request_history = self._history.trim(
                    history + (ChatMessage(ChatRole.USER, content),)
                )
                request = ChatRequest(
                    model=model,
                    messages=(ChatMessage(ChatRole.SYSTEM, self._settings.system_prompt),)
                    + request_history,
                    user_id=key.person_id,
                    timeout_seconds=self._settings.request_timeout_seconds,
                )
                logger.debug(
                    "Chat provider request started: conversation=%s model=%s history_messages=%s",
                    conversation,
                    model,
                    len(request_history),
                )
                response = await self._request_response(request)
                persisted_history = self._history.trim(
                    request_history + (ChatMessage(ChatRole.ASSISTANT, response.content),)
                )
                await self._repository.save(
                    ConversationSession(
                        key=key,
                        messages=persisted_history,
                        selected_model=selected_model,
                        expires_at=now + timedelta(seconds=self._settings.session_ttl_seconds),
                    )
                )
                logger.info(
                    "Chat request completed: conversation=%s model=%s elapsed_seconds=%.3f "
                    "input_tokens=%s output_tokens=%s",
                    conversation,
                    response.model,
                    time.perf_counter() - started_at,
                    response.input_tokens,
                    response.output_tokens,
                )
                return response

    async def clear(self, key: ConversationKey) -> None:
        """Forget all persisted history for a caller's current conversation scope."""
        self._ensure_enabled()
        async with self._locks.hold(key):
            await self._repository.delete(key)
        logger.info("Chat conversation cleared: conversation=%s", self._conversation_ref(key))

    async def select_model(self, key: ConversationKey, model: str) -> str:
        """Persist an allow-listed model choice for an explicitly allowed person."""
        self._ensure_enabled()
        if key.person_id not in self._settings.model_selection_users:
            raise AuthorizationError("The caller is not allowed to select a model")
        choice = model.strip()
        if choice not in self._settings.allowed_models:
            raise ValueError("The requested model is not allowed")

        async with self._locks.hold(key):
            now = datetime.now(UTC)
            existing = await self._load_active(key, now)
            await self._repository.save(
                ConversationSession(
                    key=key,
                    messages=existing.messages if existing is not None else (),
                    selected_model=choice,
                    expires_at=now + timedelta(seconds=self._settings.session_ttl_seconds),
                )
            )
        logger.info(
            "Chat model selected: conversation=%s model=%s", self._conversation_ref(key), choice
        )
        return choice

    async def status(self, key: ConversationKey) -> ChatStatus:
        """Return safe metadata only; message contents never leave the repository here."""
        if not self._settings.enabled:
            return ChatStatus(False, False, "", 0, None, 0.0)
        now = datetime.now(UTC)
        existing = await self._load_active(key, now)
        return ChatStatus(
            enabled=True,
            active=existing is not None,
            model=(existing.selected_model if existing is not None else None)
            or self._settings.model,
            history_message_count=len(existing.messages) if existing is not None else 0,
            expires_at=existing.expires_at if existing is not None else None,
            cooldown_seconds=await self._rate_limits.cooldown_remaining(key),
        )

    async def _load_active(
        self,
        key: ConversationKey,
        now: datetime,
    ) -> ConversationSession | None:
        session = await self._repository.get(key)
        if session is not None and session.is_expired(now):
            await self._repository.delete(key)
            return None
        return session

    async def _request_response(self, request: ChatRequest) -> ChatResponse:
        try:
            if self._settings.stream_responses:
                response = await self._stream_assembler.assemble(
                    self._provider.stream(request),
                    request.model,
                )
            else:
                response = await self._provider.complete(request)
        except ProviderError as exc:
            self._mark_provider_health(HealthState.DEGRADED, "request failed")
            logger.warning(
                "Chat provider request failed: model=%s error=%s",
                request.model,
                exception_kind(exc),
            )
            raise
        else:
            self._mark_provider_health(HealthState.HEALTHY, "last request succeeded")
            return response

    def _ensure_enabled(self) -> None:
        if not self._settings.enabled:
            raise FeatureDisabledError("Text chat is disabled")

    def _mark_provider_health(self, state: HealthState, detail: str) -> None:
        if self._health is not None:
            self._health.mark("llm", state, detail)

    @staticmethod
    def _conversation_ref(key: ConversationKey) -> str:
        return opaque_ref(key.scope, key.area_id, key.channel_id, key.person_id)
