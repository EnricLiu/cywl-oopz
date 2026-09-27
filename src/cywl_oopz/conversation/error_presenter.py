"""One safe error protocol shared by conversational entry points."""

from __future__ import annotations

from dataclasses import dataclass

from cywl_oopz.core.errors import (
    AgentInternalError,
    AuthorizationError,
    DatabaseError,
    FeatureDisabledError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitExceeded,
    UserRequestError,
)
from cywl_oopz.core.observability import exception_kind, opaque_ref


@dataclass(frozen=True, slots=True)
class ChatErrorPresentation:
    """Safe response plus structured logging classification."""

    message: str
    code: str
    responsibility: str
    reference: str = ""
    internal: bool = False


class ChatErrorPresenter:
    """Classify chat/Agent failures without exposing exception text."""

    def present(self, error: Exception, *, request_ref: str) -> ChatErrorPresentation:
        if isinstance(error, FeatureDisabledError):
            return self._expected("feature_disabled", "文字对话功能当前未启用。", "user")
        if isinstance(error, UserRequestError):
            return self._expected(error.code, error.user_message, "user")
        if isinstance(error, RateLimitExceeded):
            message = (
                f"请求过于频繁，请在 {error.retry_after_seconds:.1f} 秒后重试。"
                if error.retry_after_seconds > 0
                else "当前对话请求较多，请稍后重试。"
            )
            return self._expected("rate_limited", message, "runtime")
        if isinstance(error, ProviderTimeoutError):
            return self._expected(
                "provider_timeout",
                "模型响应超时，请稍后重试。",
                "provider",
            )
        if isinstance(error, ProviderError):
            return self._expected(
                "provider_unavailable",
                "模型服务暂时不可用，请稍后重试。",
                "provider",
            )
        if isinstance(error, DatabaseError):
            return self._expected(
                "database_unavailable",
                "会话服务暂时不可用，请稍后重试。",
                "database",
            )
        if isinstance(error, AuthorizationError):
            return self._expected(
                "permission_denied",
                "你没有执行此操作的权限。",
                "user",
            )
        code = "agent_internal" if isinstance(error, AgentInternalError) else "internal_error"
        reference = opaque_ref("chat-error", request_ref, code, exception_kind(error))
        return ChatErrorPresentation(
            message=f"这次处理在内部出错，请稍后重试（参考号：{reference}）。",
            code=code,
            responsibility="internal",
            reference=reference,
            internal=True,
        )

    @staticmethod
    def _expected(code: str, message: str, responsibility: str) -> ChatErrorPresentation:
        return ChatErrorPresentation(message, code, responsibility)
