"""FastAPI dependency providers backed by the composition root."""

from fastapi import Request

from property_assistant.chat.conversation import ConversationStore
from property_assistant.chat.service import ChatService
from property_assistant.container import AppContainer
from property_assistant.core.exceptions import IndexNotReadyError


def get_container(request: Request) -> AppContainer:
    """Return the app container created in the lifespan.

    Raises:
        IndexNotReadyError: When startup has not completed.
    """
    container: AppContainer | None = getattr(request.app.state, "container", None)
    if container is None:
        raise IndexNotReadyError("Application is still starting up")
    return container


def get_chat_service(request: Request) -> ChatService:
    """Provide the chat service."""
    return get_container(request).chat_service


def get_conversation_store(request: Request) -> ConversationStore:
    """Provide the conversation store."""
    return get_container(request).conversation_store
