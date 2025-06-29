from __future__ import annotations
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from models.telemetry_data import TelemetryData

@dataclass
class UpdatedSessionMetadata:
    last_msg_types: Optional[List[str]] = None
    topics: Dict[str, str] = field(default_factory=dict)

@dataclass
class Message:
    role: str  # 'user' or 'assistant'
    content: str


class SessionStore:
    """Tiny in-memory session cache with intent tracking."""
    def __init__(self):
        self.telemetry: Dict[str, TelemetryData] = {}
        self.intents: Dict[str, str] = {}
        self.conversation_history: Dict[str, List[Message]] = {}
        self.metadata: Dict[str, UpdatedSessionMetadata] = {}
        self.cached_contexts: Dict[str, Dict[str, str]] = {}

    def get_telemetry(self, session_id: str) -> TelemetryData:
        return self.telemetry.get(session_id)

    def add_session(self, session_id: str, data: TelemetryData):
        self.telemetry[session_id] = data
        self.intents[session_id] = "unknown"

    def set_intent(self, session_id: str, intent: str):
        self.intents[session_id] = intent

    def get_intent(self, session_id: str) -> str:
        return self.intents.get(session_id, "unknown")

    def set_topic(self, session_id: str, topic: str):
        if session_id not in self.metadata:
            self.metadata[session_id] = UpdatedSessionMetadata()
        self.metadata[session_id].topics[session_id] = topic

    def get_topic(self, session_id: str) -> str:
        return self.metadata.get(session_id, UpdatedSessionMetadata()).topics.get(session_id, "")

    def get_cached_context(self, session_id: str, msg_type: str) -> Optional[str]:
        return self.cached_contexts.get(session_id, {}).get(msg_type)

    def cache_context(self, session_id: str, msg_type: str, summary: str):
        if session_id not in self.cached_contexts:
            self.cached_contexts[session_id] = {}
        self.cached_contexts[session_id][msg_type] = summary
    
    def set_last_msg_types(self, session_id: str, msg_types: List[str]):
        if session_id not in self.metadata:
            self.metadata[session_id] = UpdatedSessionMetadata()
        self.metadata[session_id].last_msg_types = msg_types

    def get_last_msg_types(self, session_id: str) -> Optional[List[str]]:
        return self.metadata.get(session_id, UpdatedSessionMetadata()).last_msg_types
        
    def has_session(self, session_id: str) -> bool:
        return session_id in self.telemetry
        
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the conversation history for a session."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        self.conversation_history[session_id].append(Message(role=role, content=content))
    
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get the conversation history for a session as a list of message dicts."""
        if session_id not in self.conversation_history:
            return []
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.conversation_history[session_id]
        ]

store = SessionStore()