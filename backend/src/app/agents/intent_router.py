from langchain_openai import ChatOpenAI
from agents.greeting_agent import GreetingAgent
from agents.telemetry_analysis_agent import TelemetryAnalysisAgent
from agents.fallback_agent import FallbackAgent
from agents.anomaly_agent import AnomalyAgent
from langchain.schema import SystemMessage, HumanMessage
import os
import json
from core.session_store import SessionStore
import logging

logger = logging.getLogger(__name__)

class IntentRouterAgent:
    def __init__(self, session_id: str, store: SessionStore):
        self.session_id = session_id
        self.store = store
        self.INTENT_TO_AGENT_MAP = {
            "greeting": GreetingAgent(session_id, self.store),
            "factual": TelemetryAnalysisAgent(self.store.get_telemetry(session_id), session_id, self.store),
            "anomaly": AnomalyAgent(self.store.get_telemetry(session_id), session_id, self.store),
            "clarification": TelemetryAnalysisAgent(self.store.get_telemetry(session_id), session_id, self.store),
            "other": FallbackAgent(session_id, self.store)
        }
        self.INTENT_SYSTEM = SystemMessage(content=(
            "You are an intent classifier for a UAV flight log assistant.\n\n"
            "Your task is to classify the intent behind the **current user message**, using both the message "
            "itself and the **preceding conversation history** for context.\n\n"
            "Choose one of the following labels:\n\n"
            "• greeting – user is saying hello, goodbye, or expressing thanks.\n"
            "  Example: 'Hi', 'Hello', 'Thank you', 'Talk to you later'\n\n"
            "• factual – user is asking for a specific value, summary, correlation, or telemetry-derived fact.\n"
            "  Includes both simple and complex questions, as long as they’re answerable via direct telemetry analysis.\n"
            "  Examples:\n"
            "    - 'What was the max altitude?'\n"
            "    - 'List all ERR messages with severity ≥ 3.'\n"
            "    - 'Did GPS ever drop below 5 satellites?'\n"
            "    - 'When was the first RC signal loss?'\n"
            "    - 'Any Z-axis vibration spikes over 30 m/s²?'\n\n"
            "• anomaly – user is investigating deeper anomalies or behavioral trends without pointing to a specific metric.\n"
            "  Examples:\n"
            "    - 'Were there any anomalies in this flight?'\n"
            "    - 'Spot any strange behavior?'\n"
            "    - 'Can you tell if anything went wrong in the last phase?'\n\n"
            "• clarification – the user is referring to or refining a previous question or answer, \n"
            "  or is asking a vague or underspecified question that requires more detail before proceeding.  \n"
            "  This includes:\n"
            "    - narrowing the scope ('Can you zoom in on…'),\n"
            "    - asking for context ('Why did this happen?'),\n"
            "    - requesting disambiguation ('Which motor?', 'What component?'),\n"
            "    - or referencing a prior system message ('Actually, I meant…')\n"
            "\n"
            "  Typical signals: \n"
            "    'I meant...', 'Can you also...', 'Zoom in...', 'More detail about...', 'Why did…', \n"
            "    'What caused...', 'Tell me more about...', 'What happened…', 'Which...', 'What part...'\n"
            "\n"
            "  Examples:\n"
            "    - 'No, I meant after the landing.'\n"
            "    - 'Can you zoom in on that altitude drop?'\n"
            "    - 'What happened just before that mode change?'\n"
            "    - 'Actually, I was asking about the second flight.'\n"
            "    - 'And what about during LOITER?'\n"
            "    - 'Why did the current spike?'\n"
            "    - 'What caused the altitude to change?'\n"
            "    - 'Tell me more about that anomaly.'\n"
            "    - 'Which motor?'\n"
            "    - 'What component was drawing the most power?'\n\n"
            "• other – none of the above apply (e.g., off-topic, casual chat).\n\n"
            "Return a JSON with:\n"
            "    - intent: one of [\"greeting\", \"factual\", \"exploratory\", \"clarification\", \"other\"]\n"
            "    - follow_up_on: (if intent is \"clarification\", mention the topic it's following up on, else '')\n"
            "    - topic: (if intent is \"factual\" or \"exploratory\", summarize the main subject e.g., \"descent phase\", \"voltage drop\", etc.)\n\n"
            "Respond with only the JSON.\n"
            "Do not include any explanation or commentary."
        )
    )

    def classify_intent_llm(self, msg: str) -> str:
        user = HumanMessage(content=msg)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        history = self.store.get_history(self.session_id)[-2:]  # keeping it concise, no need to include long history
        msgs = [self.INTENT_SYSTEM] + history + [user]
        result = llm.invoke(msgs)
        response = result.content.strip().lower()
        logger.info(f"Intent classification result: {response}")
        try:
            parsed = json.loads(response)
            return parsed
        except Exception as e:
            logger.error(f"Failed to parse intent JSON: {response} → {e}")
            return {"intent": "other", "follow_up_on": ""}

    def classify_intent(self, user_msg: str) -> str:
        intent = self.classify_intent_llm(user_msg)
        self.store.set_intent(self.session_id, intent)
        return intent

    def route(self, user_msg: str) -> str:
        parsed = self.classify_intent_llm(user_msg)
        intent = parsed.get("intent", "other")
        follow_up_on = parsed.get("follow_up_on", "").strip()

        self.store.set_intent(self.session_id, intent)
        if intent == "factual" and follow_up_on:
            self.store.set_topic(self.session_id, follow_up_on)

        logger.info(f"classified intent: {intent}, topic: {follow_up_on}")
        agent = self.INTENT_TO_AGENT_MAP.get(intent)
        return agent.chat(user_msg) if agent else FallbackAgent(self.session_id, self.store).chat(user_msg)