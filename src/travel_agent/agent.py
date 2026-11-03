"""Trip Planner, all three workflow patterns composed into one pipeline with
ADK 2.4's graph-based `Workflow` API, instead of the (now deprecated)
SequentialAgent / ParallelAgent / LoopAgent shell-agent classes.

Where the classic shell agents nest one class per pattern, a Workflow expresses
everything as **nodes connected by edges**:

    - sequential  -> a chain of edges           (a -> b -> c)
    - parallel    -> a fan-out + a JoinNode      (a -> (b, c) -> join)
    - loop        -> a conditional route + a back-edge

Graph shape:

    START -> CaptureRequest -> IntakeClassifier -> IntakeGate
        --plan--> Preferences -> (FlightSearch ‖ HotelSearch ‖ ActivityAgent)
                  -> SearchJoin -> Composer -> Critic -> Gate --approved--> Finalize
                                                        \\--refine--> Refiner --(back to Critic)
        --chat--> AssistantReply

The single-pattern siblings (`workflow_sequential`, `workflow_parallel`,
`workflow_loop`) each demonstrate one of these patterns in isolation; this agent
combines all three. New agent *instances* are created here because an ADK agent
can only belong to one parent tree.
"""

from typing import AsyncGenerator

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events import Event, EventActions
from google.adk.models.lite_llm import LiteLlm
from google.adk.workflow import Workflow, JoinNode, START, node
from google.adk.workflow._base_node import BaseNode

from .prompts import (
    STATE_PREFERENCES,
    STATE_FLIGHTS,
    STATE_HOTELS,
    STATE_ACTIVITIES,
    STATE_ITINERARY,
    STATE_BUDGET_REVIEW,
    COMPLETION_PHRASE,
    PREFERENCES_AGENT_SYSTEM,
    FLIGHT_SEARCH_AGENT_SYSTEM,
    HOTEL_SEARCH_AGENT_SYSTEM,
    ACTIVITY_AGENT_SYSTEM,
    ITINERARY_COMPOSER_SYSTEM,
    BUDGET_CRITIC_SYSTEM,
    ITINERARY_REFINER_SYSTEM,
)

MODEL = LiteLlm("gemini/gemini-3.1-flash-lite")
MAX_REFINEMENTS = 3
STATE_REFINE_COUNT = "refine_count"
STATE_INTAKE = "intake"
STATE_USER_REQUEST = "user_request"

USER_REQUEST_BLOCK = 'The traveler wrote:\n"""\n{user_request}\n"""\n\n'

INTAKE_CLASSIFIER_SYSTEM = USER_REQUEST_BLOCK + """
You are the intake router for a travel-planning assistant.

Decide whether the traveler's message above is a concrete trip-planning request.
A concrete request names a destination AND gives at least one of: travel dates, a budget,
the number of travelers, or clearly asks you to plan/book a trip.

Respond with ONLY one word, in uppercase, nothing else:
- PLAN  — if it is a concrete trip request
- CHAT  — for anything else (greetings, "what can you do?", or missing the details above)
""".strip()

ASSISTANT_REPLY_SYSTEM = """
You are a friendly travel-planning assistant.

Briefly explain what you can help with: planning a trip end to end — finding flights and hotels,
suggesting activities that match their interests, and keeping the whole thing within budget.

Then ask the user to share, so you can start:
- Origin city
- Destination city
- Travel dates (or trip length)
- Number of travelers
- Total budget (with currency)
- Interests (e.g. food, history, nightlife, nature)

Keep it to a few short sentences. Do not invent a trip or start planning yet.
""".strip()

preferences_agent = LlmAgent(
    name="PreferencesAgent", model=MODEL, include_contents="none",
    instruction=USER_REQUEST_BLOCK + PREFERENCES_AGENT_SYSTEM,
    description="Extracts budget, dates, and interests from the traveler's request.",
    output_key=STATE_PREFERENCES,
)
flight_search_agent = LlmAgent(
    name="FlightSearchAgent", model=MODEL, include_contents="none",
    instruction=FLIGHT_SEARCH_AGENT_SYSTEM, description="Returns mock flight options.",
    output_key=STATE_FLIGHTS,
)
hotel_search_agent = LlmAgent(
    name="HotelSearchAgent", model=MODEL, include_contents="none",
    instruction=HOTEL_SEARCH_AGENT_SYSTEM, description="Returns mock hotel options.",
    output_key=STATE_HOTELS,
)
activity_agent = LlmAgent(
    name="ActivityAgent", model=MODEL, include_contents="none",
    instruction=ACTIVITY_AGENT_SYSTEM, description="Suggests matching activities.",
    output_key=STATE_ACTIVITIES,
)
itinerary_composer_agent = LlmAgent(
    name="ItineraryComposerAgent", model=MODEL, include_contents="none",
    instruction=ITINERARY_COMPOSER_SYSTEM, description="Assembles a costed itinerary.",
    output_key=STATE_ITINERARY,
)
budget_critic_agent = LlmAgent(
    name="BudgetCriticAgent", model=MODEL, include_contents="none",
    instruction=BUDGET_CRITIC_SYSTEM, description="Approves the itinerary or suggests cuts.",
    output_key=STATE_BUDGET_REVIEW,
)
itinerary_refiner_agent = LlmAgent(
    name="ItineraryRefinerAgent", model=MODEL, include_contents="none",
    instruction=ITINERARY_REFINER_SYSTEM, description="Applies budget cuts to the itinerary.",
    output_key=STATE_ITINERARY,
)


@node
def capture_request(ctx: Context, node_input) -> str:
    if isinstance(node_input, str):
        text = node_input
    else:
        parts = getattr(node_input, "parts", None) or []
        text = " ".join(p.text for p in parts if getattr(p, "text", None))
    ctx.state[STATE_USER_REQUEST] = text
    return text


intake_classifier = LlmAgent(
    name="IntakeClassifier", model=MODEL, include_contents="none",
    instruction=INTAKE_CLASSIFIER_SYSTEM,
    description="Decides whether the user's message is a concrete trip request.",
    output_key=STATE_INTAKE,
)
assistant_reply = LlmAgent(
    name="AssistantReply", model=MODEL, include_contents="none",
    instruction=ASSISTANT_REPLY_SYSTEM,
    description="Explains what the assistant can do and asks for trip details.",
)


class IntakeGate(BaseNode):
    """Route planning requests to the itinerary workflow."""

    async def _run_impl(self, *, ctx: Context, node_input) -> AsyncGenerator[Event, None]:
        decision = (ctx.state.get(STATE_INTAKE, "") or "").strip().upper()
        route = "plan" if decision == "PLAN" else "chat"
        yield Event(author=self.name, actions=EventActions(route=route))


intake_gate = IntakeGate(name="IntakeGate")


search_join = JoinNode(name="SearchJoin")


class BudgetGate(BaseNode):
    """Approve the itinerary or request a bounded refinement pass."""

    max_refinements: int = MAX_REFINEMENTS

    async def _run_impl(self, *, ctx: Context, node_input) -> AsyncGenerator[Event, None]:
        review = ctx.state.get(STATE_BUDGET_REVIEW, "") or ""
        count = ctx.state.get(STATE_REFINE_COUNT, 0)
        approved = COMPLETION_PHRASE in review or count >= self.max_refinements
        route = "approved" if approved else "refine"
        yield Event(
            author=self.name,
            actions=EventActions(
                route=route,
                state_delta={STATE_REFINE_COUNT: count + (0 if approved else 1)},
            ),
        )


budget_gate = BudgetGate(name="BudgetGate")


@node
def finalize(ctx: Context) -> str:
    return ctx.state.get(STATE_ITINERARY, "No itinerary was produced.")


root_agent = Workflow(
    name="TripPlannerWorkflow",
    edges=[
        (START, capture_request),
        (capture_request, intake_classifier),
        (intake_classifier, intake_gate),
        (intake_gate, {"plan": preferences_agent, "chat": assistant_reply}),
        (preferences_agent, (flight_search_agent, hotel_search_agent, activity_agent)),
        ((flight_search_agent, hotel_search_agent, activity_agent), search_join),
        (search_join, itinerary_composer_agent),
        (itinerary_composer_agent, budget_critic_agent),
        (budget_critic_agent, budget_gate),
        (budget_gate, {"approved": finalize, "refine": itinerary_refiner_agent}),
        (itinerary_refiner_agent, budget_critic_agent),
    ],
)
