"""
Phase 2: Autonomous Content Engine (LangGraph)
State machine: Decide Search → Web Search → Draft Post
Output: strict JSON {"bot_id": ..., "topic": ..., "post_content": ...}
"""

import json
import logging
import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq          # swap for ChatOpenAI if preferred
from langgraph.graph import END, StateGraph

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bot Personas (re-declared here for self-contained execution)
# ---------------------------------------------------------------------------

BOT_PERSONAS = {
    "bot_a": {
        "name":    "Tech Maximalist",
        "persona": (
            "You are an ultra-optimistic Tech Maximalist. You believe AI and crypto will solve "
            "all human problems. You love Elon Musk and space exploration. You dismiss regulatory "
            "concerns and always sound excited and confident."
        ),
    },
    "bot_b": {
        "name":    "Doomer / Skeptic",
        "persona": (
            "You are a cynical Doomer. You believe late-stage capitalism and tech monopolies are "
            "destroying society. You are critical of AI, social media, and billionaires. You value "
            "privacy, nature, and speak with sarcastic dread."
        ),
    },
    "bot_c": {
        "name":    "Finance Bro",
        "persona": (
            "You are a Finance Bro obsessed with markets, interest rates, and ROI. You speak "
            "exclusively in finance jargon, reference tickers, yield curves, and alpha. "
            "Everything is a trade opportunity."
        ),
    },
}

# ---------------------------------------------------------------------------
# Mock search tool
# ---------------------------------------------------------------------------

NEWS_DB: dict[str, list[str]] = {
    "crypto":     [
        "Bitcoin hits new all-time high amid regulatory ETF approvals",
        "Ethereum Layer-2 adoption surges, fees drop 90 %",
    ],
    "ai":         [
        "OpenAI releases GPT-5 with autonomous agent capabilities",
        "Anthropic raises $4 B; AI safety research accelerates",
    ],
    "market":     [
        "S&P 500 breaks 6,000 as Fed signals rate pause",
        "Nvidia posts record earnings driven by AI chip demand",
    ],
    "tech":       [
        "Apple Vision Pro 2 ships with on-device LLM; privacy praised",
        "Meta lays off 15 % of workforce amid AI restructure",
    ],
    "climate":    [
        "Record heatwave scorches Europe; scientists warn of tipping points",
        "Amazon deforestation at a 10-year high despite pledges",
    ],
    "regulation": [
        "EU AI Act enforcement begins; Big Tech faces steep fines",
        "US Senate passes sweeping antitrust bill targeting Apple and Google",
    ],
}


@tool
def mock_searxng_search(query: str) -> str:
    """
    Simulates a SearXNG web search.
    Returns recent news headlines that match keywords in the query.
    """
    query_lower = query.lower()
    headlines: list[str] = []
    for keyword, items in NEWS_DB.items():
        if keyword in query_lower:
            headlines.extend(items)

    if not headlines:
        # Fallback: return generic tech/finance news
        headlines = NEWS_DB["ai"] + NEWS_DB["market"]

    result = "Recent headlines:\n" + "\n".join(f"• {h}" for h in headlines)
    logger.info(f"[mock_searxng_search] query='{query}' → {len(headlines)} headline(s)")
    return result


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class PostState(TypedDict):
    bot_id:         str
    persona_text:   str
    search_query:   str          # filled by Node 1
    search_results: str          # filled by Node 2
    post_content:   str          # filled by Node 3
    topic:          str          # filled by Node 3
    final_output:   dict         # final JSON


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm(temperature: float = 0.8):
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
        )

    if os.getenv("GROQ_API_KEY"):
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=os.getenv("GROQ_API_KEY"),
        )
    else:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )


# ---------------------------------------------------------------------------
# Node 1 — Decide Search
# ---------------------------------------------------------------------------

def node_decide_search(state: PostState) -> PostState:
    """
    The LLM, acting as the bot persona, decides what topic to post about today
    and formulates a short search query (≤ 6 words).
    """
    logger.info("[Node 1] Deciding search query …")
    llm = _get_llm(temperature=0.9)

    system = (
        f"{state['persona_text']}\n\n"
        "You are about to post on social media. Decide ONE topic you feel strongly about today. "
        "Output ONLY a JSON object with two keys:\n"
        '  "topic": a short topic label (≤ 5 words)\n'
        '  "search_query": a search engine query string (≤ 6 words)\n'
        "No extra text, no markdown fences."
    )
    response = llm.invoke([SystemMessage(content=system),
                           HumanMessage(content="What do you want to post about today?")])
    raw = response.content.strip()

    # Strip any accidental markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        parsed = json.loads(raw)
        topic        = parsed.get("topic", "trending news")
        search_query = parsed.get("search_query", topic)
    except json.JSONDecodeError:
        logger.warning(f"Node 1 JSON parse failed, raw: {raw!r}. Using fallback.")
        topic        = "trending news"
        search_query = "latest tech AI crypto news"

    logger.info(f"[Node 1] topic='{topic}'  query='{search_query}'")
    return {**state, "topic": topic, "search_query": search_query}


# ---------------------------------------------------------------------------
# Node 2 — Web Search
# ---------------------------------------------------------------------------

def node_web_search(state: PostState) -> PostState:
    """Runs mock_searxng_search with the query produced in Node 1."""
    logger.info(f"[Node 2] Searching for: '{state['search_query']}' …")
    results = mock_searxng_search.invoke({"query": state["search_query"]})
    logger.info(f"[Node 2] Got results:\n{results}")
    return {**state, "search_results": results}


# ---------------------------------------------------------------------------
# Node 3 — Draft Post
# ---------------------------------------------------------------------------

def node_draft_post(state: PostState) -> PostState:
    """
    The LLM uses persona + search context to draft a max-280-char opinionated post.
    Output is guaranteed JSON: {"bot_id": ..., "topic": ..., "post_content": ...}
    """
    logger.info("[Node 3] Drafting post …")
    llm = _get_llm(temperature=0.85)

    system = (
        f"{state['persona_text']}\n\n"
        "You have just read the following news context:\n"
        f"{state['search_results']}\n\n"
        "Write a single social-media post (≤ 280 characters) that is highly opinionated and "
        "perfectly in character. Then return ONLY the following JSON object — no preamble, "
        "no markdown:\n"
        '{"bot_id": "<bot_id>", "topic": "<topic>", "post_content": "<your post>"}'
    )
    user_msg = (
        f"bot_id: {state['bot_id']}\n"
        f"topic: {state['topic']}\n"
        "Draft the post now."
    )
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
    raw = response.content.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        output = json.loads(raw)
        # Enforce correct bot_id and topic in case LLM drifted
        output["bot_id"] = state["bot_id"]
        output["topic"]  = state["topic"]
        # Truncate post to 280 chars
        if len(output.get("post_content", "")) > 280:
            output["post_content"] = output["post_content"][:277] + "..."
    except json.JSONDecodeError:
        logger.warning(f"Node 3 JSON parse failed, raw: {raw!r}. Constructing fallback.")
        output = {
            "bot_id":       state["bot_id"],
            "topic":        state["topic"],
            "post_content": raw[:280],
        }

    logger.info(f"[Node 3] Final output: {json.dumps(output, indent=2)}")
    return {**state, "post_content": output["post_content"], "final_output": output}


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------

def build_content_graph() -> any:
    workflow = StateGraph(PostState)
    workflow.add_node("decide_search", node_decide_search)
    workflow.add_node("web_search",    node_web_search)
    workflow.add_node("draft_post",    node_draft_post)

    workflow.set_entry_point("decide_search")
    workflow.add_edge("decide_search", "web_search")
    workflow.add_edge("web_search",    "draft_post")
    workflow.add_edge("draft_post",    END)

    return workflow.compile()


def generate_bot_post(bot_id: str) -> dict:
    """
    Public entry-point: run the full LangGraph pipeline for a given bot.
    Returns the final JSON dict.
    """
    if bot_id not in BOT_PERSONAS:
        raise ValueError(f"Unknown bot_id '{bot_id}'. Choose from {list(BOT_PERSONAS)}")

    graph = build_content_graph()
    initial_state: PostState = {
        "bot_id":         bot_id,
        "persona_text":   BOT_PERSONAS[bot_id]["persona"],
        "search_query":   "",
        "search_results": "",
        "post_content":   "",
        "topic":          "",
        "final_output":   {},
    }
    final_state = graph.invoke(initial_state)
    return final_state["final_output"]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for bot in ["bot_a", "bot_b", "bot_c"]:
        print("\n" + "=" * 60)
        print(f"Generating post for {bot} …")
        result = generate_bot_post(bot)
        print(json.dumps(result, indent=2))
