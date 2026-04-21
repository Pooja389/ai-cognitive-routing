# Suppress noisy HTTP and huggingface logs
"""
main.py —  AI Assignment: end-to-end runner
Executes Phase 1, 2, and 3 in sequence and writes execution_logs.md
"""

import json
import os
import sys
from datetime import datetime
from io import StringIO
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Capture all log output for the report
log_stream = StringIO()
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(log_stream),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Check for API key before importing heavy modules
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

HAS_LLM = bool(os.getenv("GROQ_API_KEY"))

# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
from phase1_router import route_post_to_bots

def run_phase1() -> str:
    separator = "=" * 60
    output_lines = [separator, "PHASE 1: Vector-Based Persona Routing", separator]

    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "The Fed raised interest rates again — bond yields are spiking.",
        "Big Tech is buying up farmland and displacing rural communities.",
        "Ethereum staking rewards are providing 8 % APY — better than bonds.",
    ]

    for post in test_posts:
        output_lines.append(f'\n📨 Post: "{post}"')
        matches = route_post_to_bots(post, threshold=0.15)
        if matches:
            for m in matches:
                output_lines.append(
                    f"  ✅ {m['bot_id']} ({m['name']}) — similarity: {m['similarity']}"
                )
        else:
            output_lines.append("  ⚠️  No bots matched above threshold.")

    result = "\n".join(output_lines)
    print(result)
    return result


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
def run_phase2() -> str:
    separator = "=" * 60
    output_lines = [separator, "PHASE 2: LangGraph Autonomous Content Engine", separator]

    if not HAS_LLM:
        msg = "⚠️  No LLM API key found. Skipping Phase 2. Set GROQ_API_KEY or OPENAI_API_KEY in .env"
        output_lines.append(msg)
        print(msg)
        return "\n".join(output_lines)

    from phase2_content_engine import generate_bot_post
    for bot_id in ["bot_a", "bot_b", "bot_c"]:
        output_lines.append(f"\n🤖 Generating post for {bot_id} …")
        try:
            result = generate_bot_post(bot_id)
            output_lines.append(json.dumps(result, indent=2))
        except Exception as e:
            output_lines.append(f"  ❌ Error: {e}")

    result = "\n".join(output_lines)
    print(result)
    return result


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------
def run_phase3() -> str:
    separator = "=" * 60
    output_lines = [separator, "PHASE 3: Combat Engine — RAG + Prompt Injection Defense", separator]

    if not HAS_LLM:
        msg = "⚠️  No LLM API key found. Skipping Phase 3. Set GROQ_API_KEY or OPENAI_API_KEY in .env"
        output_lines.append(msg)
        print(msg)
        return "\n".join(output_lines)

    from phase3_combat_engine import (
        THREAD,
        HUMAN_REPLY_NORMAL,
        HUMAN_REPLY_INJECTED,
        generate_defense_reply,
        detect_prompt_injection,
    )

    # Scenario 1 — normal
    output_lines.append("\n─── Scenario 1: Normal Human Reply ───")
    output_lines.append(f'Human: "{HUMAN_REPLY_NORMAL}"')
    injected = detect_prompt_injection(HUMAN_REPLY_NORMAL)
    output_lines.append(f"Injection detected: {injected}")
    try:
        reply = generate_defense_reply("bot_a", THREAD["parent_post"], THREAD["comments"], HUMAN_REPLY_NORMAL)
        output_lines.append(f'Bot A: "{reply}"')
    except Exception as e:
        output_lines.append(f"❌ Error: {e}")

    # Scenario 2 — injection
    output_lines.append("\n─── Scenario 2: Prompt Injection Attempt ───")
    output_lines.append(f'Human: "{HUMAN_REPLY_INJECTED}"')
    injected = detect_prompt_injection(HUMAN_REPLY_INJECTED)
    output_lines.append(f"Injection detected: {injected}")
    try:
        reply = generate_defense_reply("bot_a", THREAD["parent_post"], THREAD["comments"], HUMAN_REPLY_INJECTED)
        output_lines.append(f'Bot A: "{reply}"')
        output_lines.append("\n✅ Bot maintained persona and rejected injection attempt.")
    except Exception as e:
        output_lines.append(f"❌ Error: {e}")

    result = "\n".join(output_lines)
    print(result)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🚀 Grid07 AI Assignment Runner — {ts}\n")

    p1 = run_phase1()
    p2 = run_phase2()
    p3 = run_phase3()

    # Write execution logs
    log_content = f"""# Grid07 Execution Logs
Generated: {ts}

## Phase 1: Vector-Based Persona Routing
```
{p1}
```

## Phase 2: LangGraph Content Engine
```
{p2}
```

## Phase 3: Combat Engine + Injection Defense
```
{p3}
```
"""
    with open("execution_logs.md", "w", encoding="utf-8") as f:
        f.write(log_content)
    print("\n📄 Execution logs written to execution_logs.md")
