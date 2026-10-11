"""
StateGuard in an agent: check tool calls before they run, and wrap a chat
model so it asks when unsure and is told what changed when a request is stale.

Run:  python examples/agent_and_chat.py      (no API key needed: uses a fake model)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stateguard import Decision, StateGuard
from stateguard.chat import GuardedChat

guard = StateGuard()
guard.remember("me", "location", "Bengaluru", valid_from="2025-03-01")
guard.remember("me", "location", "Pune", valid_from="2026-04-10", reason="moved for work")
guard.remember("me", "diet", "vegetarian", valid_from="2026-01-01")
guard.remember("me", "diet", "vegan", valid_from="2026-07-02")

# 1) Tool calls -------------------------------------------------------------
call = {"origin": "Bengaluru", "destination": "Delhi", "date": "2026-11-02"}
r = guard.check_tool_call("book_flight", call, ["me"], arg_map={"origin": "location"})
print("book_flight:", r.decision.value, "->", r.suggested_arguments)
if r.decision is Decision.BLOCK:
    call = r.suggested_arguments          # or ask the user before running it

# 2) Chat ------------------------------------------------------------------
def fake_model(messages):
    notes = [m["content"] for m in messages if m["role"] == "system"]
    return f"(model reply; memory notes seen: {len(notes)})"

chat = GuardedChat(guard, fake_model, entity_id="me")
for msg in ["Book me a vegetarian dinner for Friday",      # BLOCK -> answered with a note
            "Is a vegetarian thali still fine for me?",     # REVIEW -> asks first
            "yes, keep it vegan"]:                          # answers the original request
    turn = chat.send(msg)
    print(f"\nuser: {msg}\nbot:  {turn.text}")
