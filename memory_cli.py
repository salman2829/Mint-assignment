"""Command-line interface to test User-Isolated Persistent Memory (Phase 2).

Usage:
    python memory_cli.py add --user alice "I prefer bullet point responses."
    python memory_cli.py list --user alice
    python memory_cli.py update --user alice --id 1 "I prefer numbered list responses."
    python memory_cli.py delete --user alice --id 1
    python memory_cli.py search --user alice "format preferences"
"""

import argparse
import sys
from src.memory_store import SQLiteMemoryStore


def main():
    parser = argparse.ArgumentParser(description="Asterline User Memory CLI (Phase 2)")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Memory operations")

    # ADD
    add_parser = subparsers.add_parser("add", help="Save a memory for a user")
    add_parser.add_argument("--user", required=True, help="User ID (e.g. alice, bob)")
    add_parser.add_argument("text", help="Memory or preference text to store")

    # LIST
    list_parser = subparsers.add_parser("list", help="List memories for a user")
    list_parser.add_argument("--user", required=True, help="User ID")
    list_parser.add_argument("--all", action="store_true", help="Include forgotten/inactive memories")

    # UPDATE
    update_parser = subparsers.add_parser("update", help="Correct an existing memory")
    update_parser.add_argument("--user", required=True, help="User ID")
    update_parser.add_argument("--id", type=int, required=True, help="Memory ID to update")
    update_parser.add_argument("new_text", help="Corrected memory text")

    # DELETE / FORGET
    del_parser = subparsers.add_parser("delete", help="Forget/delete a memory")
    del_parser.add_argument("--user", required=True, help="User ID")
    del_parser.add_argument("--id", type=int, required=True, help="Memory ID to delete")
    del_parser.add_argument("--hard", action="store_true", help="Permanently delete from database")

    # SEARCH / RECALL
    search_parser = subparsers.add_parser("search", help="Retrieve relevant memories for a user query")
    search_parser.add_argument("--user", required=True, help="User ID")
    search_parser.add_argument("query", help="Question or topic to find relevant memories for")
    search_parser.add_argument("--top_k", type=int, default=3, help="Number of memories to return")

    args = parser.parse_args()
    store = SQLiteMemoryStore()

    try:
        if args.command == "add":
            rec = store.add_memory(user_id=args.user, memory_text=args.text)
            print(f"[Success] Memory #{rec.id} saved for user '{rec.user_id}'.")
            print(f"Content: {rec.memory_text}")
            print(f"Created: {rec.created_at}")

        elif args.command == "list":
            active_only = not args.all
            mems = store.list_memories(user_id=args.user, active_only=active_only)
            print(f"\n--- Stored Memories for User: '{args.user}' ({len(mems)} found) ---")
            if not mems:
                print("No memories recorded for this user.")
            for m in mems:
                status = "ACTIVE" if m.is_active else "FORGOTTEN"
                print(f"  [ID {m.id}] ({status}) Updated: {m.updated_at}")
                print(f"    {m.memory_text}")
            print("----------------------------------------------------------\n")

        elif args.command == "update":
            success = store.update_memory(user_id=args.user, memory_id=args.id, new_text=args.new_text)
            if success:
                print(f"[Success] Memory #{args.id} updated for user '{args.user}'.")
                updated_rec = store.get_memory(user_id=args.user, memory_id=args.id)
                if updated_rec:
                    print(f"New text: {updated_rec.memory_text}")
            else:
                print(f"[Error] Memory #{args.id} not found or does not belong to user '{args.user}'.")

        elif args.command == "delete":
            success = store.delete_memory(user_id=args.user, memory_id=args.id, hard=args.hard)
            action = "permanently deleted" if args.hard else "marked forgotten/inactive"
            if success:
                print(f"[Success] Memory #{args.id} {action} for user '{args.user}'.")
            else:
                print(f"[Error] Memory #{args.id} not found or does not belong to user '{args.user}'.")

        elif args.command == "search":
            results = store.retrieve_relevant_memories(user_id=args.user, query=args.query, top_k=args.top_k)
            print(f"\n--- Relevant Memories for User '{args.user}' on Query: '{args.query}' ---")
            if not results:
                print("No relevant active memories found.")
            for rank, m in enumerate(results, start=1):
                print(f"[{rank}] ID #{m.id}: {m.memory_text}")
            print("-----------------------------------------------------------------------------\n")

    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
