from agentic_rag import AgenticRAG

rag = AgenticRAG()

while True:
    query = input("\nYou: ")
    if query.lower() in ["exit", "quit"]:
        break
    response = rag.run(query)
    print(f"\nAssistant: {response}")