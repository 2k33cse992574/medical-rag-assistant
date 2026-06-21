from src.rag_chain import ask_question

result = ask_question("What were the reductions in mortality and hospitalization seen with carvedilol in the US Carvedilol Heart Failure Trials Program?")
print("ANSWER:")
print(result["answer"])
print("\nSOURCES:")
for s in result["sources"]:
    print("---", s.get("filename"), "---")
    print(s.get("content", "")[:300])
