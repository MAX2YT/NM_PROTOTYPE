import re

_MEDICAL_ENTITY = re.compile(
    r"\b(?:[A-Z]{2,6}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|"
    r"[a-z]+(?:itis|osis|emia|pathy|ectomy|plasty|scopy|gram))\b"
)

query = "What are the treatment guidelines for Type 2 Diabetes Mellitus?"
entities = set(m.lower() for m in _MEDICAL_ENTITY.findall(query))
print(f"Query: {query}")
print(f"Entities: {entities}")

query2 = "What is the stock price of Apollo Hospitals today?"
entities2 = set(m.lower() for m in _MEDICAL_ENTITY.findall(query2))
print(f"\nQuery: {query2}")
print(f"Entities: {entities2}")
