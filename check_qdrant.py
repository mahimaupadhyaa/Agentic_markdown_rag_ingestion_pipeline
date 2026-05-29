"""
Quick inspection script for Qdrant collection.
Shows counts by level, sample content for each level,
and lets you search by level or namespace.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import config

URL = config.get("database", "url", default="http://localhost:6333")
COLLECTION = config.get("database", "collection_name", default="collection_demo")
NAMESPACE = config.get("database", "namespace", default="CaseDoneDemo")

client = QdrantClient(url=URL)


def count_by_level(namespace: str = None):
    print(f"\n=== Counts by level (namespace={namespace or 'all'}) ===")
    for level in ["table", "parent", "chunk", "proposition"]:
        must = [FieldCondition(key="metadata.level", match=MatchValue(value=level))]
        if namespace:
            must.append(FieldCondition(key="metadata.namespace", match=MatchValue(value=namespace)))
        result = client.count(
            collection_name=COLLECTION,
            count_filter=Filter(must=must),
            exact=True,
        )
        print(f"  {level:15s}: {result.count}")


def show_samples(level: str, namespace: str = None, n: int = 2):
    print(f"\n=== Sample '{level}' documents ===")
    must = [FieldCondition(key="metadata.level", match=MatchValue(value=level))]
    if namespace:
        must.append(FieldCondition(key="metadata.namespace", match=MatchValue(value=namespace)))

    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(must=must),
        limit=n,
        with_payload=True,
        with_vectors=False,
    )

    if not results:
        print(f"  No '{level}' documents found.")
        return

    for i, point in enumerate(results):
        content = point.payload.get("page_content", "")
        meta = point.payload.get("metadata", {})
        print(f"\n  [{i+1}] id={point.id}")
        print(f"      section : {meta.get('section', '')[:60]}")
        print(f"      source  : {meta.get('source', '')}")
        print(f"      content : {content[:300]}")
        print()


def collection_summary():
    info = client.get_collection(COLLECTION)
    print(f"\n=== Collection: {COLLECTION} ===")
    print(f"  Total points : {info.points_count}")
    print(f"  Status       : {info.status}")


if __name__ == "__main__":
    collection_summary()
    count_by_level(namespace=NAMESPACE)
    print()
    for level in ["table", "parent", "proposition"]:
        show_samples(level, namespace=NAMESPACE, n=1)