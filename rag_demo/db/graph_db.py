# db/graph_db.py — Neo4j Graph DB client
from neo4j import GraphDatabase
from config import NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD
import re


class GraphDB:

    def __init__(self):
        self.driver = None

    def connect(self):
        self.driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.driver.verify_connectivity()
        print(f"✅ Neo4j connected: {NEO4J_URL}")

    def close(self):
        if self.driver:
            self.driver.close()

    def ensure_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
            session.run("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE")
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
        print("  Neo4j constraints ensured")

    def upsert_document(self, doc_id: str, file_name: str):
        with self.driver.session() as session:
            session.run(
                "MERGE (d:Document {doc_id: $doc_id}) SET d.file_name = $file_name",
                doc_id=doc_id, file_name=file_name,
            )

    def upsert_chunk_node(self, chunk: dict):
        with self.driver.session() as session:
            session.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.doc_id = $doc_id, c.level = $level,
                    c.seq_no = $seq_no, c.title = $title, c.summary = $summary
                WITH c
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (c)-[:BELONGS_TO]->(d)
                """,
                chunk_id=chunk["chunk_id"], doc_id=chunk["doc_id"],
                level=chunk.get("level", 2), seq_no=chunk.get("seq_no", "0"),
                title=chunk.get("title", ""), summary=chunk.get("summary", ""),
            )

    def create_parent_relationship(self, parent_id: str, child_id: str):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (parent:Chunk {chunk_id: $parent_id})
                MATCH (child:Chunk  {chunk_id: $child_id})
                MERGE (parent)-[:PARENT_OF]->(child)
                """,
                parent_id=parent_id, child_id=child_id,
            )

    def create_next_relationship(self, from_id: str, to_id: str):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:Chunk {chunk_id: $from_id})
                MATCH (b:Chunk {chunk_id: $to_id})
                MERGE (a)-[:NEXT]->(b)
                """,
                from_id=from_id, to_id=to_id,
            )

    def upsert_entities(self, chunk_id: str, entities: list):
        if not entities:
            return
        with self.driver.session() as session:
            for entity in entities:
                name  = entity.get("name", "").strip()
                etype = entity.get("type", "CONCEPT").strip()
                if not name:
                    continue
                session.run(
                    """
                    MERGE (e:Entity {name: $name})
                    SET e.type = $etype
                    WITH e
                    MATCH (c:Chunk {chunk_id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    name=name, etype=etype, chunk_id=chunk_id,
                )

    def upsert_relations(self, relations: list):
        if not relations:
            return
        with self.driver.session() as session:
            for rel in relations:
                from_name = rel.get("from", "").strip()
                to_name   = rel.get("to", "").strip()
                relation  = rel.get("relation", "RELATES_TO").strip()
                if not from_name or not to_name:
                    continue
                session.run(
                    """
                    MERGE (a:Entity {name: $from_name})
                    MERGE (b:Entity {name: $to_name})
                    MERGE (a)-[r:RELATES_TO]->(b)
                    SET r.relation_type = $relation
                    """,
                    from_name=from_name, to_name=to_name, relation=relation,
                )

    def write_chunk_full(self, chunk: dict):
        """Write toàn bộ Level 2 chunk vào Neo4j sau enrichment."""
        self.upsert_chunk_node(chunk)
        if chunk.get("parent_id"):
            self.create_parent_relationship(chunk["parent_id"], chunk["chunk_id"])
        if chunk.get("prev_id"):
            self.create_next_relationship(chunk["prev_id"], chunk["chunk_id"])
        self.upsert_entities(chunk["chunk_id"], chunk.get("entities", []))
        self.upsert_relations(chunk.get("relations", []))

    # ─────────────────────────────────────────────────────────────────────────
    # Entity Extraction từ Query
    # ─────────────────────────────────────────────────────────────────────────

    def extract_entities_simple(self, user_query: str) -> list:
        """
        Tìm entity names trong Neo4j khớp với query.

        Vấn đề với cách cũ (đã sửa):
          WHERE toLower($query) CONTAINS toLower(e.name)
          → Chỉ tìm được entity ngắn hơn query (VD entity="Đảng" trong query dài)
          → Entity dài như "Luật Giao thông đường bộ năm 2008" KHÔNG BAO GIỜ được tìm thấy

        Cách mới — bidirectional matching theo từng keyword phrase:
          1. Tách query thành các keyword phrases (split by comma/whitespace)
          2. Lọc phrase quá ngắn (< 3 ký tự) để tránh match stopword
          3. Kiểm tra 2 chiều:
             a) entity_name CONTAINS phrase   → entity dài hơn phrase (phổ biến nhất)
             b) phrase CONTAINS entity_name   → entity ngắn hơn phrase (ít gặp hơn)

        Ví dụ:
          query  = "luật giao thông, chương chính, cấu trúc"
          phrase = "luật giao thông"
          entity = "Luật Giao thông đường bộ năm 2008"
          → entity CONTAINS phrase → MATCH ✓
        """
        if not user_query or not user_query.strip():
            return []

        # Tách thành các phrase tại dấu phẩy, sau đó tại khoảng trắng nếu quá dài
        raw_phrases = re.split(r'[,;]+', user_query)
        phrases = []
        for ph in raw_phrases:
            ph = ph.strip()
            if not ph:
                continue
            # Nếu phrase < 3 từ → dùng nguyên
            # Nếu phrase >= 3 từ → tách thêm các bigram để tăng recall
            words = ph.split()
            phrases.append(ph)  # phrase gốc
            if len(words) >= 3:
                # Thêm các bigram liên tiếp (cửa sổ 2 từ)
                for i in range(len(words) - 1):
                    bigram = f"{words[i]} {words[i+1]}"
                    if len(bigram) >= 4:
                        phrases.append(bigram)

        # Lọc phrase quá ngắn (< 3 ký tự), bỏ duplicate
        min_len  = 3
        seen     = set()
        filtered = []
        for p in phrases:
            p_lower = p.lower()
            if len(p) >= min_len and p_lower not in seen:
                seen.add(p_lower)
                filtered.append(p)

        if not filtered:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $phrases AS phrase
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(phrase)
                   OR toLower(phrase) CONTAINS toLower(e.name)
                RETURN DISTINCT e.name AS name
                LIMIT 15
                """,
                phrases=filtered,
            )
            return [r["name"] for r in result.data()]

    # ─────────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────────

    def search(self, query_entities: list, top_k: int = 3) -> list:
        """
        Tìm các L2 chunk có nhiều entity khớp nhất.
        Dùng kết quả từ extract_entities_simple() làm input.

        Scoring: số entity khớp / tổng entity trong chunk
        → ưu tiên chunk nhỏ khớp chính xác hơn chunk lớn khớp 1/100 entity.
        """
        if not query_entities:
            return []

        with self.driver.session() as session:
            result = session.run(
                """
                UNWIND $entity_names AS ename
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower(ename)
                   OR toLower(ename) CONTAINS toLower(e.name)
                MATCH (c:Chunk)-[:MENTIONS]->(e)
                WHERE c.level = 2
                WITH c, count(DISTINCT e) AS match_count
                ORDER BY match_count DESC
                LIMIT $top_k
                RETURN c.chunk_id AS chunk_id,
                       c.title    AS title,
                       c.summary  AS summary,
                       match_count AS score
                """,
                entity_names=query_entities,
                top_k=top_k,
            )
            rows = result.data()

        return [
            {
                "chunk_id":   r["chunk_id"],
                "score":      float(r["score"]),
                "title":      r.get("title", ""),
                "clean_text": "",
                "source":     "neo4j",
            }
            for r in rows
        ]

    def delete_by_doc(self, doc_id: str):
        with self.driver.session() as session:
            session.run(
                "MATCH (c:Chunk {doc_id: $doc_id}) DETACH DELETE c",
                doc_id=doc_id,
            )
