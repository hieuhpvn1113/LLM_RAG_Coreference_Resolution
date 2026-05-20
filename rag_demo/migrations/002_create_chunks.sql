-- =============================================
-- Migration 002: Tạo bảng chunks + search_logs
-- =============================================
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id      UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,

    level       SMALLINT NOT NULL CHECK (level IN (0, 1, 2)),
    parent_id   UUID REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    prev_id     UUID REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    next_id     UUID REFERENCES chunks(chunk_id) ON DELETE SET NULL,
    seq_no      TEXT NOT NULL DEFAULT '0',

    raw_text    TEXT NOT NULL,
    clean_text  TEXT,
    title       TEXT,
    summary     TEXT,

    keywords                JSONB DEFAULT '[]',
    entities                JSONB DEFAULT '[]',
    hypothetical_questions  JSONB DEFAULT '[]',
    relations               JSONB DEFAULT '[]',

    token_count     INTEGER,
    source_file     TEXT,
    page_no         INTEGER,
    char_start      INTEGER,
    char_end        INTEGER,
    embed_model     TEXT,
    embed_status    TEXT DEFAULT 'pending'
                    CHECK (embed_status IN ('pending', 'done', 'error')),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS search_logs (
    log_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_original  TEXT NOT NULL,
    query_rewritten JSONB,
    chunks_retrieved JSONB,
    llm_response    TEXT,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
