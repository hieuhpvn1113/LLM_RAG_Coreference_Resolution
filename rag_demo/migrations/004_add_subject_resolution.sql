-- =============================================
-- Migration 004: Subject resolution support
-- =============================================

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS subject_name TEXT;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS doc_subject TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_subject_name
    ON documents (subject_name);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_subject
    ON chunks (doc_subject);

