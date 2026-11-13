// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SQLite-backed trace store for the routing transcript portal.
//!
//! [`TraceStore`] mirrors the routing JSONL record (timing + tokens) plus the
//! captured request/response/wire payloads so a session's full per-turn
//! transcript can be read back over HTTP. It is additive to the JSONL routing
//! log; enabling `--trace-db` writes one row per finished call.

use std::path::Path;
use std::sync::Mutex;
use std::time::SystemTime;

use humantime::format_rfc3339_millis;
use rusqlite::{params, Connection, Result as SqliteResult};
use serde_json::Value;

use switchyard_protocol::Usage;

/// Byte cap applied to each captured body column before insertion.
pub(crate) const CAPTURED_BODY_CAP: usize = 512 * 1024;

/// One finished call's trace row. Payload columns hold the redacted,
/// truncated-to-[\`CAPTURED_BODY_CAP\`] JSON text; `truncated` is set when any
/// of the three bodies exceeded the cap.
pub(crate) struct TraceRow {
    pub session_id: Option<String>,
    pub turn_id: Option<String>,
    pub ts: String,
    pub model: String,
    pub route_id: String,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub completion_tokens: u64,
    pub reasoning_tokens: u64,
    pub duration_s: f64,
    pub ttft_s: Option<f64>,
    pub request_body: Option<String>,
    pub response_body: Option<String>,
    pub wire_body: Option<String>,
    pub truncated: bool,
}

/// One finished call's trace inputs, owned so it can be moved into an async
/// stream. The store and redactor are shared [`Arc`] handles into the
/// [`ServerState`](crate::ServerState); the request/wire bodies are captured
/// before the call is routed and redacted at write time.
pub(crate) struct TraceCapture {
    pub(crate) store: std::sync::Arc<TraceStore>,
    pub(crate) redactor: std::sync::Arc<crate::redaction::Redactor>,
    pub(crate) session_id: Option<String>,
    pub(crate) turn_id: Option<String>,
    pub(crate) route_id: String,
    pub(crate) request_body: Value,
    pub(crate) wire_body: Value,
}

impl TraceCapture {
    /// Persists one finished call. `request_body`, `wire_body`, and
    /// `response_body` are redacted against [`Redactor`](crate::redaction::Redactor)
    /// and capped at [`CAPTURED_BODY_CAP`] before insertion; `truncated` is set
    /// when any of the three exceeded the cap.
    pub(crate) fn write(
        &self,
        model: &str,
        usage: &Usage,
        duration_s: f64,
        ttft_s: Option<f64>,
        response_body: &Value,
    ) -> std::io::Result<()> {
        let redactor = &self.redactor;
        let (request_body, req_trunc) =
            cap_body(Some(redactor.value(self.request_body.clone())));
        let (wire_body, wire_trunc) = cap_body(Some(redactor.value(self.wire_body.clone())));
        let (response_body, resp_trunc) = cap_body(Some(redactor.value(response_body.clone())));
        let usage = crate::usage_metrics::token_usage(usage);
        let row = TraceRow {
            session_id: self.session_id.clone(),
            turn_id: self.turn_id.clone(),
            ts: format_rfc3339_millis(SystemTime::now()).to_string(),
            model: model.to_string(),
            route_id: self.route_id.clone(),
            prompt_tokens: usage.prompt_tokens,
            cached_tokens: usage.cached_tokens,
            completion_tokens: usage.completion_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            duration_s,
            ttft_s,
            request_body,
            response_body,
            wire_body,
            truncated: req_trunc || wire_trunc || resp_trunc,
        };
        self.store
            .record(&row)
            .map_err(|error| std::io::Error::other(error.to_string()))
    }
}

/// A SQLite-backed store writing `trace_records` rows for the transcript portal.
pub(crate) struct TraceStore {
    conn: Mutex<Connection>,
}

impl TraceStore {
    fn poisoned() -> rusqlite::Error {
        rusqlite::Error::SqliteFailure(
            rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
            Some("trace store connection poisoned".to_string()),
        )
    }

    /// Opens (creating the directory + table as needed) the trace database.
    pub(crate) fn open(path: &Path) -> SqliteResult<Self> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|_| Self::poisoned())?;
        }
        let conn =
            Connection::open(path).map_err(|_| Self::poisoned())?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS trace_records (
                session_id TEXT,
                turn_id TEXT,
                ts TEXT NOT NULL,
                model TEXT NOT NULL,
                route_id TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                cached_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                reasoning_tokens INTEGER NOT NULL,
                duration_s REAL NOT NULL,
                ttft_s REAL,
                request_body TEXT,
                response_body TEXT,
                wire_body TEXT,
                truncated INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_trace_records_session ON trace_records (session_id);",
        )?;
        Ok(Self { conn: Mutex::new(conn) })
    }
    /// Inserts one finished-call row.
    pub(crate) fn record(&self, row: &TraceRow) -> SqliteResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| Self::poisoned())?;
        conn.execute(
            "INSERT INTO trace_records (
                session_id, turn_id, ts, model, route_id,
                prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens,
                duration_s, ttft_s,
                request_body, response_body, wire_body, truncated
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5,
                ?6, ?7, ?8, ?9,
                ?10, ?11,
                ?12, ?13, ?14, ?15
             )",
            params![
                row.session_id,
                row.turn_id,
                row.ts,
                row.model,
                row.route_id,
                row.prompt_tokens as i64,
                row.cached_tokens as i64,
                row.completion_tokens as i64,
                row.reasoning_tokens as i64,
                row.duration_s,
                row.ttft_s,
                row.request_body,
                row.response_body,
                row.wire_body,
                row.truncated,
            ],
        )?;
        Ok(())
    }

    /// Sentinel session id for rows whose `session_id` is NULL (no client
    /// `x-switchyard-session-id` header was sent).  The portal renders this
    /// as "Unattributed".
    pub const UNATTRIBUTED: &str = "__unattributed__";

    /// Distinct session ids with their call count and first/last timestamp,
    /// newest first.  Rows with NULL session_id are grouped under the
    /// [`UNATTRIBUTED`] sentinel.  `limit` is clamped to 1..=500.
    pub(crate) fn sessions(
        &self,
        limit: usize,
    ) -> SqliteResult<Vec<(String, u64, String, String)>> {
        let limit = limit.clamp(1, 500);
        let conn = self
            .conn
            .lock()
            .map_err(|_| Self::poisoned())?;
        let mut stmt = conn.prepare(
            "SELECT COALESCE(s.session_id, ?1) AS session_id,
                    COUNT(*) AS calls, MIN(r.ts) AS first_ts, MAX(r.ts) AS last_ts
             FROM (SELECT DISTINCT session_id FROM trace_records) AS s
             JOIN trace_records AS r ON (r.session_id = s.session_id
                OR (s.session_id IS NULL AND r.session_id IS NULL))
             GROUP BY COALESCE(s.session_id, ?1)
             ORDER BY last_ts DESC
             LIMIT ?2",
        )?;
        let mut rows = stmt.query(params![Self::UNATTRIBUTED, limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let session_id: String = row.get(0)?;
            let calls: i64 = row.get(1)?;
            let first_ts: String = row.get(2)?;
            let last_ts: String = row.get(3)?;
            out.push((session_id, calls as u64, first_ts, last_ts));
        }
        Ok(out)
    }

    /// All `trace_records` rows for one session in timestamp order, each as a
    /// JSON object mirroring the record (bodies parse back to JSON when intact,
    /// otherwise a raw string for a truncated body).  `limit` caps the row
    /// count; `offset` skips leading rows.  Both `None` returns every row.
    pub(crate) fn session_transcript(
        &self,
        session_id: &str,
        limit: Option<u64>,
        offset: Option<u64>,
    ) -> SqliteResult<Vec<Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| Self::poisoned())?;
        let is_unattributed = session_id == Self::UNATTRIBUTED;
        let bind_id: Option<&str> = if is_unattributed { None } else { Some(session_id) };
        let mut sql = String::from(
            "SELECT ts, session_id, turn_id, model, route_id,
                    prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens,
                    duration_s, ttft_s,
                    request_body, response_body, wire_body, truncated
             FROM trace_records WHERE session_id IS ?1
             ORDER BY ts ASC, rowid ASC",
        );
        if limit.is_some() || offset.is_some() {
            sql.push_str(" LIMIT ?2 OFFSET ?3");
        }
        let mut stmt = conn.prepare(&sql)?;
        let mut rows_iter = if limit.is_some() || offset.is_some() {
            stmt.query(params![bind_id, limit.unwrap_or(0) as i64, offset.unwrap_or(0) as i64])?
        } else {
            stmt.query(params![bind_id])?
        };
        let mut out = Vec::new();
        while let Some(row) = rows_iter.next()? {
            let truncated: i64 = row.get(14)?;
            out.push(serde_json::json!({
                "ts": row.get::<_, String>(0)?,
                "session_id": row.get::<_, Option<String>>(1)?,
                "turn_id": row.get::<_, Option<String>>(2)?,
                "model": row.get::<_, String>(3)?,
                "route_id": row.get::<_, String>(4)?,
                "prompt_tokens": row.get::<_, i64>(5)? as u64,
                "cached_tokens": row.get::<_, i64>(6)? as u64,
                "completion_tokens": row.get::<_, i64>(7)? as u64,
                "reasoning_tokens": row.get::<_, i64>(8)? as u64,
                "duration_s": row.get::<_, f64>(9)?,
                "ttft_s": row.get::<_, Option<f64>>(10)?,
                "request": parse_json_text(row.get::<_, Option<String>>(11)?),
                "response": parse_json_text(row.get::<_, Option<String>>(12)?),
                "wire": parse_json_text(row.get::<_, Option<String>>(13)?),
                "truncated": truncated != 0,
            }));
        }
        Ok(out)
    }
}

/// Parses a stored body back to a `Value`; non-JSON text (e.g. a truncated
/// body that was cut mid-document) is returned as a plain string.
fn parse_json_text(text: Option<String>) -> Value {
    match text {
        Some(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
        None => Value::Null,
    }
}

/// Caps a captured JSON body to [`CAPTURED_BODY_CAP`] bytes, returning the
/// stored text (a JSON document or a truncated JSON string) and whether it was
/// truncated.
pub(crate) fn cap_body(body: Option<Value>) -> (Option<String>, bool) {
    let Some(body) = body else {
        return (None, false);
    };
    let Ok(text) = serde_json::to_string(&body) else {
        return (None, false);
    };
    if text.len() <= CAPTURED_BODY_CAP {
        return (Some(text), false);
    }
    // Truncate at a char boundary so the stored text stays valid UTF-8. The
    // result is a JSON prefix that no longer parses as a whole document; the
    // reader falls back to the raw string when that happens.
    let boundary = text.floor_char_boundary(CAPTURED_BODY_CAP);
    (Some(text[..boundary].to_string()), true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_oversized_bodies_at_the_byte_cap() {
        let big = "x".repeat(CAPTURED_BODY_CAP + 1024);
        let (capped, truncated) = cap_body(Some(Value::String(big)));
        assert!(truncated);
        let Some(text) = capped else {
            panic!("expected capped text");
        };
        assert!(text.len() <= CAPTURED_BODY_CAP + 512, "cap exceeded: {}", text.len());
    }

    #[test]
    fn small_bodies_are_not_truncated() {
        let (capped, truncated) = cap_body(Some(json_body("hi")));
        assert!(!truncated);
        assert_eq!(capped, Some(r#""hi""#.to_string()));
    }

    #[test]
    fn absent_bodies_stay_none() {
        let (capped, truncated) = cap_body(None);
        assert!(!truncated);
        assert!(capped.is_none());
    }

    #[test]
    fn store_round_trips_a_session() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("trace.db");
        let store = TraceStore::open(&path).expect("open");
        for model in ["a", "b", "c"] {
            let capture = TraceCapture {
                store: std::sync::Arc::new(TraceStore::open(&path).expect("open")),
                redactor: std::sync::Arc::new(crate::redaction::Redactor::default()),
                session_id: Some("sess-1".to_string()),
                turn_id: Some("turn-1".to_string()),
                route_id: "route".to_string(),
                request_body: json_body(model),
                wire_body: json_body(model),
            };
            capture
                .write(model, &Usage::default(), 0.5, Some(0.1), &json_body(model))
                .expect("write");
        }
        let sessions = store.sessions(50).expect("sessions");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].0, "sess-1");
        assert_eq!(sessions[0].1, 3);
        let transcript = store.session_transcript("sess-1", None, None).expect("transcript");
        assert_eq!(transcript.len(), 3);
        assert_eq!(transcript[0]["model"], "a");
        assert_eq!(transcript[0]["duration_s"], 0.5);
        let missing = store.session_transcript("nope", None, None).expect("missing");
        assert!(missing.is_empty());
    }

    fn json_body(s: &str) -> Value {
        serde_json::json!(s)
    }
}
