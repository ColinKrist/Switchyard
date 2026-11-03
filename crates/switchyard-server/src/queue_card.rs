// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A short JSON card of who is already busy, refreshed off the request path.
//!
//! The decision judge reads it from request metadata. Missing or failed polls
//! stay in the card as `ok: false` so the judge can treat that engine as busy
//! instead of inventing a free queue.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde_json::{Value, json};

pub const QUEUE_KEY: &str = "switchyard.queue";
const HIGH_URL_ENV: &str = "SWITCHYARD_QUEUE_HIGH_URL";
const TASK_URL_ENV: &str = "SWITCHYARD_QUEUE_TASK_URL";
pub fn spawn(http: reqwest::Client, card: Arc<RwLock<String>>) {
    let high = std::env::var(HIGH_URL_ENV).ok().filter(|url| !url.is_empty());
    let task = std::env::var(TASK_URL_ENV).ok().filter(|url| !url.is_empty());
    if high.is_none() && task.is_none() {
        *card.write() = unavailable("queue urls are unset");
        return;
    }
    tokio::spawn(async move {
        loop {
            let next = poll(&http, high.as_deref(), task.as_deref()).await;
            *card.write() = next;
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

async fn poll(http: &reqwest::Client, high: Option<&str>, task: Option<&str>) -> String {
    let (high, task) = tokio::join!(sample(http, high), sample(http, task));
    json!({
        "sampled_unix_s": unix_now(),
        "high": high,
        "task": task,
    })
    .to_string()
}

async fn sample(http: &reqwest::Client, url: Option<&str>) -> Value {
    let Some(url) = url else {
        return json!({"ok": false, "reason": "unset"});
    };
    let response = match http.get(url).timeout(Duration::from_millis(800)).send().await {
        Ok(response) if response.status().is_success() => response,
        Ok(response) => return json!({"ok": false, "reason": format!("http {}", response.status())}),
        Err(error) => return json!({"ok": false, "reason": error.to_string()}),
    };
    let body = match response.text().await {
        Ok(body) => body,
        Err(error) => return json!({"ok": false, "reason": error.to_string()}),
    };
    parse_sample(&body).unwrap_or_else(|| json!({"ok": false, "reason": "unrecognized metrics"}))
}

fn parse_sample(body: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(body) {
        let live = value.get("live")?;
        let slots = live.get("slots").and_then(Value::as_array);
        let held: Vec<i64> = slots
            .map(|slots| {
                slots
                    .iter()
                    .filter_map(|slot| slot.get("held_tokens").and_then(Value::as_i64))
                    .collect()
            })
            .unwrap_or_default();
        let decode = value
            .get("requests")
            .and_then(Value::as_array)
            .and_then(|rows| rows.first())
            .and_then(|row| row.get("decode_tok_s"))
            .and_then(Value::as_f64);
        return Some(json!({
            "ok": true,
            "waiting": live.get("waiting").and_then(Value::as_i64).unwrap_or(0),
            "running": live.get("running").and_then(Value::as_i64).unwrap_or(0),
            "parallel": live.get("parallel").and_then(Value::as_i64),
            "held_tokens": held,
            "decode_tok_s": decode,
        }));
    }
    let running = prometheus_gauge(body, "vllm:num_requests_running")?;
    let waiting = prometheus_gauge(body, "vllm:num_requests_waiting").unwrap_or(0.0);
    Some(json!({
        "ok": true,
        "waiting": waiting as i64,
        "running": running as i64,
    }))
}

fn prometheus_gauge(body: &str, name: &str) -> Option<f64> {
    body.lines().find_map(|line| {
        let line = line.trim();
        if line.starts_with('#') || !line.starts_with(name) {
            return None;
        }
        line.split_whitespace().last()?.parse().ok()
    })
}

fn unavailable(reason: &str) -> String {
    json!({
        "sampled_unix_s": unix_now(),
        "high": {"ok": false, "reason": reason},
        "task": {"ok": false, "reason": reason},
    })
    .to_string()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_strata_live_and_vllm_gauges() {
        let strata = r#"{"live":{"waiting":2,"running":1,"parallel":4,"slots":[{"held_tokens":100},{"held_tokens":0}]},"requests":[{"decode_tok_s":35.1}]}"#;
        let parsed = parse_sample(strata).unwrap();
        assert_eq!(parsed["waiting"], 2);
        assert_eq!(parsed["held_tokens"][0], 100);
        assert_eq!(parsed["decode_tok_s"], 35.1);
        let prom = "vllm:num_requests_running{engine=\"0\"} 3.0\nvllm:num_requests_waiting{engine=\"0\"} 1.0\n";
        let parsed = parse_sample(prom).unwrap();
        assert_eq!(parsed["running"], 3);
        assert_eq!(parsed["waiting"], 1);
    }
}
