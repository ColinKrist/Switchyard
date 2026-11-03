// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-target in-flight request tracking for load-aware candidate selection.
//!
//! [`ModelLoadTracker`] maintains a per-target counter of in-flight answer calls.
//! The client loop calls [`try_acquire`](ModelLoadTracker::try_acquire) before
//! dispatching to a candidate and [`release`](ModelLoadTracker::release) after the
//! call completes. When a target is at its [`max_in_flight`](ModelLoadTracker::new)
//! capacity, the caller skips to the next candidate.

use std::collections::HashMap;

use parking_lot::Mutex;
use switchyard_protocol::ModelId;

/// Tracks in-flight answer-call slots per target.
///
/// Shared through `Arc` by every concurrent request on a route.
pub struct ModelLoadTracker {
    /// In-flight counts per target. Absent keys default to zero.
    counts: Mutex<HashMap<ModelId, u32>>,
    /// Per-target capacity. Targets absent from this map are unlimited.
    limits: HashMap<ModelId, u32>,
}

impl ModelLoadTracker {
    /// Creates a tracker with the given per-target capacity limits.
    pub fn new(limits: Vec<(ModelId, u32)>) -> Self {
        let limits = limits.into_iter().collect();
        Self {
            counts: Mutex::new(HashMap::new()),
            limits,
        }
    }
    /// Atomically orders candidates load-aware and acquires a slot on the chosen first candidate.
    ///
    /// When `models[0]` (the algorithm's preferred selection) is under its capacity limit,
    /// it is preserved as the first candidate and any fallbacks are sorted least-loaded first.
    ///
    /// When `models[0]` is at or above capacity, traffic is directed to the least-loaded
    /// candidate: candidates under capacity come first (ordered lowest in-flight first),
    /// followed by candidates at/above capacity (ordered lowest in-flight first).
    ///
    /// The chosen first candidate's in-flight count is atomically incremented.
    /// The caller MUST call [`release`](Self::release) when the call completes.
    pub fn order_and_acquire(&self, models: &[ModelId]) -> Vec<ModelId> {
        let mut counts = self.counts.lock();
        if models.is_empty() {
            return Vec::new();
        }

        let preferred = &models[0];
        let preferred_count = counts.get(preferred).copied().unwrap_or(0);
        let preferred_limit = self.limits.get(preferred).copied();
        let preferred_under_cap = preferred_limit.map_or(true, |limit| preferred_count < limit);

        let mut ordered = models.to_vec();
        if preferred_under_cap {
            if ordered.len() > 1 {
                ordered[1..].sort_by_key(|target| counts.get(target).copied().unwrap_or(0));
            }
        } else {
            ordered.sort_by_key(|target| {
                let count = counts.get(target).copied().unwrap_or(0);
                let at_cap = self.limits.get(target).is_some_and(|&limit| count >= limit);
                (at_cap, count)
            });
        }

        if let Some(first) = ordered.first() {
            *counts.entry(first.clone()).or_insert(0) += 1;
        }

        ordered
    }

    /// Acquire a slot on `target` directly.
    pub fn acquire(&self, target: &ModelId) {
        let mut counts = self.counts.lock();
        *counts.entry(target.clone()).or_insert(0) += 1;
    }

    /// Atomically check-and-acquire a slot for `target`.
    ///
    /// Returns `true` when a slot was acquired (the caller **must** call
    /// [`release`](Self::release) when the call completes). Returns `false`
    /// when `target` is at its configured capacity.
    pub fn try_acquire(&self, target: &ModelId) -> bool {
        let mut counts = self.counts.lock();
        let at_capacity = self
            .limits
            .get(target)
            .is_some_and(|&limit| counts.get(target).copied().unwrap_or(0) >= limit);
        if at_capacity {
            return false;
        }
        *counts.entry(target.clone()).or_insert(0) += 1;
        true
    }

    /// Release a previously acquired slot for `target`.
    ///
    /// Calling this without a matching acquire is a programming error; the count
    /// saturates at zero.
    pub fn release(&self, target: &ModelId) {
        let mut counts = self.counts.lock();
        if let Some(count) = counts.get_mut(target) {
            *count = count.saturating_sub(1);
        }
    }

    /// Current in-flight count for `target`.
    pub fn in_flight(&self, target: &ModelId) -> u32 {
        self.counts.lock().get(target).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker(limit: u32) -> (ModelLoadTracker, ModelId) {
        let target = ModelId::from("test-target");
        let tracker = ModelLoadTracker::new(vec![(target.clone(), limit)]);
        (tracker, target)
    }

    #[test]
    fn acquires_until_at_capacity() {
        let (tracker, target) = tracker(3);
        assert!(tracker.try_acquire(&target));
        assert!(tracker.try_acquire(&target));
        assert!(tracker.try_acquire(&target));
        assert!(!tracker.try_acquire(&target));
        assert_eq!(tracker.in_flight(&target), 3);
    }

    #[test]
    fn release_frees_a_slot() {
        let (tracker, target) = tracker(1);
        assert!(tracker.try_acquire(&target));
        assert!(!tracker.try_acquire(&target));
        tracker.release(&target);
        assert!(tracker.try_acquire(&target));
    }

    #[test]
    fn unlimited_target_never_refuses() {
        let target = ModelId::from("unlimited");
        let tracker = ModelLoadTracker::new(Vec::new());
        for _ in 0..100 {
            assert!(tracker.try_acquire(&target));
        }
        assert_eq!(tracker.in_flight(&target), 100);
    }

    #[test]
    fn release_without_acquire_saturates() {
        let (tracker, target) = tracker(1);
        tracker.release(&target);
        assert_eq!(tracker.in_flight(&target), 0);
        assert!(tracker.try_acquire(&target));
    }

    #[test]
    fn independent_targets_do_not_share_capacity() {
        let a = ModelId::from("a");
        let b = ModelId::from("b");
        let tracker = ModelLoadTracker::new(vec![(a.clone(), 1), (b.clone(), 1)]);
        assert!(tracker.try_acquire(&a));
        assert!(tracker.try_acquire(&b));
        assert!(!tracker.try_acquire(&a));
        assert!(!tracker.try_acquire(&b));
    }

    #[test]
    fn order_and_acquire_preserves_preferred_when_under_capacity() {
        let a = ModelId::from("a");
        let b = ModelId::from("b");
        let tracker = ModelLoadTracker::new(vec![(a.clone(), 2), (b.clone(), 2)]);
        let ordered = tracker.order_and_acquire(&[a.clone(), b.clone()]);
        assert_eq!(ordered, vec![a.clone(), b.clone()]);
        assert_eq!(tracker.in_flight(&a), 1);
        assert_eq!(tracker.in_flight(&b), 0);
    }

    #[test]
    fn order_and_acquire_directs_to_least_loaded_when_preferred_at_capacity() {
        let a = ModelId::from("a");
        let b = ModelId::from("b");
        let tracker = ModelLoadTracker::new(vec![(a.clone(), 1), (b.clone(), 2)]);
        // Pre-fill 'a' to its limit of 1
        assert!(tracker.try_acquire(&a));

        // 'a' is preferred, but at capacity; traffic goes to 'b' (least loaded)
        let ordered = tracker.order_and_acquire(&[a.clone(), b.clone()]);
        assert_eq!(ordered, vec![b.clone(), a.clone()]);
        assert_eq!(tracker.in_flight(&a), 1);
        assert_eq!(tracker.in_flight(&b), 1);
    }

    #[test]
    fn order_and_acquire_directs_to_least_loaded_when_all_at_capacity() {
        let a = ModelId::from("a");
        let b = ModelId::from("b");
        let tracker = ModelLoadTracker::new(vec![(a.clone(), 1), (b.clone(), 1)]);
        tracker.acquire(&a);
        tracker.acquire(&a); // 'a' has 2 in flight (cap 1)
        tracker.acquire(&b); // 'b' has 1 in flight (cap 1)

        // Both over capacity, 'b' has lower in-flight count (1 < 2)
        let ordered = tracker.order_and_acquire(&[a.clone(), b.clone()]);
        assert_eq!(ordered, vec![b.clone(), a.clone()]);
        assert_eq!(tracker.in_flight(&b), 2);
        assert_eq!(tracker.in_flight(&a), 2);
    }
}
