// Package cache — Stateful-inference orchestrator: decides at request time
// whether to inject a cached fragment, inject a sparsified fallback, or fall
// through to full recompute, based on current RAM pressure and lookup latency
// budget.
//
// WHAT THIS IS, AND WHAT IT ISN'T
// ─────────────────────────────────
// This is a decision layer on top of the existing pieces (FragmentStore,
// Compactor, HNSW lookup in the adapter package) — it does not replace any
// of them. The existing DifferentialEngine (cache/differential.go) already
// implements the EXACT/PARTIAL/MISS branching based on similarity score.
// What's new here is a SECOND axis of decision-making, orthogonal to
// similarity: given that a usable fragment exists, is it actually worth
// using right now, given current device conditions?
//
// Two fragments can have identical similarity scores and lead to different
// decisions:
//   - A device under heavy memory pressure (background apps competing for
//     RAM) may not have room to inject a 24MB exact-layer fragment, but CAN
//     afford a 3MB sparsified version of the same fragment (cache/sparsify.go).
//   - A device with plenty of RAM but currently CPU-throttled (thermal
//     throttling, as documented for the Dimensity 6300 after sustained load)
//     may prefer to inject the full fragment even at higher memory cost,
//     since recompute is comparatively more expensive when the CPU is slow.
//
// This module formalizes that tradeoff as an explicit, testable decision
// function rather than leaving it as an implicit assumption baked into
// fixed thresholds.
//
// WHAT THIS MODULE DOES NOT DO
// ──────────────────────────────
// It does not measure RAM or CPU itself — that's platform-specific (see
// monitor/energy_android.go for the existing Android battery/power signal
// reader; a RAM/thermal reader would live alongside it). This module accepts
// a DeviceState snapshot as input and is therefore fully unit-testable
// without any Android dependency.
package cache

import (
	"fmt"
	"time"
)

// ─────────────────────────────────────────────────────────────────────────────
// DeviceState — a snapshot of resource availability, supplied by the caller
// ─────────────────────────────────────────────────────────────────────────────

// DeviceState describes the current resource conditions the orchestrator
// uses to make its decision. Populate this from platform-specific signals
// (Android: ActivityManager.MemoryInfo, /sys/class/thermal, etc.) once per
// request or on a periodic timer — it does not need to be sampled on every
// single cache lookup if that's too expensive.
type DeviceState struct {
	// AvailableMemoryBytes is free/reclaimable RAM as reported by the OS.
	// On Android: ActivityManager.MemoryInfo.availMem.
	AvailableMemoryBytes int64

	// LowMemoryThreshold is the OS's own low-memory cutoff, if known.
	// On Android: ActivityManager.MemoryInfo.threshold. Used to compute
	// a margin rather than relying on an absolute byte count, since "enough
	// RAM" varies enormously across device tiers.
	LowMemoryThreshold int64

	// ThermalThrottled indicates the CPU is currently running below its
	// nominal clock due to thermal limits (e.g. Dimensity 6300 dropping to
	// ~84% of peak after sustained inference — see monitor package). When
	// true, recompute cost estimates should be scaled up accordingly.
	ThermalThrottled bool

	// ThermalThrottleFactor is the estimated current-to-peak performance
	// ratio when ThermalThrottled is true (e.g. 0.84 for the documented
	// Dimensity 6300 sustained-load figure). Ignored if ThermalThrottled is false.
	ThermalThrottleFactor float64

	// NetworkLatencyMs is round-trip latency to a remote fragment source, if
	// the deployment supports fetching fragments from a server/peer (see the
	// differential-sync direction referenced in security/signing.go's
	// cache-poisoning rationale). Zero if no remote source is configured —
	// the orchestrator then only considers local store + recompute.
	NetworkLatencyMs float64

	// SampledAt records when this snapshot was taken, so callers can detect
	// stale state and decide whether to re-sample before trusting the decision.
	SampledAt time.Time
}

// IsStale reports whether this DeviceState snapshot is older than maxAge.
func (d DeviceState) IsStale(maxAge time.Duration) bool {
	if d.SampledAt.IsZero() {
		return true
	}
	return time.Since(d.SampledAt) > maxAge
}

// MemoryMarginBytes returns how far above the low-memory threshold the
// device currently sits. Negative means the device is already in (or below)
// its own low-memory zone.
func (d DeviceState) MemoryMarginBytes() int64 {
	return d.AvailableMemoryBytes - d.LowMemoryThreshold
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision type
// ─────────────────────────────────────────────────────────────────────────────

// InjectionStrategy is the orchestrator's decision for how to handle a
// candidate fragment match.
type InjectionStrategy int

const (
	// StrategyFullInject: inject the fragment as-is (all captured layers,
	// all tokens). Used when memory and CPU conditions both favor it.
	StrategyFullInject InjectionStrategy = iota

	// StrategySparsifiedInject: sparsify the fragment (cache/sparsify.go)
	// before injecting, trading some quality for a smaller memory footprint.
	// Used under memory pressure when a smaller version would fit.
	StrategySparsifiedInject

	// StrategyRecompute: skip the cached fragment entirely and run a full
	// prefill. Used when neither full nor sparsified injection is favorable
	// — e.g. severe memory pressure where even the sparsified fragment
	// wouldn't safely fit, or the fragment is judged too stale/low-confidence
	// to be worth the injection overhead relative to just recomputing.
	StrategyRecompute

	// StrategyFetchRemote: fetch the fragment from a configured remote source
	// rather than using the local copy or recomputing. Only ever returned
	// when NetworkLatencyMs indicates a remote fetch would be faster than
	// local recompute AND a remote source is actually configured by the caller
	// (this module doesn't perform the fetch — it only recommends it).
	StrategyFetchRemote
)

func (s InjectionStrategy) String() string {
	switch s {
	case StrategyFullInject:
		return "FULL_INJECT"
	case StrategySparsifiedInject:
		return "SPARSIFIED_INJECT"
	case StrategyRecompute:
		return "RECOMPUTE"
	case StrategyFetchRemote:
		return "FETCH_REMOTE"
	default:
		return "UNKNOWN"
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Decision
// ─────────────────────────────────────────────────────────────────────────────

// Decision is the orchestrator's recommendation, with the reasoning exposed
// so callers (and tests) can audit why a particular strategy was chosen —
// this is deliberately not a black box, given the stakes of silently
// degrading inference quality under sparsification.
type Decision struct {
	Strategy InjectionStrategy
	Reason   string

	// EstimatedFullInjectBytes is the memory cost of StrategyFullInject for
	// this fragment, for logging/telemetry.
	EstimatedFullInjectBytes int

	// EstimatedSparsifiedBytes is the memory cost of StrategySparsifiedInject,
	// using the orchestrator's configured SparsificationConfig.
	EstimatedSparsifiedBytes int

	// EstimatedRecomputeMs is the projected latency of a full prefill under
	// current (possibly throttled) device conditions.
	EstimatedRecomputeMs float64
}

// ─────────────────────────────────────────────────────────────────────────────
// Orchestrator
// ─────────────────────────────────────────────────────────────────────────────

// OrchestratorConfig tunes the decision thresholds. Defaults are conservative
// (prefer correctness/full-quality over aggressive sparsification) and are
// meant to be tuned per-deployment once real device telemetry is available —
// see the project's stated principle of validating cost models against
// measured hardware rather than assumed constants (cf. benchmark/runner.go's
// Cortex-A55 timing model derived from published llama.cpp numbers).
type OrchestratorConfig struct {
	// MinMemoryMarginForFullInject: if DeviceState.MemoryMarginBytes() is
	// below this, full injection is not attempted even if the fragment would
	// technically fit, to leave headroom for the rest of the app.
	MinMemoryMarginForFullInject int64

	// MinMemoryMarginForSparsifiedInject: floor for attempting the sparsified
	// fallback. Below this, the orchestrator recommends recompute instead of
	// risking an OOM from even the reduced fragment.
	MinMemoryMarginForSparsifiedInject int64

	// SparsificationConfig is used to estimate the sparsified fragment size
	// when evaluating StrategySparsifiedInject.
	SparsificationConfig SparsificationConfig

	// MsPerTokenPrefillNominal: baseline prefill cost per token, used to
	// estimate recompute latency. Should match the deployment's actual
	// measured hardware constant (see benchmark/runner.go's
	// MsPerTokenPrefill for the Cortex-A55 reference value used elsewhere
	// in this project — pass the same constant here for consistency, or a
	// device-specific one if available).
	MsPerTokenPrefillNominal float64

	// RemoteFetchWorthwhileMs: if NetworkLatencyMs plus an assumed transfer
	// time for the fragment is below this relative to EstimatedRecomputeMs,
	// StrategyFetchRemote becomes eligible. This is intentionally a coarse
	// heuristic — real deployments should replace it with measured transfer
	// throughput once a remote fragment source exists.
	RemoteFetchWorthwhileMs float64
}

// DefaultOrchestratorConfig returns conservative defaults: prefer full
// injection whenever there's reasonable headroom, fall back to sparsified
// only under real pressure, and use the same MsPerTokenPrefill constant the
// rest of the project already validates against (benchmark/runner.go).
func DefaultOrchestratorConfig() OrchestratorConfig {
	return OrchestratorConfig{
		MinMemoryMarginForFullInject:       64 * 1024 * 1024, // 64MB headroom
		MinMemoryMarginForSparsifiedInject: 16 * 1024 * 1024, // 16MB headroom
		SparsificationConfig:               DefaultSparsificationConfig(),
		MsPerTokenPrefillNominal:           6.8, // matches benchmark/runner.go MsPerTokenPrefill
		RemoteFetchWorthwhileMs:            200,
	}
}

// Orchestrator makes the recompute-vs-inject decision for a candidate fragment.
type Orchestrator struct {
	cfg OrchestratorConfig
}

// NewOrchestrator creates an Orchestrator with the given configuration.
func NewOrchestrator(cfg OrchestratorConfig) *Orchestrator {
	return &Orchestrator{cfg: cfg}
}

// Decide evaluates whether and how to use candidate given the current
// device state and the token count that would need to be recomputed on a
// MISS (used to estimate the recompute baseline this decision is being
// weighed against).
//
// candidate may be nil (no fragment match was found by the HNSW lookup) —
// in that case Decide always returns StrategyRecompute (or StrategyFetchRemote
// if a remote source looks favorable), since there is nothing local to inject.
func (o *Orchestrator) Decide(candidate *KVFragment, missTokenCount int, state DeviceState) Decision {
	recomputeMs := o.estimateRecomputeMs(missTokenCount, state)

	if candidate == nil {
		if o.remoteFetchFavorable(state, recomputeMs) {
			return Decision{
				Strategy:             StrategyFetchRemote,
				Reason:               "no local fragment match; remote fetch estimated faster than recompute",
				EstimatedRecomputeMs: recomputeMs,
			}
		}
		return Decision{
			Strategy:             StrategyRecompute,
			Reason:               "no candidate fragment available",
			EstimatedRecomputeMs: recomputeMs,
		}
	}

	fullBytes := candidate.SizeBytes()
	sparsifiedBytes := o.estimateSparsifiedBytes(candidate)
	margin := state.MemoryMarginBytes()

	decision := Decision{
		EstimatedFullInjectBytes: fullBytes,
		EstimatedSparsifiedBytes: sparsifiedBytes,
		EstimatedRecomputeMs:     recomputeMs,
	}

	switch {
	case margin >= o.cfg.MinMemoryMarginForFullInject && int64(fullBytes) <= margin:
		decision.Strategy = StrategyFullInject
		decision.Reason = fmt.Sprintf(
			"memory margin %d bytes covers full fragment (%d bytes) with headroom to spare (min required %d)",
			margin, fullBytes, o.cfg.MinMemoryMarginForFullInject,
		)

	case margin >= o.cfg.MinMemoryMarginForSparsifiedInject && int64(sparsifiedBytes) <= margin:
		decision.Strategy = StrategySparsifiedInject
		decision.Reason = fmt.Sprintf(
			"memory margin %d bytes insufficient for full fragment (%d bytes) but covers sparsified version (%d bytes)",
			margin, fullBytes, sparsifiedBytes,
		)

	default:
		if o.remoteFetchFavorable(state, recomputeMs) {
			decision.Strategy = StrategyFetchRemote
			decision.Reason = fmt.Sprintf(
				"memory margin %d bytes insufficient for even sparsified fragment (%d bytes); remote fetch favorable over recompute",
				margin, sparsifiedBytes,
			)
		} else {
			decision.Strategy = StrategyRecompute
			decision.Reason = fmt.Sprintf(
				"memory margin %d bytes insufficient for sparsified fragment (%d bytes, min required %d); falling back to recompute",
				margin, sparsifiedBytes, o.cfg.MinMemoryMarginForSparsifiedInject,
			)
		}
	}

	return decision
}

// estimateRecomputeMs projects prefill latency for missTokenCount tokens
// under current device conditions, scaling up if the device is thermally
// throttled.
func (o *Orchestrator) estimateRecomputeMs(missTokenCount int, state DeviceState) float64 {
	base := float64(missTokenCount) * o.cfg.MsPerTokenPrefillNominal
	if state.ThermalThrottled && state.ThermalThrottleFactor > 0 && state.ThermalThrottleFactor < 1.0 {
		// Lower throttle factor = slower CPU = higher latency.
		// E.g. factor 0.84 (Dimensity 6300 sustained-load figure) → cost / 0.84.
		base = base / state.ThermalThrottleFactor
	}
	return base
}

// estimateSparsifiedBytes projects the memory footprint of the sparsified
// version of candidate without actually performing the sparsification
// (avoids the copy cost just for a sizing estimate).
func (o *Orchestrator) estimateSparsifiedBytes(candidate *KVFragment) int {
	tokenSpan := candidate.TokenSpan()
	if tokenSpan <= 0 {
		return candidate.SizeBytes()
	}

	pivotIndices, _ := selectPivotIndices(tokenSpan, o.cfg.SparsificationConfig)
	keptCount := len(pivotIndices)
	if keptCount >= tokenSpan {
		return candidate.SizeBytes() // sparsification wouldn't reduce anything
	}

	ratio := float64(keptCount) / float64(tokenSpan)
	return int(float64(candidate.SizeBytes()) * ratio)
}

// remoteFetchFavorable reports whether fetching from a remote source looks
// faster than the projected recompute cost. This is a coarse heuristic:
// it does not account for fragment transfer size/bandwidth, since this
// project's current scope is single-device (see README's "What's not solved
// yet" — distributed fragment sharing is explicitly listed as future work).
// This function exists so the orchestrator's decision space is forward-
// compatible with that future work without requiring a rewrite when it lands.
func (o *Orchestrator) remoteFetchFavorable(state DeviceState, recomputeMs float64) bool {
	if state.NetworkLatencyMs <= 0 {
		return false // no remote source configured
	}
	return state.NetworkLatencyMs < o.cfg.RemoteFetchWorthwhileMs &&
		state.NetworkLatencyMs < recomputeMs
}
