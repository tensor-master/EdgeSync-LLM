package cache

import (
	"testing"
	"time"
)

func makeOrchTestFragment(t *testing.T, tokens int) *KVFragment {
	t.Helper()
	return makeSparsifyTestFragment(t, tokens)
}

func freshDeviceState(avail, threshold int64) DeviceState {
	return DeviceState{
		AvailableMemoryBytes: avail,
		LowMemoryThreshold:   threshold,
		SampledAt:            time.Now(),
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// DeviceState tests
// ─────────────────────────────────────────────────────────────────────────────

func TestDeviceState_MemoryMarginBytes(t *testing.T) {
	d := DeviceState{AvailableMemoryBytes: 500 * 1024 * 1024, LowMemoryThreshold: 100 * 1024 * 1024}
	want := int64(400 * 1024 * 1024)
	if got := d.MemoryMarginBytes(); got != want {
		t.Errorf("MemoryMarginBytes: want %d, got %d", want, got)
	}
}

func TestDeviceState_IsStale(t *testing.T) {
	fresh := DeviceState{SampledAt: time.Now()}
	if fresh.IsStale(time.Minute) {
		t.Error("freshly sampled state should not be stale")
	}

	old := DeviceState{SampledAt: time.Now().Add(-time.Hour)}
	if !old.IsStale(time.Minute) {
		t.Error("hour-old state should be stale with a 1-minute max age")
	}

	zero := DeviceState{}
	if !zero.IsStale(time.Hour) {
		t.Error("zero-value SampledAt should always be considered stale")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Orchestrator.Decide — no candidate
// ─────────────────────────────────────────────────────────────────────────────

func TestDecide_NilCandidate_NoRemote_ReturnsRecompute(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())
	state := freshDeviceState(500*1024*1024, 100*1024*1024) // plenty of memory, irrelevant here

	d := o.Decide(nil, 256, state)
	if d.Strategy != StrategyRecompute {
		t.Errorf("want StrategyRecompute for nil candidate with no remote source, got %s", d.Strategy)
	}
}

func TestDecide_NilCandidate_FavorableRemote_ReturnsFetchRemote(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())
	state := freshDeviceState(500*1024*1024, 100*1024*1024)
	state.NetworkLatencyMs = 50 // well under RemoteFetchWorthwhileMs (200) and under recompute cost

	d := o.Decide(nil, 256, state)
	if d.Strategy != StrategyFetchRemote {
		t.Errorf("want StrategyFetchRemote for favorable network conditions, got %s", d.Strategy)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Orchestrator.Decide — full injection path
// ─────────────────────────────────────────────────────────────────────────────

func TestDecide_AmpleMemory_ReturnsFullInject(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())
	f := makeOrchTestFragment(t, 256)
	// Plenty of memory: way above MinMemoryMarginForFullInject (64MB) and
	// above the fragment's own size.
	state := freshDeviceState(1024*1024*1024, 50*1024*1024) // ~974MB margin

	d := o.Decide(f, 256, state)
	if d.Strategy != StrategyFullInject {
		t.Errorf("want StrategyFullInject under ample memory, got %s (%s)", d.Strategy, d.Reason)
	}
	if d.EstimatedFullInjectBytes != f.SizeBytes() {
		t.Errorf("EstimatedFullInjectBytes mismatch: want %d, got %d", f.SizeBytes(), d.EstimatedFullInjectBytes)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Orchestrator.Decide — sparsified fallback path
// ─────────────────────────────────────────────────────────────────────────────

func TestDecide_ModeratePressure_ReturnsSparsifiedInject(t *testing.T) {
	cfg := DefaultOrchestratorConfig()
	o := NewOrchestrator(cfg)
	f := makeOrchTestFragment(t, 256)

	fullSize := int64(f.SizeBytes())
	sparseSize := int64(o.estimateSparsifiedBytes(f))

	// Construct a margin that sits between sparseSize and fullSize, and is
	// also below MinMemoryMarginForFullInject so the full-inject branch is
	// correctly excluded, but above MinMemoryMarginForSparsifiedInject.
	margin := sparseSize + 1024*1024 // just enough for sparsified, not full
	if margin >= fullSize {
		t.Skip("test fragment too small for this scenario on this build; sparsification didn't reduce enough")
	}
	available := margin + cfg.MinMemoryMarginForSparsifiedInject // doesn't matter, margin is what we compute from threshold
	state := DeviceState{
		AvailableMemoryBytes: available,
		LowMemoryThreshold:   cfg.MinMemoryMarginForSparsifiedInject,
		SampledAt:            time.Now(),
	}
	// Recompute the actual margin the orchestrator will see
	actualMargin := state.MemoryMarginBytes()
	if actualMargin < cfg.MinMemoryMarginForSparsifiedInject || actualMargin > cfg.MinMemoryMarginForFullInject {
		t.Skipf("constructed margin %d not in the intended sparsified-only band [%d, %d]",
			actualMargin, cfg.MinMemoryMarginForSparsifiedInject, cfg.MinMemoryMarginForFullInject)
	}

	d := o.Decide(f, 256, state)
	if d.Strategy != StrategySparsifiedInject {
		t.Errorf("want StrategySparsifiedInject under moderate memory pressure, got %s (%s)", d.Strategy, d.Reason)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Orchestrator.Decide — recompute fallback under severe pressure
// ─────────────────────────────────────────────────────────────────────────────

func TestDecide_SeverePressure_NoRemote_ReturnsRecompute(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())
	f := makeOrchTestFragment(t, 256)
	// Available memory below the low-memory threshold itself: negative margin.
	state := freshDeviceState(10*1024*1024, 50*1024*1024)

	d := o.Decide(f, 256, state)
	if d.Strategy != StrategyRecompute {
		t.Errorf("want StrategyRecompute under severe memory pressure with no remote source, got %s (%s)",
			d.Strategy, d.Reason)
	}
}

func TestDecide_SeverePressure_FavorableRemote_ReturnsFetchRemote(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())
	f := makeOrchTestFragment(t, 256)
	state := freshDeviceState(10*1024*1024, 50*1024*1024)
	state.NetworkLatencyMs = 30 // very fast remote, favorable vs recompute

	d := o.Decide(f, 256, state)
	if d.Strategy != StrategyFetchRemote {
		t.Errorf("want StrategyFetchRemote under severe memory pressure with favorable network, got %s (%s)",
			d.Strategy, d.Reason)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Thermal throttling effect on recompute estimate
// ─────────────────────────────────────────────────────────────────────────────

func TestEstimateRecomputeMs_ThermalThrottleIncreasesEstimate(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())

	normalState := DeviceState{ThermalThrottled: false}
	throttledState := DeviceState{ThermalThrottled: true, ThermalThrottleFactor: 0.84} // Dimensity 6300 figure

	normalMs := o.estimateRecomputeMs(256, normalState)
	throttledMs := o.estimateRecomputeMs(256, throttledState)

	if throttledMs <= normalMs {
		t.Errorf("throttled estimate (%.1fms) should exceed normal estimate (%.1fms)", throttledMs, normalMs)
	}
}

func TestEstimateRecomputeMs_IgnoresInvalidThrottleFactor(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())

	// ThermalThrottleFactor = 0 or > 1 should not be applied (guarded in implementation)
	state := DeviceState{ThermalThrottled: true, ThermalThrottleFactor: 0}
	ms := o.estimateRecomputeMs(256, state)
	baseline := o.estimateRecomputeMs(256, DeviceState{})

	if ms != baseline {
		t.Errorf("invalid throttle factor (0) should not change the estimate: got %.1f, want %.1f", ms, baseline)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// remoteFetchFavorable tests
// ─────────────────────────────────────────────────────────────────────────────

func TestRemoteFetchFavorable_NoNetworkConfigured(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())
	state := DeviceState{NetworkLatencyMs: 0}
	if o.remoteFetchFavorable(state, 1000) {
		t.Error("remoteFetchFavorable should be false when NetworkLatencyMs is 0 (no remote source)")
	}
}

func TestRemoteFetchFavorable_SlowerThanRecompute(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())
	state := DeviceState{NetworkLatencyMs: 500} // slower than the recompute estimate below
	if o.remoteFetchFavorable(state, 100) {
		t.Error("remoteFetchFavorable should be false when network latency exceeds recompute cost")
	}
}

func TestRemoteFetchFavorable_ExceedsWorthwhileThreshold(t *testing.T) {
	cfg := DefaultOrchestratorConfig() // RemoteFetchWorthwhileMs = 200
	o := NewOrchestrator(cfg)
	state := DeviceState{NetworkLatencyMs: 250} // exceeds threshold even if "fast" vs a huge recompute
	if o.remoteFetchFavorable(state, 10000) {
		t.Error("remoteFetchFavorable should respect RemoteFetchWorthwhileMs as an absolute ceiling")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// estimateSparsifiedBytes tests
// ─────────────────────────────────────────────────────────────────────────────

func TestEstimateSparsifiedBytes_SmallerThanFull(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())
	f := makeOrchTestFragment(t, 256)

	sparse := o.estimateSparsifiedBytes(f)
	if sparse >= f.SizeBytes() {
		t.Errorf("sparsified estimate (%d) should be smaller than full size (%d)", sparse, f.SizeBytes())
	}
}

func TestEstimateSparsifiedBytes_ZeroTokenSpanFallsBackToFullSize(t *testing.T) {
	o := NewOrchestrator(DefaultOrchestratorConfig())
	f := makeOrchTestFragment(t, 256)
	f.TokenEnd = f.TokenStart // degenerate zero span

	got := o.estimateSparsifiedBytes(f)
	if got != f.SizeBytes() {
		t.Errorf("zero-span fragment should fall back to full SizeBytes(): want %d, got %d", f.SizeBytes(), got)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// InjectionStrategy.String tests
// ─────────────────────────────────────────────────────────────────────────────

func TestInjectionStrategy_String(t *testing.T) {
	cases := map[InjectionStrategy]string{
		StrategyFullInject:       "FULL_INJECT",
		StrategySparsifiedInject: "SPARSIFIED_INJECT",
		StrategyRecompute:        "RECOMPUTE",
		StrategyFetchRemote:      "FETCH_REMOTE",
		InjectionStrategy(99):    "UNKNOWN",
	}
	for strategy, want := range cases {
		if got := strategy.String(); got != want {
			t.Errorf("String() for %d: want %q, got %q", strategy, want, got)
		}
	}
}
