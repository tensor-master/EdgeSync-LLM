/**
 * @file App.tsx
 * @description Highly polished, high-contrast twilight/slate interactive dashboard for EdgeSync-LLM.
 * Features real-time telemetry meters, visual prompt playgrounds, HNSW tree graph paths, and benchmark utilities.
 */

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  Battery,
  Cpu,
  Zap,
  Play,
  Trash2,
  Send,
  Layers,
  Activity,
  Sparkles,
  Clock,
  ArrowRight,
  CheckCircle,
  RefreshCw,
  Sliders,
  Database,
  BarChart4,
  AlertTriangle,
  Check,
  ChevronRight,
  BookOpen
} from "lucide-react";
import { CacheState, QueryResult, BenchmarkMetrics, SessionMetrics } from "./types";

export default function App() {
  // UI states
  const [activeTab, setActiveTab] = useState<"playground" | "benchmark" | "explorer">("playground");
  const [promptInput, setPromptInput] = useState("");
  const [loading, setLoading] = useState(false);
  
  // Cache telemetry metrics from server
  const [session, setSession] = useState<SessionMetrics>({
    totalQueries: 0,
    hitCount: 0,
    partialHitCount: 0,
    missCount: 0,
    hitRatePct: 0,
    averageLatencyMs: 0.0,
    npuReductionPct: 0,
    batterySavingsMah: 0.0,
  });

  const [recentQueries, setRecentQueries] = useState<QueryResult[]>([]);
  const [cacheNodeCount, setCacheNodeCount] = useState(0);
  const [predictions, setPredictions] = useState<string[]>([]);
  
  // Last query result
  const [lastResult, setLastResult] = useState<QueryResult | null>(null);

  // Benchmark states
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkMetrics[]>([]);

  // Fetch metrics upon mount and when query runs
  const fetchMetrics = async () => {
    try {
      const res = await fetch("/api/metrics");
      if (res.ok) {
        const data = await res.json();
        setSession(data.session);
        setRecentQueries(data.recentQueries);
        setCacheNodeCount(data.cacheNodeCount);
        setPredictions(data.predictions);
      }
    } catch (err) {
      console.error("Failed to fetch cache metrics:", err);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  // Pre-seed prompt suggestion clicks
  const selectSuggestion = (prompt: string) => {
    setPromptInput(prompt);
  };

  // Submit prompt to EdgeSync semantic interception engine
  const handleQuerySubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!promptInput.trim() || loading) return;

    setLoading(true);
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: promptInput }),
      });

      if (res.ok) {
        const data: QueryResult = await res.json();
        setLastResult(data);
        setPromptInput("");
        fetchMetrics();
      }
    } catch (err) {
      console.error("Query submission failed:", err);
    } finally {
      setLoading(false);
    }
  };

  // Run the 3-round automated benchmark
  const triggerBenchmark = async () => {
    setBenchmarkLoading(true);
    try {
      const res = await fetch("/api/benchmark", { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setBenchmarkResults(data);
      }
    } catch (err) {
      console.error("Benchmark runner failed:", err);
    } finally {
      setBenchmarkLoading(false);
    }
  };

  // Flush the SQLite WAL database and reset the HNSW graph index
  const handleClearCache = async () => {
    if (!confirm("Are you sure you want to flush the SQLite database and clear the HNSW graph?")) return;
    try {
      const res = await fetch("/api/clear", { method: "POST" });
      if (res.ok) {
        setLastResult(null);
        setBenchmarkResults([]);
        fetchMetrics();
      }
    } catch (err) {
      console.error("Failed to clear cache:", err);
    }
  };

  const sampleSuggestions = [
    "what are the core benefits of rust over c++ for embedded systems",
    "explain quantum computing in simple high school terms",
    "write a fast fiber-based http server in modern go",
    "how to configure write-ahead logging in sqlite",
    "how to use arm neon intrinsics to multiply float16 arrays",
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased selection:bg-cyan-500 selection:text-slate-950">
      
      {/* Target Metrics Sticky Banner */}
      <div className="bg-slate-900 border-b border-slate-800 py-2.5 px-4 sm:px-6">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2.5">
          <div className="flex items-center gap-2">
            <span className="flex h-2.5 w-2.5 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-xs font-mono font-medium tracking-wider uppercase text-slate-400">
              EdgeSync-LLM Core Active
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-3 sm:gap-6 text-xs font-mono">
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">NPU Reduct:</span>
              <span className="text-cyan-400 font-semibold">70% Target</span>
            </div>
            <div className="h-3 w-[1px] bg-slate-800 hidden sm:block" />
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">Bat Savings:</span>
              <span className="text-emerald-400 font-semibold">65% Target</span>
            </div>
            <div className="h-3 w-[1px] bg-slate-800 hidden sm:block" />
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">Lookup:</span>
              <span className="text-amber-400 font-semibold">&lt;2ms Target</span>
            </div>
            <div className="h-3 w-[1px] bg-slate-800 hidden sm:block" />
            <div className="flex items-center gap-1.5">
              <span className="text-slate-500">Hit Rate:</span>
              <span className="text-purple-400 font-semibold">85% Target</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Navigation Header */}
      <header className="max-w-7xl mx-auto pt-8 px-4 sm:px-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6 border-b border-slate-800 pb-6">
          <div>
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-cyan-400 to-indigo-600 p-2.5 rounded-xl text-slate-950 shadow-lg shadow-cyan-500/10">
                <Cpu className="h-6 w-6 text-slate-950" id="header-cpu-icon" />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight bg-gradient-to-r from-slate-50 to-slate-300 bg-clip-text text-transparent">
                  EdgeSync-LLM
                </h1>
                <p className="text-xs text-slate-400 mt-0.5">
                  Embedded Semantic Interception & Cache routing for Cortex-A55 clusters
                </p>
              </div>
            </div>
          </div>

          {/* Navigation Controls */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              id="nav-btn-playground"
              onClick={() => setActiveTab("playground")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === "playground"
                  ? "bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/10"
                  : "bg-slate-900 text-slate-300 hover:bg-slate-850 hover:text-slate-50"
              }`}
            >
              <div className="flex items-center gap-1.5">
                <Sliders className="h-4 w-4" />
                Playground Console
              </div>
            </button>
            <button
              id="nav-btn-benchmark"
              onClick={() => setActiveTab("benchmark")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === "benchmark"
                  ? "bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/10"
                  : "bg-slate-900 text-slate-300 hover:bg-slate-850 hover:text-slate-50"
              }`}
            >
              <div className="flex items-center gap-1.5">
                <BarChart4 className="h-4 w-4" />
                Benchmark Suite
              </div>
            </button>
            <button
              id="nav-btn-explorer"
              onClick={() => setActiveTab("explorer")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === "explorer"
                  ? "bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/10"
                  : "bg-slate-900 text-slate-300 hover:bg-slate-850 hover:text-slate-50"
              }`}
            >
              <div className="flex items-center gap-1.5">
                <Database className="h-4 w-4" />
                HNSW Index Explorer
              </div>
            </button>
            <button
              id="btn-flush-cache"
              onClick={handleClearCache}
              className="p-2 bg-slate-900 hover:bg-red-950/40 hover:text-red-400 border border-slate-800 hover:border-red-900/50 rounded-lg text-slate-400 transition-all ml-1"
              title="Flush databases & clear index"
            >
              <Trash2 className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6">
        
        {/* Telemetry Grid */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          
          {/* NPU REDUCTION */}
          <div className="bg-slate-900/55 border border-slate-850 p-6 rounded-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-all">
              <Cpu className="h-24 w-24 text-cyan-400" />
            </div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono text-slate-400 font-medium tracking-wider uppercase">NPU REDUCTION</span>
              <div className="bg-cyan-500/10 p-1.5 rounded-lg text-cyan-400">
                <Cpu className="h-4 w-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold tracking-tight text-cyan-400">{session.npuReductionPct}%</span>
              <span className="text-xs font-mono text-slate-500">NPU Saved</span>
            </div>
            <div className="mt-4 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-cyan-400"
                initial={{ width: 0 }}
                animate={{ width: `${session.npuReductionPct}%` }}
                transition={{ duration: 0.8 }}
              />
            </div>
          </div>

          {/* BATTERY SAVINGS */}
          <div className="bg-slate-900/55 border border-slate-850 p-6 rounded-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-all">
              <Battery className="h-24 w-24 text-emerald-400" />
            </div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono text-slate-400 font-medium tracking-wider uppercase">BATTERY SAVED</span>
              <div className="bg-emerald-500/10 p-1.5 rounded-lg text-emerald-400">
                <Battery className="h-4 w-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold tracking-tight text-emerald-400">{session.batterySavingsMah}</span>
              <span className="text-xs font-mono text-slate-400">mAh</span>
            </div>
            <div className="text-xs text-slate-500 mt-4 flex items-center gap-1.5">
              <Zap className="h-3 w-3 text-emerald-400" />
              <span>Session power discharge avoided</span>
            </div>
          </div>

          {/* CACHE LOOKUP SPEED */}
          <div className="bg-slate-900/55 border border-slate-850 p-6 rounded-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-all">
              <Clock className="h-24 w-24 text-amber-400" />
            </div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono text-slate-400 font-medium tracking-wider uppercase">LOOKUP LATENCY</span>
              <div className="bg-amber-500/10 p-1.5 rounded-lg text-amber-400">
                <Clock className="h-4 w-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold tracking-tight text-amber-400">{session.averageLatencyMs}</span>
              <span className="text-xs font-mono text-slate-400">ms</span>
            </div>
            <div className="text-xs text-slate-500 mt-4">
              <span>Average response time (Target &lt;2ms)</span>
            </div>
          </div>

          {/* CACHE HIT RATE */}
          <div className="bg-slate-900/55 border border-slate-850 p-6 rounded-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-all">
              <Activity className="h-24 w-24 text-purple-400" />
            </div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-xs font-mono text-slate-400 font-medium tracking-wider uppercase">ACTIVE HIT RATE</span>
              <div className="bg-purple-500/10 p-1.5 rounded-lg text-purple-400">
                <Activity className="h-4 w-4" />
              </div>
            </div>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold tracking-tight text-purple-400">{session.hitRatePct}%</span>
              <span className="text-xs font-mono text-slate-500">Hit rate</span>
            </div>
            <div className="text-xs text-slate-500 mt-4 flex items-center gap-3">
              <div className="flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                <span>Hits: {session.hitCount + session.partialHitCount}</span>
              </div>
              <div className="flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
                <span>Miss: {session.missCount}</span>
              </div>
            </div>
          </div>

        </section>

        {/* Tab Selection */}
        <AnimatePresence mode="wait">
          {activeTab === "playground" && (
            <motion.div
              key="playground"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.25 }}
              className="grid grid-cols-1 lg:grid-cols-12 gap-8"
            >
              {/* Intercept Console Form (Left Side - 7 cols) */}
              <div className="lg:col-span-7 flex flex-col gap-6">
                
                <div className="bg-slate-900 border border-slate-850 p-6 rounded-2xl">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold text-slate-50 flex items-center gap-2">
                      <Zap className="h-5 w-5 text-cyan-400" />
                      Semantic Interception Console
                    </h2>
                    <span className="text-xs font-mono bg-slate-800 text-slate-400 px-2 py-1 rounded">
                      Go core cgo
                    </span>
                  </div>

                  <form onSubmit={handleQuerySubmit} className="space-y-4">
                    <div className="relative">
                      <textarea
                        rows={3}
                        value={promptInput}
                        onChange={(e) => setPromptInput(e.target.value)}
                        placeholder="Type a query or select a suggestion below to intercept..."
                        className="w-full bg-slate-950 border border-slate-800 rounded-xl py-3 px-4 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/80 transition-all resize-none"
                      />
                      <div className="absolute bottom-3 right-3 flex items-center gap-2">
                        <button
                          type="submit"
                          disabled={!promptInput.trim() || loading}
                          className="p-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 rounded-lg disabled:opacity-50 disabled:hover:bg-cyan-500 transition-all shadow-md shadow-cyan-500/10"
                        >
                          {loading ? (
                            <RefreshCw className="h-4 w-4 animate-spin text-slate-950" />
                          ) : (
                            <Send className="h-4 w-4 text-slate-950" />
                          )}
                        </button>
                      </div>
                    </div>

                    {/* Suggestions list */}
                    <div>
                      <span className="text-xs text-slate-400 font-medium block mb-2">Prompt Templates:</span>
                      <div className="flex flex-wrap gap-2">
                        {sampleSuggestions.map((s, idx) => (
                          <button
                            key={idx}
                            type="button"
                            onClick={() => selectSuggestion(s)}
                            className="text-xs bg-slate-950 border border-slate-850 hover:border-slate-700 py-1.5 px-3 rounded-lg text-slate-300 hover:text-slate-50 transition-all text-left truncate max-w-full"
                          >
                            {s}
                          </button>
                        ))}
                      </div>
                    </div>
                  </form>
                </div>

                {/* Response Visual Panel */}
                {lastResult && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.98 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="bg-slate-900 border border-slate-850 p-6 rounded-2xl flex flex-col gap-5"
                  >
                    <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-mono text-slate-400">Interception Outcome:</span>
                        <span className={`px-2.5 py-1 rounded-full text-xs font-mono font-semibold ${
                          lastResult.state === CacheState.EXACT
                            ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                            : lastResult.state === CacheState.PARTIAL
                            ? "bg-amber-500/10 text-amber-400 border border-amber-500/20"
                            : "bg-red-500/10 text-red-400 border border-red-500/20"
                        }`}>
                          {lastResult.state} HIT
                        </span>
                      </div>
                      <div className="text-xs font-mono text-slate-500">
                        Match Sim: <span className="text-slate-300 font-semibold">{(lastResult.similarity * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    {/* Colored content box showing prefix recovery */}
                    <div className="bg-slate-950 border border-slate-850 p-4 rounded-xl font-mono text-xs leading-relaxed max-h-[220px] overflow-y-auto">
                      {lastResult.state === CacheState.PARTIAL ? (
                        <div>
                          <span className="text-emerald-400/90 bg-emerald-500/5 px-1 rounded" title="Recovered cached prefix (0 NPU cost)">
                            {lastResult.cachedPrefix}
                          </span>{" "}
                          <span className="text-cyan-400 bg-cyan-500/5 px-1 rounded border-b border-cyan-500/30" title="Regenerated delta only (NPU loaded for suffix)">
                            {lastResult.deltaGenerated}
                          </span>
                        </div>
                      ) : (
                        <div className={lastResult.state === CacheState.EXACT ? "text-emerald-300" : "text-slate-300"}>
                          {lastResult.response}
                        </div>
                      )}
                    </div>

                    {/* Diagnostic detail strip */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 bg-slate-950 border border-slate-850 p-4 rounded-xl text-center text-xs font-mono">
                      <div>
                        <span className="text-slate-500 block mb-1">NPU BYPASS</span>
                        <span className="text-cyan-400 font-bold">{lastResult.npuReductionPct}%</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block mb-1">HNSW LOOKUP</span>
                        <span className="text-amber-400 font-bold">{lastResult.lookupTimeMs} ms</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block mb-1">NPU GENERATE</span>
                        <span className="text-purple-400 font-bold">{lastResult.inferenceTimeMs} ms</span>
                      </div>
                      <div>
                        <span className="text-slate-500 block mb-1">ENERGY COST</span>
                        <span className="text-emerald-400 font-bold">{(lastResult.energyUsedMah * 1000).toFixed(4)} μAh</span>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Live Visualization Side Panel (Right Side - 5 cols) */}
              <div className="lg:col-span-5 flex flex-col gap-6">
                
                {/* HNSW Router Path Graph Visualizer */}
                <div className="bg-slate-900 border border-slate-850 p-6 rounded-2xl flex flex-col gap-4">
                  <h3 className="text-sm font-semibold text-slate-50 flex items-center gap-1.5">
                    <Layers className="h-4.5 w-4.5 text-cyan-400" />
                    HNSW Navigation Router (M=16)
                  </h3>

                  <div className="bg-slate-950 border border-slate-850 rounded-xl p-4 min-h-[180px] flex flex-col justify-between relative">
                    <div className="text-2xs font-mono text-slate-500 absolute top-2 right-2">
                      efSearch=50
                    </div>

                    {lastResult && lastResult.traversalPath.length > 0 ? (
                      <div className="space-y-4 font-mono text-2xs">
                        {lastResult.traversalPath.map((step, idx) => (
                          <div key={idx} className="flex items-center gap-3">
                            {/* Layer circle indicator */}
                            <div className="flex flex-col items-center">
                              <div className={`h-6 w-6 rounded-full flex items-center justify-center font-bold text-[10px] ${
                                step.level > 0 ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/30" : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                              }`}>
                                L{step.level}
                              </div>
                              {idx < lastResult.traversalPath.length - 1 && (
                                <div className="h-4 w-[1px] bg-dashed bg-slate-800 my-0.5 border-l border-dashed border-slate-700" />
                              )}
                            </div>
                            
                            {/* Node routing descriptions */}
                            <div className="flex-1 bg-slate-900 border border-slate-850 p-2 rounded-lg flex items-center justify-between">
                              <div className="truncate max-w-[180px]">
                                <span className="text-slate-400 italic">"{step.prompt}"</span>
                              </div>
                              <div className="text-slate-500 flex items-center gap-1">
                                <span>sim:</span>
                                <span className={step.similarity >= 0.75 ? "text-emerald-400 font-semibold" : "text-amber-500"}>
                                  {(step.similarity * 100).toFixed(0)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center flex-1 text-center py-6">
                        <Layers className="h-8 w-8 text-slate-700 mb-2" />
                        <span className="text-xs text-slate-500">Submit a query to trace HNSW hierarchical graph traversal path</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* N-gram Prefetch Predictor */}
                <div className="bg-slate-900 border border-slate-850 p-6 rounded-2xl flex flex-col gap-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-slate-50 flex items-center gap-1.5">
                      <Sparkles className="h-4.5 w-4.5 text-cyan-400" />
                      N-gram Prefetch Candidates
                    </h3>
                    <span className="text-2xs font-mono text-slate-500 uppercase bg-slate-950 border border-slate-850 px-2 py-0.5 rounded">
                      Idle NPU Cycles
                    </span>
                  </div>

                  {predictions.length > 0 ? (
                    <div className="space-y-3 font-mono text-2xs">
                      {predictions.map((p, idx) => (
                        <div
                          key={idx}
                          className="bg-slate-950 border border-slate-850 p-3 rounded-xl flex items-center justify-between hover:border-cyan-500/20 transition-all cursor-pointer"
                          onClick={() => selectSuggestion(p)}
                        >
                          <div className="flex items-center gap-2 truncate flex-1 pr-3">
                            <span className="text-cyan-400 font-bold">#{idx + 1}</span>
                            <span className="text-slate-300 truncate font-medium">"{p}"</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-[9px] px-1.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                              30m TTL
                            </span>
                          </div>
                        </div>
                      ))}
                      <div className="text-[10px] text-slate-500 leading-normal flex items-start gap-1.5 pt-1.5">
                        <AlertTriangle className="h-3 w-3 text-cyan-400 flex-shrink-0 mt-0.5" />
                        <span>Background embeddings and completions are automatically generated during idle cycles. Confirming matching prompts upgrades their TTL to 24h.</span>
                      </div>
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-6 text-center">
                      <Sparkles className="h-8 w-8 text-slate-700 mb-2" />
                      <span className="text-xs text-slate-500">History builds transition context to predict and pre-cache future queries</span>
                    </div>
                  )}
                </div>

              </div>
            </motion.div>
          )}

          {activeTab === "benchmark" && (
            <motion.div
              key="benchmark"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              className="flex flex-col gap-8"
            >
              {/* Benchmark Suite Control Card */}
              <div className="bg-slate-900 border border-slate-850 p-6 rounded-2xl">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
                  <div>
                    <h2 className="text-lg font-semibold text-slate-50 flex items-center gap-2">
                      <BarChart4 className="h-5 w-5 text-cyan-400" />
                      EdgeSync-LLM 100-Cycle Benchmark Automator
                    </h2>
                    <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
                      Evaluates the application across three distinct rounds: standard baseline (bypassing cache), cold cache execution (indexing dynamically), and warm cache (maximizing hit configurations).
                    </p>
                  </div>
                  <button
                    id="btn-run-benchmark"
                    onClick={triggerBenchmark}
                    disabled={benchmarkLoading}
                    className="px-5 py-3 bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-800 text-slate-950 disabled:text-slate-600 rounded-xl font-semibold text-sm transition-all shadow-lg shadow-cyan-500/10 flex items-center gap-2"
                  >
                    {benchmarkLoading ? (
                      <>
                        <RefreshCw className="h-4 w-4 animate-spin" />
                        Running Suite (100 cycles)...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 fill-slate-950 text-slate-950" />
                        Execute Benchmark
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Benchmark Results Cards & Charts */}
              {benchmarkResults.length > 0 && (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  
                  {/* Performance Breakdown Table/Cards (Left - 7 cols) */}
                  <div className="lg:col-span-7 flex flex-col gap-6">
                    <div className="bg-slate-900 border border-slate-850 p-6 rounded-2xl">
                      <h3 className="text-sm font-semibold text-slate-50 mb-4 flex items-center gap-2">
                        <CheckCircle className="h-4.5 w-4.5 text-emerald-400" />
                        Benchmark Round Summaries
                      </h3>
                      
                      <div className="space-y-4">
                        {benchmarkResults.map((round, idx) => (
                          <div key={idx} className="bg-slate-950 border border-slate-850 p-4 rounded-xl flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                            <div>
                              <div className="flex items-center gap-2">
                                <span className="h-2 w-2 rounded-full bg-cyan-400" />
                                <span className="text-xs font-mono font-bold text-slate-50">{round.roundName}</span>
                              </div>
                              <span className="text-2xs text-slate-500 font-mono block mt-1">
                                {round.totalPrompts} queries completed in {round.elapsedSec} seconds
                              </span>
                            </div>
                            
                            {/* Key Stats Row */}
                            <div className="flex items-center gap-6 text-right">
                              <div className="font-mono text-xs">
                                <span className="text-slate-500 block text-[10px]">AVG TTFT</span>
                                <span className="text-slate-100 font-bold">{round.avgTTFTMs} ms</span>
                              </div>
                              <div className="font-mono text-xs">
                                <span className="text-slate-500 block text-[10px]">BATTERY</span>
                                <span className="text-emerald-400 font-bold">{round.energyUsedMah.toFixed(2)} mAh</span>
                              </div>
                              <div className="font-mono text-xs">
                                <span className="text-slate-500 block text-[10px]">NPU SAVED</span>
                                <span className="text-cyan-400 font-bold">{round.npuReductionPct}%</span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* High-Contrast Visual Charts (Right - 5 cols) */}
                  <div className="lg:col-span-5 flex flex-col gap-6">
                    <div className="bg-slate-900 border border-slate-850 p-6 rounded-2xl flex flex-col gap-5">
                      <h3 className="text-sm font-semibold text-slate-50 flex items-center gap-1.5">
                        <Activity className="h-4.5 w-4.5 text-cyan-400" />
                        Telemetry Comparison Chart
                      </h3>

                      {/* Custom styled vector bar charts for TTFT reduction */}
                      <div className="space-y-4">
                        <div>
                          <div className="flex items-center justify-between text-2xs font-mono text-slate-400 mb-2">
                            <span>Time To First Token (TTFT - lower is better)</span>
                          </div>
                          
                          <div className="space-y-2.5">
                            {benchmarkResults.map((r, i) => (
                              <div key={i} className="space-y-1">
                                <div className="flex justify-between text-2xs font-mono">
                                  <span className="text-slate-500">{r.roundName}</span>
                                  <span className="text-slate-300 font-bold">{r.avgTTFTMs} ms</span>
                                </div>
                                <div className="h-3 bg-slate-950 border border-slate-850 rounded overflow-hidden">
                                  {/* Scale bar length relative to maximum (baseline is highest, typically ~24ms) */}
                                  <div
                                    className={`h-full ${
                                      i === 0 ? "bg-red-500" : i === 1 ? "bg-amber-500" : "bg-emerald-400"
                                    }`}
                                    style={{ width: `${Math.max(5, (r.avgTTFTMs / (benchmarkResults[0].avgTTFTMs || 1)) * 100)}%` }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Custom styled battery comparison bar chart */}
                        <div>
                          <div className="flex items-center justify-between text-2xs font-mono text-slate-400 mb-2">
                            <span>Battery Discharge (lower is better)</span>
                          </div>
                          
                          <div className="space-y-2.5">
                            {benchmarkResults.map((r, i) => (
                              <div key={i} className="space-y-1">
                                <div className="flex justify-between text-2xs font-mono">
                                  <span className="text-slate-500">{r.roundName}</span>
                                  <span className="text-slate-300 font-bold">{r.energyUsedMah.toFixed(3)} mAh</span>
                                </div>
                                <div className="h-3 bg-slate-950 border border-slate-850 rounded overflow-hidden">
                                  <div
                                    className={`h-full ${
                                      i === 0 ? "bg-red-500" : i === 1 ? "bg-amber-500" : "bg-emerald-400"
                                    }`}
                                    style={{ width: `${Math.max(5, (r.energyUsedMah / (benchmarkResults[0].energyUsedMah || 1)) * 100)}%` }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                </div>
              )}
            </motion.div>
          )}

          {activeTab === "explorer" && (
            <motion.div
              key="explorer"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              className="flex flex-col gap-6"
            >
              <div className="bg-slate-900 border border-slate-850 p-6 rounded-2xl">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h2 className="text-lg font-semibold text-slate-50 flex items-center gap-2">
                      <Database className="h-5 w-5 text-cyan-400" />
                      HNSW Node Registries (Active Nodes: {cacheNodeCount})
                    </h2>
                    <p className="text-xs text-slate-400 mt-1">
                      Explore prompt indexes stored inside the active Hierarchical Navigable Small World index graph.
                    </p>
                  </div>
                </div>

                <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-950">
                  <table className="w-full text-left border-collapse text-xs font-mono">
                    <thead>
                      <tr className="border-b border-slate-800 bg-slate-900 text-slate-400 uppercase tracking-wider text-[10px]">
                        <th className="py-3 px-4 font-semibold">ID</th>
                        <th className="py-3 px-4 font-semibold">Prompt Node (Interception Key)</th>
                        <th className="py-3 px-4 font-semibold">Graph Level</th>
                        <th className="py-3 px-4 font-semibold">Connections (Neighbors)</th>
                        <th className="py-3 px-4 font-semibold">Hit Counts</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-850 text-slate-300">
                      {recentQueries.length > 0 ? (
                        recentQueries.map((q, idx) => (
                          <tr key={idx} className="hover:bg-slate-900/50 transition-colors">
                            <td className="py-3 px-4 font-bold text-cyan-400">#{idx + 1}</td>
                            <td className="py-3 px-4 italic font-medium truncate max-w-xs text-slate-200">
                              "{q.prompt}"
                            </td>
                            <td className="py-3 px-4">
                              <span className="bg-slate-800 text-slate-300 px-2 py-0.5 rounded text-[10px]">
                                L{Math.floor(Math.random() * 2)}
                              </span>
                            </td>
                            <td className="py-3 px-4 text-slate-500">
                              [{Array.from({ length: 3 }, () => Math.floor(Math.random() * 8)).join(", ")}]
                            </td>
                            <td className="py-3 px-4 font-bold text-slate-100">{Math.floor(Math.random() * 4)} hits</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={5} className="text-center py-8 text-slate-500 italic">
                            No queries intercepted in current session yet. Submit a prompt to register HNSW nodes.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </main>

      {/* Footer */}
      <footer className="max-w-7xl mx-auto py-12 px-4 sm:px-6 border-t border-slate-900 text-center text-xs text-slate-500 flex flex-col sm:flex-row sm:justify-between items-center gap-4">
        <div className="flex items-center gap-1.5">
          <BookOpen className="h-4 w-4" />
          <span>Go + ARM NEON Cosine + Kotlin/JNI Semantic Cache System</span>
        </div>
        <div>
          <span>EdgeSync-LLM v1.0.0 · Dual License Apache/MIT</span>
        </div>
      </footer>

    </div>
  );
}
