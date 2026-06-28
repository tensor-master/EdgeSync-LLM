/**
 * @file types.ts
 * @description Shared TypeScript types, interfaces, and enums for EdgeSync-LLM.
 */

export enum CacheState {
  EXACT = "EXACT",
  PARTIAL = "PARTIAL",
  MISS = "MISS",
}

export interface HNSWNode {
  id: number;
  prompt: string;
  level: number;
  neighbors: number[][]; // Level-based neighbor IDs
}

export interface HNSWTraversalStep {
  level: number;
  nodeId: number;
  prompt: string;
  similarity: number;
}

export interface QueryResult {
  state: CacheState;
  prompt: string;
  response: string;
  similarity: number;
  lookupTimeMs: number;
  inferenceTimeMs: number;
  npuReductionPct: number;
  energyUsedMah: number;
  tokensGenerated: number;
  timestamp: string;
  cachedPrefix?: string;
  deltaGenerated?: string;
  traversalPath: HNSWTraversalStep[];
  predictions: string[];
}

export interface BenchmarkMetrics {
  roundName: string;
  totalPrompts: number;
  avgTTFTMs: number;
  npuReductionPct: number;
  energyUsedMah: number;
  hitRatePct: number;
  tokensGenerated: number;
  elapsedSec: number;
}

export interface SessionMetrics {
  totalQueries: number;
  hitCount: number;
  partialHitCount: number;
  missCount: number;
  hitRatePct: number;
  averageLatencyMs: number;
  npuReductionPct: number;
  batterySavingsMah: number;
}
