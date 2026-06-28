/**
 * @file cacheManager.ts
 * @description Pure TypeScript HNSW index and N-gram prefetch predictor for server-side semantic caching.
 * Matches the mathematical specifications of the core Go & C engines.
 */

import { HNSWTraversalStep, CacheState } from "./types";

export interface CacheEntryNode {
  id: number;
  prompt: string;
  vector: number[]; // 384 dimensions
  response: string;
  level: number;
  neighbors: number[][]; // Neighbors[level] = list of neighbor IDs
  hitCount: number;
  createdAt: number;
  expiresAt: number;
  confirmed: boolean;
}

export class CacheManager {
  // HNSW parameters
  private M = 16;
  private M0 = 32;
  private mL = 1.0 / Math.log(16);
  private maxLevel = -1;
  private enterNodeId = -1;
  public nodes: Map<number, CacheEntryNode> = new Map();
  private nextId = 1;

  // Predictor state
  public promptHistory: string[] = [];
  private transitions: Map<string, Map<string, number>> = new Map();

  constructor() {
    this.seedInitialGraph();
  }

  // Calculate cosine similarity between two float arrays
  public cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    if (normA <= 0 || normB <= 0) return 0;
    return dot / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  // Generate random level for HNSW
  private generateLevel(): number {
    const r = Math.random() || 0.0001;
    return Math.floor(-Math.log(r) * this.mL);
  }

  // Insert a new vector embedding into the HNSW graph
  public insert(
    prompt: string,
    vector: number[],
    response: string,
    ttlMinutes = 1440, // 24 hours standard TTL
    confirmed = true
  ): CacheEntryNode {
    const id = this.nextId++;
    const level = this.generateLevel();
    const now = Date.now();
    const expiresAt = now + ttlMinutes * 60 * 1000;

    const node: CacheEntryNode = {
      id,
      prompt,
      vector: vector.slice(0, 384), // Slice to 384 dimensions matching C NEON
      response,
      level,
      neighbors: Array.from({ length: level + 1 }, () => []),
      hitCount: 0,
      createdAt: now,
      expiresAt,
      confirmed,
    };

    this.nodes.set(id, node);

    if (this.maxLevel === -1) {
      this.maxLevel = level;
      this.enterNodeId = id;
      return node;
    }

    let currNodeId = this.enterNodeId;
    let currDist = 1.0 - this.cosineSimilarity(vector, this.nodes.get(currNodeId)!.vector);

    // Phase 1: Downward search to locate closest entry point at inserted level
    for (let l = this.maxLevel; l > level; l--) {
      let changed = true;
      while (changed) {
        changed = false;
        const currNode = this.nodes.get(currNodeId)!;
        for (const neighborId of currNode.neighbors[l] || []) {
          const neighbor = this.nodes.get(neighborId)!;
          const dist = 1.0 - this.cosineSimilarity(vector, neighbor.vector);
          if (dist < currDist) {
            currDist = dist;
            currNodeId = neighborId;
            changed = true;
          }
        }
      }
    }

    // Phase 2: Link nodes at intersecting levels
    let candidates = [currNodeId];
    const topLevelToInsert = Math.min(level, this.maxLevel);
    for (let l = topLevelToInsert; l >= 0; l--) {
      candidates = this.searchLayer(vector, candidates, 30, l);
      const mMax = l === 0 ? this.M0 : this.M;
      this.connectNeighbors(node, candidates, mMax, l);
    }

    if (level > this.maxLevel) {
      this.maxLevel = level;
      this.enterNodeId = id;
    }

    return node;
  }

  // Explore a graph layer for nearest neighbors
  private searchLayer(query: number[], enterPoints: number[], ef: number, level: number): number[] {
    const visited = new Set<number>(enterPoints);
    
    interface NodeDist {
      id: number;
      dist: number;
    }

    const candidates: NodeDist[] = [];
    const result: NodeDist[] = [];

    for (const ep of enterPoints) {
      const node = this.nodes.get(ep)!;
      const d = 1.0 - this.cosineSimilarity(query, node.vector);
      candidates.push({ id: ep, dist: d });
      result.push({ id: ep, dist: d });
    }

    const sortFn = (a: NodeDist, b: NodeDist) => a.dist - b.dist;
    candidates.sort(sortFn);
    result.sort(sortFn);

    while (candidates.length > 0) {
      const curr = candidates.shift()!;
      const furthestInResult = result[result.length - 1];
      if (curr.dist > furthestInResult.dist) {
        break;
      }

      const currNode = this.nodes.get(curr.id)!;
      for (const neighborId of currNode.neighbors[level] || []) {
        if (!visited.has(neighborId)) {
          visited.add(neighborId);
          const neighbor = this.nodes.get(neighborId)!;
          const d = 1.0 - this.cosineSimilarity(query, neighbor.vector);

          if (d < furthestInResult.dist || result.length < ef) {
            candidates.push({ id: neighborId, dist: d });
            result.push({ id: neighborId, dist: d });
            candidates.sort(sortFn);
            result.sort(sortFn);

            if (result.length > ef) {
              result.pop();
            }
          }
        }
      }
    }

    return result.map((r) => r.id);
  }

  // Build bilateral links between node and candidates
  private connectNeighbors(node: CacheEntryNode, candidates: number[], mMax: number, level: number) {
    const limit = Math.min(candidates.length, mMax);
    for (let i = 0; i < limit; i++) {
      const candidateId = candidates[i];
      const candidate = this.nodes.get(candidateId)!;

      node.neighbors[level].push(candidateId);
      candidate.neighbors[level].push(node.id);

      // Prune if neighbors exceed limit
      const cMax = level === 0 ? this.M0 : this.M;
      if (candidate.neighbors[level].length > cMax) {
        this.pruneConnections(candidate, level, cMax);
      }
    }
  }

  // Prune node neighbors retaining closest ones
  private pruneConnections(node: CacheEntryNode, level: number, maxConnections: number) {
    const sortedNeighbors = node.neighbors[level]
      .map((neighborId) => {
        const neighbor = this.nodes.get(neighborId)!;
        return {
          id: neighborId,
          dist: 1.0 - this.cosineSimilarity(node.vector, neighbor.vector),
        };
      })
      .sort((a, b) => a.dist - b.dist);

    node.neighbors[level] = sortedNeighbors.slice(0, maxConnections).map((s) => s.id);
  }

  // Query HNSW graph returning approximate nearest neighbors + complete level-by-level traversal path for UI rendering
  public search(query: number[], k = 1): { node: CacheEntryNode; similarity: number; traversalPath: HNSWTraversalStep[] }[] {
    if (this.maxLevel === -1 || this.nodes.size === 0) {
      return [];
    }

    const traversalPath: HNSWTraversalStep[] = [];
    let currNodeId = this.enterNodeId;
    let currNode = this.nodes.get(currNodeId)!;
    let currSim = this.cosineSimilarity(query, currNode.vector);

    traversalPath.push({
      level: this.maxLevel,
      nodeId: currNodeId,
      prompt: currNode.prompt,
      similarity: currSim,
    });

    // Step 1: Greedy routing down top levels
    for (let l = this.maxLevel; l > 0; l--) {
      let changed = true;
      while (changed) {
        changed = false;
        currNode = this.nodes.get(currNodeId)!;
        for (const neighborId of currNode.neighbors[l] || []) {
          const neighbor = this.nodes.get(neighborId)!;
          const sim = this.cosineSimilarity(query, neighbor.vector);
          if (sim > currSim) {
            currSim = sim;
            currNodeId = neighborId;
            changed = true;
            traversalPath.push({
              level: l,
              nodeId: neighborId,
              prompt: neighbor.prompt,
              similarity: sim,
            });
          }
        }
      }
    }

    // Step 2: Layer 0 exploration with efSearch = 50
    const results = this.searchLayer(query, [currNodeId], 50, 0);

    // Step 3: Package top-k results
    const finalResults = results.slice(0, k).map((id) => {
      const node = this.nodes.get(id)!;
      const sim = this.cosineSimilarity(query, node.vector);
      return {
        node,
        similarity: sim,
        traversalPath: [...traversalPath, {
          level: 0,
          nodeId: id,
          prompt: node.prompt,
          similarity: sim,
        }],
      };
    });

    return finalResults;
  }

  // Record a prompt in history, training N-gram transition weights
  public recordPrompt(prompt: string) {
    const clean = prompt.toLowerCase().trim();
    if (!clean) return;

    if (this.promptHistory.length > 0) {
      const prev = this.promptHistory[this.promptHistory.length - 1];
      if (!this.transitions.has(prev)) {
        this.transitions.set(prev, new Map());
      }
      const prevTrans = this.transitions.get(prev)!;
      prevTrans.set(clean, (prevTrans.get(clean) || 0) + 1);
    }

    this.promptHistory.push(clean);
    if (this.promptHistory.length > 20) {
      this.promptHistory.shift();
    }
  }

  // Predict top-3 upcoming prompts based on N-gram transition weights
  public predictNext(): string[] {
    if (this.promptHistory.length === 0) return [];
    const current = this.promptHistory[this.promptHistory.length - 1];

    const trans = this.transitions.get(current);
    if (!trans) {
      // Fallback: match by common keywords in history
      const keywords = current.split(" ").filter((w) => w.length > 3);
      if (keywords.length === 0) return [];
      const matches: string[] = [];
      const seen = new Set<string>();

      for (let i = this.promptHistory.length - 1; i >= 0; i--) {
        const hist = this.promptHistory[i];
        if (hist !== current && keywords.some((k) => hist.includes(k)) && !seen.has(hist)) {
          seen.add(hist);
          matches.push(hist);
          if (matches.length >= 3) break;
        }
      }
      return matches;
    }

    const sorted = [...trans.entries()]
      .sort((a, b) => b[1] - a[1])
      .map((entry) => entry[0]);

    return sorted.slice(0, 3);
  }

  // Confirm a prefetch item, upgrading its TTL to 24 hours
  public confirmPrompt(prompt: string) {
    const clean = prompt.toLowerCase().trim();
    for (const [_, node] of this.nodes.entries()) {
      if (node.prompt.toLowerCase().trim() === clean) {
        node.confirmed = true;
        node.expiresAt = Date.now() + 24 * 60 * 60 * 1000; // Extend to 24h
        break;
      }
    }
  }

  // Evict items that have exceeded their TTL (30 min for prefetch, 24 hours for confirmed)
  public evictExpired(): string[] {
    const now = Date.now();
    const evicted: string[] = [];
    for (const [id, node] of this.nodes.entries()) {
      if (now > node.expiresAt) {
        evicted.push(node.prompt);
        this.nodes.delete(id);
      }
    }
    return evicted;
  }

  // Clear cache completely
  public clear() {
    this.nodes.clear();
    this.promptHistory = [];
    this.transitions.clear();
    this.maxLevel = -1;
    this.enterNodeId = -1;
    this.nextId = 1;
    this.seedInitialGraph();
  }

  // Pre-seed the HNSW graph with realistic technical contexts so the initial dashboard has gorgeous visual components
  private seedInitialGraph() {
    const samples = [
      {
        prompt: "what are the core benefits of rust over c++ for embedded systems",
        response: "Rust provides memory safety guarantees without a garbage collector through its ownership model, preventing common bugs like buffer overflows and null pointer dereferences which are critical in embedded environments. It also supports seamless C interop, rich abstraction mechanisms, and modern build tooling.",
      },
      {
        prompt: "explain quantum computing in simple high school terms",
        response: "Standard computers use bits representing 0 or 1. Quantum computers use qubits which can exist as 0, 1, or both at the same time (superposition). This allows quantum computers to process complex calculations exponentially faster, solving massive problems in chemistry, encryption, and modeling.",
      },
      {
        prompt: "write a fast fiber-based http server in modern go",
        response: "Using the Fiber framework in Go, you can build super fast routers:\n```go\npackage main\nimport \"github.com/gofiber/fiber/v2\"\n\nfunc main() {\n    app := fiber.New()\n    app.Get(\"/\", func(c *fiber.Ctx) error {\n        return c.SendString(\"Hello World\")\n    })\n    app.Listen(\":3000\")\n}\n```",
      },
      {
        prompt: "how to configure write-ahead logging in sqlite",
        response: "To activate WAL mode in SQLite, execute the pragma command: `PRAGMA journal_mode = WAL;`. This enables multiple concurrent readers to read the database while a writer is writing, preventing locking blockages and significantly improving transaction speeds on disk.",
      }
    ];

    for (const s of samples) {
      // Create a dummy 384-dimension vector with distinct patterns based on character values
      const vector = Array.from({ length: 384 }, (_, idx) => {
        const charSum = s.prompt.charCodeAt(idx % s.prompt.length) || 0;
        return Math.sin(charSum + idx) * 0.05 + 0.05;
      });
      // Normalize vector for clean cosine similarity
      const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
      const normalized = vector.map((v) => v / (magnitude || 1));

      this.insert(s.prompt, normalized, s.response, 1440, true);
    }
  }
}
