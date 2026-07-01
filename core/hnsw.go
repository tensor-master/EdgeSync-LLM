package core

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
)

// HNSW configures the Hierarchical Navigable Small World approximate nearest neighbor index.
type HNSW struct {
	M              int           // Max links per node at non-zero layers
	M0             int           // Max links per node at layer 0
	EfConstruction int           // Size of the dynamic candidate list during construction
	EfSearch       int           // Size of the dynamic candidate list during search
	ML             float64       // Normalization factor for level generation
	MaxLevel       int           // Max level of the graph
	EnterNode      int           // Node ID of the entry node
	Nodes          map[int]*Node // Index of all nodes in the graph
}

// Node represents a single element in the HNSW graph.
type Node struct {
	ID        int           // Corresponds to the database record ID
	Vector    []float32     // Vector embedding (typically 384 dimensions)
	Level     int           // Topmost layer level this node exists on
	Neighbors [][]int       // Connections at each layer: Neighbors[level][neighbor_node_id]
}

// SearchResult represents a single approximate nearest neighbor candidate.
type SearchResult struct {
	ID         int
	Similarity float32
}

// NewHNSW instantiates an HNSW index with target default values.
func NewHNSW(m int, efSearch int) *HNSW {
	if m <= 0 {
		m = 16
	}
	if efSearch <= 0 {
		efSearch = 50
	}
	return &HNSW{
		M:              m,
		M0:             m * 2,
		EfConstruction: 100,
		EfSearch:       efSearch,
		ML:             1.0 / math.Log(float64(m)),
		MaxLevel:       -1,
		EnterNode:      -1,
		Nodes:          make(map[int]*Node),
	}
}

// cosineDistance calculates the cosine distance (1.0 - cosine_similarity).
// When built with CGO enabled, this tries the accelerated C path in
// cosine_neon.c (NEON on ARM, portable scalar C elsewhere) first; the
// pure-Go path below is always available as a fallback and is the only
// path used in no-CGO host builds (see README "Host build" mode).
func cosineDistance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0
	}
	if sim, ok := cosineSimilarityAccelerated(a, b); ok {
		if sim > 1.0 {
			sim = 1.0
		} else if sim < -1.0 {
			sim = -1.0
		}
		return 1.0 - sim
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA <= 0 || normB <= 0 {
		return 1.0
	}
	sim := dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	// Clamp similarity to [-1.0, 1.0]
	if sim > 1.0 {
		sim = 1.0
	} else if sim < -1.0 {
		sim = -1.0
	}
	return 1.0 - sim
}

// generateLevel computes a random layer height for a new node using an exponential decay distribution.
func (h *HNSW) generateLevel() int {
	r := rand.Float64()
	if r == 0 {
		r = 0.0001
	}
	return int(-math.Log(r) * h.ML)
}

// Insert adds a new vector embedding to the HNSW graph, constructing its link layers.
func (h *HNSW) Insert(id int, vector []float32) {
	// Create Node
	level := h.generateLevel()
	node := &Node{
		ID:        id,
		Vector:    vector,
		Level:     level,
		Neighbors: make([][]int, level+1),
	}
	for i := 0; i <= level; i++ {
		node.Neighbors[i] = []int{}
	}
	h.Nodes[id] = node

	// If index is empty, make this node the entry point
	if h.MaxLevel == -1 {
		h.MaxLevel = level
		h.EnterNode = id
		return
	}

	currNodeID := h.EnterNode
	currDist := cosineDistance(vector, h.Nodes[currNodeID].Vector)

	// Phase 1: Search top levels down to inserted level to locate closest entry node
	for l := h.MaxLevel; l > level; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range h.Nodes[currNodeID].Neighbors[l] {
				d := cosineDistance(vector, h.Nodes[neighborID].Vector)
				if d < currDist {
					currDist = d
					currNodeID = neighborID
					changed = true
				}
			}
		}
	}

	// Phase 2: Insert node at lower levels, establishing bilateral links
	// Keep a slice of closest points found across levels
	candidates := []int{currNodeID}
	for l := min(level, h.MaxLevel); l >= 0; l-- {
		// Find nearest neighbors on layer l
		candidates = h.searchLayer(vector, candidates, h.EfConstruction, l)
		
		// Establish links up to parameter M
		mMax := h.M
		if l == 0 {
			mMax = h.M0
		}
		
		// Connect bidirectionally
		h.connectNeighbors(node, candidates, mMax, l)
	}

	// Update global entry point if this node's level is higher
	if level > h.MaxLevel {
		h.MaxLevel = level
		h.EnterNode = id
	}
}

// searchLayer explores a single graph layer, returning the closest candidates found.
func (h *HNSW) searchLayer(query []float32, enterPoints []int, ef int, level int) []int {
	visited := make(map[int]bool)
	for _, ep := range enterPoints {
		visited[ep] = true
	}

	// Dynamic priority lists
	// candidates contains points to explore (ordered by distance ascending)
	// result contains best candidates found (ordered by distance ascending)
	type pair struct {
		id   int
		dist float32
	}

	var candidates []pair
	var result []pair

	for _, ep := range enterPoints {
		d := cosineDistance(query, h.Nodes[ep].Vector)
		candidates = append(candidates, pair{ep, d})
		result = append(result, pair{ep, d})
	}

	// Helper to sort pair slices
	sortPairs := func(p []pair) {
		for i := 0; i < len(p)-1; i++ {
			for j := i + 1; j < len(p); j++ {
				if p[i].dist > p[j].dist {
					p[i], p[j] = p[j], p[i]
				}
			}
		}
	}

	sortPairs(candidates)
	sortPairs(result)

	for len(candidates) > 0 {
		// Get nearest element from candidates
		curr := candidates[0]
		candidates = candidates[1:]

		// Get furthest element from result
		furthestInResult := result[len(result)-1]
		if curr.dist > furthestInResult.dist {
			break
		}

		// Explore neighbors of curr
		for _, neighborID := range h.Nodes[curr.id].Neighbors[level] {
			if !visited[neighborID] {
				visited[neighborID] = true
				d := cosineDistance(query, h.Nodes[neighborID].Vector)
				
				furthestInResult = result[len(result)-1]
				if d < furthestInResult.dist || len(result) < ef {
					candidates = append(candidates, pair{neighborID, d})
					result = append(result, pair{neighborID, d})
					sortPairs(candidates)
					sortPairs(result)

					if len(result) > ef {
						result = result[:ef]
					}
				}
			}
		}
	}

	resIDs := make([]int, len(result))
	for i, p := range result {
		resIDs[i] = p.id
	}
	return resIDs
}

// connectNeighbors establishes bilateral edges between the inserted node and closest layer nodes.
func (h *HNSW) connectNeighbors(node *Node, candidates []int, mMax int, level int) {
	// Simple heuristic: connect up to mMax nearest candidates
	limit := mMax
	if len(candidates) < limit {
		limit = len(candidates)
	}

	for i := 0; i < limit; i++ {
		candidateID := candidates[i]
		candidate := h.Nodes[candidateID]

		// Add connection
		node.Neighbors[level] = append(node.Neighbors[level], candidateID)
		candidate.Neighbors[level] = append(candidate.Neighbors[level], node.ID)

		// Prune candidate connections if they exceed limit
		cMax := h.M
		if level == 0 {
			cMax = h.M0
		}
		if len(candidate.Neighbors[level]) > cMax {
			h.pruneConnections(candidate, level, cMax)
		}
	}
}

// pruneConnections shrinks neighbor list down to standard capacity limit, retaining closest vectors.
func (h *HNSW) pruneConnections(node *Node, level int, maxConnections int) {
	type pair struct {
		id   int
		dist float32
	}
	var pairs []pair
	for _, neighborID := range node.Neighbors[level] {
		d := cosineDistance(node.Vector, h.Nodes[neighborID].Vector)
		pairs = append(pairs, pair{neighborID, d})
	}

	// Sort ascending
	for i := 0; i < len(pairs)-1; i++ {
		for j := i + 1; j < len(pairs); j++ {
			if pairs[i].dist > pairs[j].dist {
				pairs[i], pairs[j] = pairs[j], pairs[i]
			}
		}
	}

	// Slice and re-assign neighbors
	keep := maxConnections
	if len(pairs) < keep {
		keep = len(pairs)
	}
	newNeighbors := make([]int, keep)
	for i := 0; i < keep; i++ {
		newNeighbors[i] = pairs[i].id
	}
	node.Neighbors[level] = newNeighbors
}

// Search queries the HNSW index to find the approximate top-k nearest neighbors.
func (h *HNSW) Search(query []float32, k int) []SearchResult {
	if h.MaxLevel == -1 || len(h.Nodes) == 0 {
		return nil
	}

	currNodeID := h.EnterNode
	currDist := cosineDistance(query, h.Nodes[currNodeID].Vector)

	// Step 1: Greedy routing down top layers
	for l := h.MaxLevel; l > 0; l-- {
		changed := true
		for changed {
			changed = false
			for _, neighborID := range h.Nodes[currNodeID].Neighbors[l] {
				d := cosineDistance(query, h.Nodes[neighborID].Vector)
				if d < currDist {
					currDist = d
					currNodeID = neighborID
					changed = true
				}
			}
		}
	}

	// Step 2: HNSW Search at layer 0 using efSearch parameter
	results := h.searchLayer(query, []int{currNodeID}, h.EfSearch, 0)

	// Step 3: Package top-k results
	limit := k
	if len(results) < limit {
		limit = len(results)
	}

	finalResults := make([]SearchResult, limit)
	for i := 0; i < limit; i++ {
		id := results[i]
		n := h.Nodes[id]
		dist := cosineDistance(query, n.Vector)
		similarity := 1.0 - dist // Map back to cosine similarity [-1.0, 1.0]
		finalResults[i] = SearchResult{
			ID:         id,
			Similarity: similarity,
		}
	}

	return finalResults
}


// Delete removes a node from the HNSW index and purges all references to it
// from neighboring nodes. The entry node is updated if necessary.
//
// Time complexity: O(degree × M) — bounded by the max neighbor count.
// Thread safety: caller must hold an exclusive lock if using concurrently.
func (h *HNSW) Delete(id int) {
	node, exists := h.Nodes[id]
	if !exists {
		return
	}

	// Remove all references to this node from its neighbors
	for level, neighbors := range node.Neighbors {
		for _, neighborID := range neighbors {
			neighbor, ok := h.Nodes[neighborID]
			if !ok {
				continue
			}
			if level >= len(neighbor.Neighbors) {
				continue
			}
			// Filter out the deleted node from this neighbor's list
			filtered := neighbor.Neighbors[level][:0]
			for _, nid := range neighbor.Neighbors[level] {
				if nid != id {
					filtered = append(filtered, nid)
				}
			}
			neighbor.Neighbors[level] = filtered
		}
	}

	// Remove the node itself
	delete(h.Nodes, id)

	// If the deleted node was the entry point, elect a new one
	if h.EnterNode == id {
		h.EnterNode = -1
		h.MaxLevel = -1
		for nid, n := range h.Nodes {
			if h.MaxLevel == -1 || n.Level > h.MaxLevel {
				h.MaxLevel = n.Level
				h.EnterNode = nid
			}
		}
	}
}

// Len returns the number of nodes currently in the index.
func (h *HNSW) Len() int {
	return len(h.Nodes)
}

// GobNode is a helper representation of a Node for encoding.
type GobNode struct {
	ID        int
	Vector    []float32
	Level     int
	Neighbors [][]int
}

// Serialize converts the entire graph into a binary format.
func (h *HNSW) Serialize() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	// We store HNSW config + list of Nodes
	gobNodes := make([]GobNode, 0, len(h.Nodes))
	for _, n := range h.Nodes {
		gobNodes = append(gobNodes, GobNode{
			ID:        n.ID,
			Vector:    n.Vector,
			Level:     n.Level,
			Neighbors: n.Neighbors,
		})
	}

	// Register structures
	gob.Register(GobNode{})

	err := enc.Encode(h.M)
	if err != nil {
		return nil, err
	}
	_ = enc.Encode(h.M0)
	_ = enc.Encode(h.EfConstruction)
	_ = enc.Encode(h.EfSearch)
	_ = enc.Encode(h.ML)
	_ = enc.Encode(h.MaxLevel)
	_ = enc.Encode(h.EnterNode)
	_ = enc.Encode(gobNodes)

	return buf.Bytes(), nil
}

// Deserialize restores the graph from serialized binary data.
func (h *HNSW) Deserialize(data []byte) error {
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)

	var m, m0, efConst, efSearch int
	var ml float64
	var maxLevel, enterNode int
	var gobNodes []GobNode

	if err := dec.Decode(&m); err != nil {
		return fmt.Errorf("failed to decode M: %w", err)
	}
	_ = dec.Decode(&m0)
	_ = dec.Decode(&efConst)
	_ = dec.Decode(&efSearch)
	_ = dec.Decode(&ml)
	_ = dec.Decode(&maxLevel)
	_ = dec.Decode(&enterNode)
	_ = dec.Decode(&gobNodes)

	h.M = m
	h.M0 = m0
	h.EfConstruction = efConst
	h.EfSearch = efSearch
	h.ML = ml
	h.MaxLevel = maxLevel
	h.EnterNode = enterNode
	h.Nodes = make(map[int]*Node)

	for _, gn := range gobNodes {
		h.Nodes[gn.ID] = &Node{
			ID:        gn.ID,
			Vector:    gn.Vector,
			Level:     gn.Level,
			Neighbors: gn.Neighbors,
		}
	}

	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
