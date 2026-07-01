//go:build !cgo

package core

// cosineSimilarityAccelerated is a no-op stub when CGO is disabled
// (e.g. `go run ./benchmark/` per the README's no-CGO host build mode).
// cosineDistance() in hnsw.go falls back to the pure-Go implementation.
func cosineSimilarityAccelerated(a, b []float32) (float32, bool) {
	return 0, false
}
