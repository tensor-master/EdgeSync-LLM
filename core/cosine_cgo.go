//go:build cgo

package core

/*
#cgo CFLAGS: -O3
float cosine_similarity_neon_f32(const float *a, const float *b, int dims);
*/
import "C"
import "unsafe"

// cosineSimilarityAccelerated calls into cosine_neon.c via CGO. On ARM with
// NEON support, this uses the vectorized code path (cosine_similarity_neon_f32);
// on any other target (including x86_64 host builds), the same C file's #else
// branch runs a portable scalar C implementation. Either way this exercises
// the real C file — it is no longer orphaned/unused.
//
// This is intentionally separate from the pure-Go cosineDistance() in hnsw.go,
// which remains the default and the only path used when CGO_ENABLED=0 (see
// cosine_nocgo.go and the README's "Host build (no CGO)" mode).
func cosineSimilarityAccelerated(a, b []float32) (float32, bool) {
	if len(a) == 0 || len(a) != len(b) {
		return 0, false
	}
	sim := C.cosine_similarity_neon_f32(
		(*C.float)(unsafe.Pointer(&a[0])),
		(*C.float)(unsafe.Pointer(&b[0])),
		C.int(len(a)),
	)
	return float32(sim), true
}
