// Package main — JNI bridge between Kotlin (EdgeSyncLLM.kt) and the Go cache engine.
//
// BUILD
// ──────
// This file is compiled as a shared library for Android ARM64:
//
//   Option A — gomobile bind (recommended for most projects):
//     go install golang.org/x/mobile/cmd/gomobile@latest
//     gomobile init
//     gomobile bind -target=android/arm64 -o edgecache.aar ./sdk/android/
//
//   Option B — manual NDK cross-compilation (for full CGO control):
//     export ANDROID_NDK_HOME=/path/to/ndk
//     export CC=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang
//     CGO_ENABLED=1 GOOS=android GOARCH=arm64 \
//         go build -buildmode=c-shared \
//         -o sdk/android/libs/arm64-v8a/libedgecache.so \
//         ./sdk/android/
//
// JNI NAMING CONVENTION
// ──────────────────────
// export functions must follow: Java_<package_underscored>_<Class>_<method>
// Package: com.edgecache.sdk → com_edgecache_sdk
// Class: EdgeSyncLLM
// Example: nativeInitialize → Java_com_edgecache_sdk_EdgeSyncLLM_nativeInitialize
//
// CGO NOTE
// ─────────
// JNI types (jstring, jfloatArray, jboolean, etc.) are defined in jni.h.
// The Android NDK provides jni.h automatically.
package main

// #cgo CFLAGS: -I${SRCDIR}/../../core
// #include <jni.h>
// #include <stdlib.h>
// #include <string.h>
import "C"

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"react-example/adapter"
	"react-example/cache"
	"react-example/core"
	"react-example/embedding"
)

// ─────────────────────────────────────────────────────────────────────────────
// Global state (one instance per Android process)
// ─────────────────────────────────────────────────────────────────────────────

var (
	globalMu       sync.RWMutex
	globalHNSW     *core.HNSW
	globalStore    *cache.FragmentStore
	globalEncoder  embedding.Encoder
	globalAdapter  adapter.KVAdapter
	globalConfig   bridgeConfig
)

type bridgeConfig struct {
	engine      string
	layerStride int
	model       cache.ModelID
}

// fragmentIndex maps fragment IDs to their embedding vectors for HNSW retrieval.
var (
	fragmentIndex   = make(map[int]string)   // hnsw_id → fragment_id
	fragmentVecs    = make(map[int][]float32) // hnsw_id → embedding
	nextHNSWID      = 1
	fragmentIndexMu sync.RWMutex
)

// ─────────────────────────────────────────────────────────────────────────────
// nativeInitialize
// ─────────────────────────────────────────────────────────────────────────────

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeInitialize
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeInitialize(
	env *C.JNIEnv, obj C.jobject,
	jEngine C.jstring,
	jModelIdJson C.jstring,
	jEmbeddingModelPath C.jstring,
	jDbPath C.jstring,
	jBlobDir C.jstring,
	jLayerStride C.jint,
	jUseNeon C.jboolean,
) C.jboolean {
	engine := jstringToGo(env, jEngine)
	modelIdJson := jstringToGo(env, jModelIdJson)
	embPath := jstringToGo(env, jEmbeddingModelPath)
	dbPath := jstringToGo(env, jDbPath)
	blobDir := jstringToGo(env, jBlobDir)
	layerStride := int(jLayerStride)

	// Parse ModelID
	var model cache.ModelID
	if err := json.Unmarshal([]byte(modelIdJson), &model); err != nil {
		fmt.Printf("jni: parse modelId failed: %v\n", err)
		return C.jboolean(0)
	}

	// Initialize embedding encoder
	enc, err := embedding.NewEncoder(embPath)
	if err != nil {
		fmt.Printf("jni: init encoder failed: %v\n", err)
		return C.jboolean(0)
	}

	// Initialize fragment store
	store, err := cache.NewFragmentStore(dbPath, blobDir)
	if err != nil {
		fmt.Printf("jni: init store failed: %v\n", err)
		return C.jboolean(0)
	}

	// Initialize HNSW index
	hnsw := core.NewHNSW(16, 50)

	globalMu.Lock()
	globalEncoder = enc
	globalStore = store
	globalHNSW = hnsw
	globalConfig = bridgeConfig{
		engine:      engine,
		layerStride: layerStride,
		model:       model,
	}
	globalMu.Unlock()

	fmt.Printf("jni: initialized engine=%s model=%s encoder=%s\n",
		engine, model.String(), enc.Name())

	return C.jboolean(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeEmbed
// ─────────────────────────────────────────────────────────────────────────────

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeEmbed
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeEmbed(
	env *C.JNIEnv, obj C.jobject,
	jText C.jstring,
) C.jfloatArray {
	text := jstringToGo(env, jText)

	globalMu.RLock()
	enc := globalEncoder
	globalMu.RUnlock()

	if enc == nil {
		return nil
	}

	vec, err := enc.Encode(text)
	if err != nil {
		fmt.Printf("jni: embed failed: %v\n", err)
		return nil
	}

	return floatSliceToJFloatArray(env, vec)
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeLookup
// ─────────────────────────────────────────────────────────────────────────────

// LookupResultJNI holds the lookup result returned to Kotlin as a jobject.
// We return a JSON string and let Kotlin deserialize — simpler than constructing
// a Java object from JNI.
type lookupResultJSON struct {
	FragmentID string  `json:"fragmentId"`
	Similarity float32 `json:"similarity"`
	TokenStart int     `json:"tokenStart"`
	TokenEnd   int     `json:"tokenEnd"`
}

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeLookup
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeLookup(
	env *C.JNIEnv, obj C.jobject,
	jEmbedding C.jfloatArray, jK C.jint,
) C.jstring {
	embedding := jfloatArrayToSlice(env, jEmbedding)
	k := int(jK)

	globalMu.RLock()
	hnsw := globalHNSW
	store := globalStore
	globalMu.RUnlock()

	if hnsw == nil || store == nil {
		return nil
	}

	neighbors := hnsw.Search(embedding, k)
	if len(neighbors) == 0 {
		return goStringToJString(env, "null")
	}

	// Find the best neighbor
	best := neighbors[0]
	fragmentIndexMu.RLock()
	fragmentID, ok := fragmentIndex[best.ID]
	bestVec := fragmentVecs[best.ID]
	fragmentIndexMu.RUnlock()

	if !ok || len(bestVec) == 0 {
		return goStringToJString(env, "null")
	}

	// Compute cosine similarity using our stored vectors
	sim := cosineSim(embedding, bestVec)

	// Get fragment metadata for token range
	frag, err := store.Get(fragmentID)
	if err != nil || frag == nil {
		return goStringToJString(env, "null")
	}

	result := lookupResultJSON{
		FragmentID: fragmentID,
		Similarity: sim,
		TokenStart: frag.TokenStart,
		TokenEnd:   frag.TokenEnd,
	}

	jsonBytes, _ := json.Marshal(result)
	return goStringToJString(env, string(jsonBytes))
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeInjectFragment
// ─────────────────────────────────────────────────────────────────────────────

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeInjectFragment
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeInjectFragment(
	env *C.JNIEnv, obj C.jobject,
	jFragmentId C.jstring,
) C.jboolean {
	fragmentID := jstringToGo(env, jFragmentId)

	globalMu.RLock()
	store := globalStore
	a := globalAdapter
	globalMu.RUnlock()

	if store == nil || a == nil {
		return C.jboolean(0)
	}

	frag, err := store.Get(fragmentID)
	if err != nil || frag == nil {
		fmt.Printf("jni: fragment %q not found: %v\n", fragmentID, err)
		return C.jboolean(0)
	}

	// Use CanInjectWithReshape to handle cross-engine fragments transparently
	readyFrag, err := adapter.CanInjectWithReshape(a, frag)
	if err != nil {
		fmt.Printf("jni: cannot inject fragment %q: %v\n", fragmentID, err)
		return C.jboolean(0)
	}

	ctx := context.Background()
	if err := a.InjectFragment(ctx, readyFrag); err != nil {
		fmt.Printf("jni: inject failed: %v\n", err)
		return C.jboolean(0)
	}

	frag.RecordHit()
	_ = store.Store(frag) // update hit count asynchronously

	return C.jboolean(1)
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeGenerateFromPos
// ─────────────────────────────────────────────────────────────────────────────

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeGenerateFromPos
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeGenerateFromPos(
	env *C.JNIEnv, obj C.jobject,
	jPrompt C.jstring,
	jStartTokenPos C.jint,
	jMaxTokens C.jint,
) C.jstring {
	prompt := jstringToGo(env, jPrompt)
	startPos := int(jStartTokenPos)
	maxTokens := int(jMaxTokens)

	globalMu.RLock()
	a := globalAdapter
	globalMu.RUnlock()

	if a == nil {
		return goStringToJString(env, "[adapter not initialized]")
	}

	ctx := context.Background()
	text, _, err := a.Generate(ctx, prompt, startPos, maxTokens)
	if err != nil {
		return goStringToJString(env, fmt.Sprintf("[generate error: %v]", err))
	}

	return goStringToJString(env, text)
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeExtractAndStore
// ─────────────────────────────────────────────────────────────────────────────

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeExtractAndStore
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeExtractAndStore(
	env *C.JNIEnv, obj C.jobject,
	jPrompt C.jstring,
	jEmbedding C.jfloatArray,
) C.jstring {
	prompt := jstringToGo(env, jPrompt)
	embedding := jfloatArrayToSlice(env, jEmbedding)

	globalMu.RLock()
	a := globalAdapter
	store := globalStore
	hnsw := globalHNSW
	cfg := globalConfig
	globalMu.RUnlock()

	if a == nil || store == nil {
		return nil
	}

	ctx := context.Background()
	tokenIDs, err := a.Tokenize(ctx, prompt)
	if err != nil {
		fmt.Printf("jni: tokenize failed: %v\n", err)
		return nil
	}

	if len(tokenIDs) < cache.FragmentGranularityTokens {
		// Too short to be worth caching
		return nil
	}

	frag, err := a.ExtractFragment(
		ctx, tokenIDs,
		0, cfg.model.NumLayers, cfg.layerStride,
		embedding,
	)
	if err != nil {
		fmt.Printf("jni: extract failed: %v\n", err)
		return nil
	}

	if err := store.Store(frag); err != nil {
		fmt.Printf("jni: store failed: %v\n", err)
		return nil
	}

	// Index in HNSW
	fragmentIndexMu.Lock()
	id := nextHNSWID
	nextHNSWID++
	fragmentIndex[id] = frag.ID
	fragmentVecs[id] = embedding
	fragmentIndexMu.Unlock()

	hnsw.Insert(id, embedding)

	return goStringToJString(env, frag.ID)
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeCompact
// ─────────────────────────────────────────────────────────────────────────────

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeCompact
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeCompact(
	env *C.JNIEnv, obj C.jobject,
) C.jstring {
	globalMu.RLock()
	store := globalStore
	globalMu.RUnlock()

	if store == nil {
		return goStringToJString(env, "store not initialized")
	}

	compactor := cache.NewCompactor(store)
	result, err := compactor.Run()
	if err != nil {
		return goStringToJString(env, fmt.Sprintf("compaction error: %v", err))
	}

	summary := fmt.Sprintf(
		"compaction complete in %s: %d duplicates removed, %d fragments merged, %d KB freed",
		result.Duration.Round(time.Millisecond),
		result.DuplicatesRemoved,
		result.FragmentsMerged,
		result.BytesFreed/1024,
	)
	return goStringToJString(env, summary)
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeReshapeFragment
// ─────────────────────────────────────────────────────────────────────────────

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeReshapeFragment
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeReshapeFragment(
	env *C.JNIEnv, obj C.jobject,
	jFragmentId C.jstring,
	jTargetEngine C.jstring,
) C.jstring {
	fragmentID := jstringToGo(env, jFragmentId)
	targetEngine := jstringToGo(env, jTargetEngine)

	globalMu.RLock()
	store := globalStore
	globalMu.RUnlock()

	frag, err := store.Get(fragmentID)
	if err != nil || frag == nil {
		return nil
	}

	reshaped, err := adapter.ReshapeForEngine(frag, targetEngine)
	if err != nil {
		fmt.Printf("jni: reshape failed: %v\n", err)
		return nil
	}

	if err := store.Store(reshaped); err != nil {
		return nil
	}

	return goStringToJString(env, reshaped.ID)
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeGetHotCacheSize / nativeGetPersistentCacheSize
// ─────────────────────────────────────────────────────────────────────────────

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeGetHotCacheSize
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeGetHotCacheSize(
	env *C.JNIEnv, obj C.jobject,
) C.jint {
	globalMu.RLock()
	store := globalStore
	globalMu.RUnlock()
	if store == nil {
		return 0
	}
	hot, _, _ := store.Stats()
	return C.jint(hot)
}

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeGetPersistentCacheSize
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeGetPersistentCacheSize(
	env *C.JNIEnv, obj C.jobject,
) C.jint {
	globalMu.RLock()
	store := globalStore
	globalMu.RUnlock()
	if store == nil {
		return 0
	}
	_, persistent, _ := store.Stats()
	return C.jint(persistent)
}

// ─────────────────────────────────────────────────────────────────────────────
// nativeClose
// ─────────────────────────────────────────────────────────────────────────────

//export Java_com_edgecache_sdk_EdgeSyncLLM_nativeClose
func Java_com_edgecache_sdk_EdgeSyncLLM_nativeClose(
	env *C.JNIEnv, obj C.jobject,
) {
	globalMu.Lock()
	defer globalMu.Unlock()

	if globalEncoder != nil {
		globalEncoder.Close()
		globalEncoder = nil
	}
	if globalStore != nil {
		globalStore.Close()
		globalStore = nil
	}
	if globalAdapter != nil {
		globalAdapter.Close()
		globalAdapter = nil
	}
	globalHNSW = nil
}

// ─────────────────────────────────────────────────────────────────────────────
// JNI helper functions
// ─────────────────────────────────────────────────────────────────────────────

// jstringToGo converts a JNI jstring to a Go string.
func jstringToGo(env *C.JNIEnv, s C.jstring) string {
	if s == nil {
		return ""
	}
	cStr := C.GetStringUTFChars(env, s, nil)
	defer C.ReleaseStringUTFChars(env, s, cStr)
	return C.GoString(cStr)
}

// goStringToJString converts a Go string to a JNI jstring.
func goStringToJString(env *C.JNIEnv, s string) C.jstring {
	cStr := C.CString(s)
	defer C.free(unsafe.Pointer(cStr))
	return C.NewStringUTF(env, cStr)
}

// floatSliceToJFloatArray converts a Go []float32 to a Java float[].
func floatSliceToJFloatArray(env *C.JNIEnv, vec []float32) C.jfloatArray {
	arr := C.NewFloatArray(env, C.jsize(len(vec)))
	if arr == nil {
		return nil
	}
	C.SetFloatArrayRegion(env, arr, 0, C.jsize(len(vec)),
		(*C.jfloat)(unsafe.Pointer(&vec[0])))
	return arr
}

// jfloatArrayToSlice converts a Java float[] to a Go []float32.
func jfloatArrayToSlice(env *C.JNIEnv, arr C.jfloatArray) []float32 {
	length := int(C.GetArrayLength(env, C.jarray(arr)))
	result := make([]float32, length)
	if length > 0 {
		C.GetFloatArrayRegion(env, arr, 0, C.jsize(length),
			(*C.jfloat)(unsafe.Pointer(&result[0])))
	}
	return result
}

// cosineSim computes cosine similarity between two normalized float32 vectors.
func cosineSim(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

// Required for CGO shared library
func main() {}
