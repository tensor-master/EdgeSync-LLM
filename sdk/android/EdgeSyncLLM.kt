package com.edgecache.sdk

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * EdgeSyncLLM — Android SDK for the EdgeCache KV fragment engine.
 *
 * This is a complete rewrite of the original SDK to expose the adapter/ package API:
 *   - KVFragment extraction and injection (replacing naive string cache)
 *   - Real MiniLM-L6-v2 embedding via ONNX Runtime Mobile
 *   - Three-mode inference: EXACT / PARTIAL / MISS with KV tensor reuse
 *   - Persistent fragment store (SQLite WAL + blob files)
 *   - Cross-engine reshape support (llamacpp ↔ onnx)
 *
 * ARCHITECTURE
 * ─────────────
 *   Kotlin (coroutines, Android lifecycle) ──► JNI ──► Go (adapter/, cache/, embedding/)
 *
 * The Go layer is compiled as a shared library: libedgecache.so
 * via gomobile bind or a custom NDK toolchain (see sdk/android/BUILD.md).
 *
 * JNI FUNCTION NAMING
 * ────────────────────
 * All JNI functions in jni_bridge.go follow the pattern:
 *   Java_com_edgecache_sdk_EdgeSyncLLM_native<FunctionName>
 *
 * THREAD SAFETY
 * ──────────────
 * All public methods are safe to call from any coroutine dispatcher.
 * Heavy operations (extract, inject, generate) are dispatched to Dispatchers.Default.
 * Metrics reads are main-thread safe.
 */
class EdgeSyncLLM private constructor(
    private val context: Context,
    private val config: Config
) {

    // ─────────────────────────────────────────────────────────────────────────
    // Configuration
    // ─────────────────────────────────────────────────────────────────────────

    data class Config(
        /** Directory for fragment store DB and blob files. Default: app internal storage. */
        val storeDir: String,

        /** Path to all-MiniLM-L6-v2.ort (ONNX Runtime Mobile format). */
        val embeddingModelPath: String,

        /** Which engine adapter to use: "llamacpp", "mlc", or "onnx". */
        val engine: String = "llamacpp",

        /** Layer stride for fragment extraction. 1 = all layers, 2 = every other layer. */
        val layerStride: Int = 2,

        /** Max tokens to generate per response. */
        val maxGenerateTokens: Int = 200,

        /** Use ARM NEON intrinsics for cosine similarity (ARM64 only). */
        val useNeon: Boolean = true,
    )

    // ─────────────────────────────────────────────────────────────────────────
    // Result types — KV-level cache outcomes
    // ─────────────────────────────────────────────────────────────────────────

    sealed class InferResult {

        /**
         * EXACT HIT (similarity ≥ 0.92): KV fragment covers full prefix.
         * Fragment was injected into the engine; only a short suffix was generated.
         * TTFT is typically 8–15ms.
         */
        data class ExactHit(
            val response: String,
            val similarity: Float,
            val lookupMs: Float,
            val injectMs: Float,
            val fragmentId: String,
            val tokensCached: Int,
        ) : InferResult()

        /**
         * PARTIAL HIT (0.75 ≤ similarity < 0.92): fragment covers shared prefix.
         * Delta tokens were generated from the divergence point.
         * TTFT is typically 100–400ms depending on delta size.
         */
        data class PartialHit(
            val response: String,
            val similarity: Float,
            val lookupMs: Float,
            val injectMs: Float,
            val deltaInferMs: Float,
            val fragmentId: String,
            val tokensCached: Int,
            val tokensDelta: Int,
            val memBandwidthSavedPct: Float,
        ) : InferResult()

        /**
         * MISS (similarity < 0.75): full prefill + generate.
         * New fragment extracted and stored for future reuse.
         * TTFT is typically 1500–2000ms on Cortex-A55.
         */
        data class Miss(
            val response: String,
            val similarity: Float,
            val lookupMs: Float,
            val fullInferMs: Float,
            val newFragmentId: String?,
        ) : InferResult()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Session metrics
    // ─────────────────────────────────────────────────────────────────────────

    data class SessionMetrics(
        val totalQueries: Long,
        val exactHits: Long,
        val partialHits: Long,
        val misses: Long,
        val hitRatePct: Float,
        val avgLatencyMs: Float,
        val totalEnergySavedMah: Float,
        val avgMemBandwidthPct: Float,
        val fragmentsInHotCache: Int,
        val fragmentsInPersistentStore: Int,
    )

    // ─────────────────────────────────────────────────────────────────────────
    // Fragment handle (opaque reference to a KVFragment in Go memory)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Opaque handle to a KVFragment managed by the Go cache layer.
     * Do not hold references beyond the current request — Go may GC the fragment.
     * Use fragmentId to retrieve it from the store in subsequent requests.
     */
    data class FragmentHandle(
        val fragmentId: String,
        val tokenStart: Int,
        val tokenEnd: Int,
        val layerStart: Int,
        val layerEnd: Int,
        val layerStride: Int,
        val engine: String,
        val sizeKb: Int,
        val hitCount: Int,
        val expiresAtEpochSec: Long,
    )

    // ─────────────────────────────────────────────────────────────────────────
    // Initialization
    // ─────────────────────────────────────────────────────────────────────────

    private var initialized = false

    init {
        try {
            System.loadLibrary("edgecache")
            Log.i(TAG, "libedgecache.so loaded")
        } catch (e: UnsatisfiedLinkError) {
            Log.e(TAG, "Failed to load libedgecache.so: ${e.message}")
            Log.w(TAG, "Running in stub mode — no real inference will occur")
        }
    }

    /**
     * Initializes the Go engine: loads the embedding model, opens the fragment
     * store, and registers the engine adapter.
     *
     * Must be called before any infer() or extract() calls.
     * Safe to call multiple times (idempotent).
     *
     * @param modelId Serialized ModelID JSON (use ModelIdBuilder below).
     * @return true on success, false if initialization failed.
     */
    suspend fun initialize(modelId: String): Boolean = withContext(Dispatchers.Default) {
        if (initialized) return@withContext true

        val dbPath = "${config.storeDir}/fragments.db"
        val blobDir = "${config.storeDir}/blobs"
        File(config.storeDir).mkdirs()

        val ok = nativeInitialize(
            engine = config.engine,
            modelIdJson = modelId,
            embeddingModelPath = config.embeddingModelPath,
            dbPath = dbPath,
            blobDir = blobDir,
            layerStride = config.layerStride,
            useNeon = config.useNeon,
        )
        initialized = ok
        if (!ok) Log.e(TAG, "nativeInitialize failed")
        ok
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Core inference pipeline
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Runs inference with KV fragment cache.
     *
     * Pipeline:
     *   1. Embed prompt (MiniLM-L6-v2, ~8ms)
     *   2. HNSW lookup (~3ms)
     *   3a. EXACT HIT  → inject fragment + generate suffix (~8ms total)
     *   3b. PARTIAL HIT → inject prefix + generate delta (~100-400ms)
     *   3c. MISS        → full prefill + generate + extract fragment (~1800ms)
     *
     * @param prompt The user prompt.
     * @param fallback Called on MISS for the actual LLM inference.
     *                 Receives the full prompt; returns the generated text.
     */
    suspend fun infer(
        prompt: String,
        fallback: suspend (prompt: String) -> String,
    ): InferResult = withContext(Dispatchers.Default) {

        checkInitialized()

        val t0 = System.currentTimeMillis()

        // Step 1: Embed prompt
        val embedding = nativeEmbed(prompt)

        // Step 2: HNSW lookup
        val lookupResult = nativeLookup(embedding, k = 1)
        val lookupMs = (System.currentTimeMillis() - t0).toFloat()

        when {
            lookupResult != null && lookupResult.similarity >= THRESHOLD_EXACT -> {
                // ── EXACT HIT ────────────────────────────────────────────────
                val t1 = System.currentTimeMillis()
                val injectOk = nativeInjectFragment(lookupResult.fragmentId)
                val injectMs = (System.currentTimeMillis() - t1).toFloat()

                if (!injectOk) {
                    // Injection failed — fall through to MISS
                    return@withContext handleMiss(prompt, lookupResult.similarity, lookupMs, fallback)
                }

                val t2 = System.currentTimeMillis()
                val response = nativeGenerateFromPos(
                    prompt = prompt,
                    startTokenPos = lookupResult.tokenEnd,
                    maxTokens = config.maxGenerateTokens,
                )
                val generateMs = (System.currentTimeMillis() - t2).toFloat()

                updateMetrics(exactHit = true, latencyMs = lookupMs + injectMs + generateMs)

                InferResult.ExactHit(
                    response = response,
                    similarity = lookupResult.similarity,
                    lookupMs = lookupMs,
                    injectMs = injectMs,
                    fragmentId = lookupResult.fragmentId,
                    tokensCached = lookupResult.tokenEnd - lookupResult.tokenStart,
                )
            }

            lookupResult != null && lookupResult.similarity >= THRESHOLD_PARTIAL -> {
                // ── PARTIAL HIT ───────────────────────────────────────────────
                val t1 = System.currentTimeMillis()
                val injectOk = nativeInjectFragment(lookupResult.fragmentId)
                val injectMs = (System.currentTimeMillis() - t1).toFloat()

                if (!injectOk) {
                    return@withContext handleMiss(prompt, lookupResult.similarity, lookupMs, fallback)
                }

                val t2 = System.currentTimeMillis()
                val response = nativeGenerateFromPos(
                    prompt = prompt,
                    startTokenPos = lookupResult.tokenEnd,
                    maxTokens = config.maxGenerateTokens,
                )
                val deltaMs = (System.currentTimeMillis() - t2).toFloat()

                val tokensCached = lookupResult.tokenEnd - lookupResult.tokenStart
                val tokensDelta = nativeCountTokens(prompt) - tokensCached
                val memSavedPct = tokensCached.toFloat() / nativeCountTokens(prompt).toFloat() * 100f

                updateMetrics(partialHit = true, latencyMs = lookupMs + injectMs + deltaMs,
                    memBwPct = 1f - (tokensCached.toFloat() / nativeCountTokens(prompt)))

                InferResult.PartialHit(
                    response = response,
                    similarity = lookupResult.similarity,
                    lookupMs = lookupMs,
                    injectMs = injectMs,
                    deltaInferMs = deltaMs,
                    fragmentId = lookupResult.fragmentId,
                    tokensCached = tokensCached,
                    tokensDelta = tokensDelta.coerceAtLeast(0),
                    memBandwidthSavedPct = memSavedPct,
                )
            }

            else -> {
                // ── MISS ──────────────────────────────────────────────────────
                handleMiss(prompt, lookupResult?.similarity ?: 0f, lookupMs, fallback)
            }
        }
    }

    private suspend fun handleMiss(
        prompt: String,
        similarity: Float,
        lookupMs: Float,
        fallback: suspend (prompt: String) -> String,
    ): InferResult.Miss {
        val t = System.currentTimeMillis()
        val response = fallback(prompt)
        val inferMs = (System.currentTimeMillis() - t).toFloat()

        // Extract and store fragment asynchronously
        val fragmentId = nativeExtractAndStore(prompt, embedding = nativeEmbed(prompt))

        updateMetrics(miss = true, latencyMs = lookupMs + inferMs, memBwPct = 1f)

        return InferResult.Miss(
            response = response,
            similarity = similarity,
            lookupMs = lookupMs,
            fullInferMs = inferMs,
            newFragmentId = fragmentId,
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fragment management API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Manually extracts a KV fragment for the given prompt prefix.
     * Useful for pre-warming the cache with known system prompts.
     *
     * @param promptPrefix The text whose KV tensors should be cached.
     * @return FragmentHandle on success, null on failure.
     */
    suspend fun extractFragment(promptPrefix: String): FragmentHandle? =
        withContext(Dispatchers.Default) {
            checkInitialized()
            val embedding = nativeEmbed(promptPrefix)
            val fragmentId = nativeExtractAndStore(promptPrefix, embedding)
                ?: return@withContext null
            nativeGetFragmentHandle(fragmentId)
        }

    /**
     * Retrieves a fragment handle by ID.
     * Use this to inspect cached fragments or verify pre-warming succeeded.
     */
    suspend fun getFragment(fragmentId: String): FragmentHandle? =
        withContext(Dispatchers.Default) {
            checkInitialized()
            nativeGetFragmentHandle(fragmentId)
        }

    /**
     * Lists all non-expired fragments in the hot cache.
     * Sorted by hit count descending.
     */
    suspend fun listFragments(): List<FragmentHandle> =
        withContext(Dispatchers.Default) {
            checkInitialized()
            nativeListFragments() ?: emptyList()
        }

    /**
     * Runs the compaction pass: deduplicates and merges adjacent fragments.
     * Should be called periodically (e.g. on app background / low battery).
     *
     * @return Summary string describing what was compacted.
     */
    suspend fun compact(): String = withContext(Dispatchers.Default) {
        checkInitialized()
        nativeCompact() ?: "compaction not available"
    }

    /**
     * Reshapes a fragment from one engine layout to another.
     * Example: reshape a llamacpp fragment to onnx format for cross-engine injection.
     *
     * @param fragmentId Source fragment ID.
     * @param targetEngine Target engine name: "llamacpp", "onnx", or "mlc".
     * @return New fragment ID (reshaped copy), or null if reshape not supported.
     */
    suspend fun reshapeFragment(fragmentId: String, targetEngine: String): String? =
        withContext(Dispatchers.Default) {
            checkInitialized()
            nativeReshapeFragment(fragmentId, targetEngine)
        }

    // ─────────────────────────────────────────────────────────────────────────
    // Metrics
    // ─────────────────────────────────────────────────────────────────────────

    private var totalQueries = 0L
    private var exactHits = 0L
    private var partialHits = 0L
    private var misses = 0L
    private var totalLatencyMs = 0.0
    private var totalEnergySavedMah = 0.0f
    private var totalMemBwFraction = 0.0

    private fun updateMetrics(
        exactHit: Boolean = false,
        partialHit: Boolean = false,
        miss: Boolean = false,
        latencyMs: Float = 0f,
        memBwPct: Float = 0f,
    ) {
        totalQueries++
        totalLatencyMs += latencyMs
        totalMemBwFraction += memBwPct
        when {
            exactHit  -> { exactHits++;   totalEnergySavedMah += ENERGY_SAVED_EXACT_MAH  }
            partialHit -> { partialHits++; totalEnergySavedMah += ENERGY_SAVED_PARTIAL_MAH }
            miss      -> misses++
        }
    }

    fun getMetrics(): SessionMetrics {
        val hotCount = nativeGetHotCacheSize()
        val persistentCount = nativeGetPersistentCacheSize()
        return SessionMetrics(
            totalQueries = totalQueries,
            exactHits = exactHits,
            partialHits = partialHits,
            misses = misses,
            hitRatePct = if (totalQueries > 0) (exactHits + partialHits).toFloat() / totalQueries * 100f else 0f,
            avgLatencyMs = if (totalQueries > 0) (totalLatencyMs / totalQueries).toFloat() else 0f,
            totalEnergySavedMah = totalEnergySavedMah,
            avgMemBandwidthPct = if (totalQueries > 0) (totalMemBwFraction / totalQueries * 100).toFloat() else 0f,
            fragmentsInHotCache = hotCount,
            fragmentsInPersistentStore = persistentCount,
        )
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Closes the fragment store and releases native resources.
     * Must be called when the SDK is no longer needed (e.g. in onDestroy).
     */
    fun close() {
        nativeClose()
        initialized = false
        INSTANCE = null
    }

    private fun checkInitialized() {
        check(initialized) {
            "EdgeSyncLLM is not initialized. Call initialize() first."
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // JNI native declarations
    // All implemented in sdk/android/jni_bridge.go (exported via CGO/gomobile)
    // ─────────────────────────────────────────────────────────────────────────

    private external fun nativeInitialize(
        engine: String,
        modelIdJson: String,
        embeddingModelPath: String,
        dbPath: String,
        blobDir: String,
        layerStride: Int,
        useNeon: Boolean,
    ): Boolean

    private external fun nativeEmbed(text: String): FloatArray

    private data class LookupResult(
        val fragmentId: String,
        val similarity: Float,
        val tokenStart: Int,
        val tokenEnd: Int,
    )

    private external fun nativeLookup(embedding: FloatArray, k: Int): LookupResult?

    private external fun nativeInjectFragment(fragmentId: String): Boolean

    private external fun nativeGenerateFromPos(
        prompt: String,
        startTokenPos: Int,
        maxTokens: Int,
    ): String

    private external fun nativeExtractAndStore(
        prompt: String,
        embedding: FloatArray,
    ): String?

    private external fun nativeGetFragmentHandle(fragmentId: String): FragmentHandle?

    private external fun nativeListFragments(): List<FragmentHandle>?

    private external fun nativeCountTokens(text: String): Int

    private external fun nativeCompact(): String?

    private external fun nativeReshapeFragment(fragmentId: String, targetEngine: String): String?

    private external fun nativeGetHotCacheSize(): Int

    private external fun nativeGetPersistentCacheSize(): Int

    private external fun nativeClose()

    // ─────────────────────────────────────────────────────────────────────────
    // ModelID builder — type-safe construction of the JSON passed to nativeInitialize
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Builder for ModelID JSON.
     *
     * Usage:
     *   val modelId = EdgeSyncLLM.ModelId(
     *       architecture = "qwen",
     *       name = "Qwen2.5-0.5B",
     *       quantization = "Q4_K_M",
     *       contextLength = 4096,
     *       headDim = 64,
     *       numKVHeads = 8,
     *       numLayers = 24,
     *   ).toJson()
     */
    data class ModelId(
        val architecture: String,
        val name: String,
        val quantization: String,
        val contextLength: Int,
        val headDim: Int,
        val numKVHeads: Int,
        val numLayers: Int,
    ) {
        fun toJson(): String = """
            {
              "Architecture": "$architecture",
              "Name": "$name",
              "Quantization": "$quantization",
              "ContextLength": $contextLength,
              "HeadDim": $headDim,
              "NumKVHeads": $numKVHeads,
              "NumLayers": $numLayers
            }
        """.trimIndent()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Companion object — singleton factory
    // ─────────────────────────────────────────────────────────────────────────

    companion object {

        private const val TAG = "EdgeSyncLLM"

        private const val THRESHOLD_EXACT   = 0.92f
        private const val THRESHOLD_PARTIAL = 0.75f

        // Energy estimates per request type (Cortex-A55, Qwen2.5-0.5B, Q4_K_M)
        // Based on measured 850mA active draw, 18.4ms/token generate, 6.8ms/token prefill
        private const val ENERGY_SAVED_EXACT_MAH   = 0.42f  // ~1800ms prefill avoided
        private const val ENERGY_SAVED_PARTIAL_MAH = 0.21f  // ~900ms prefill avoided (avg)

        @Volatile
        private var INSTANCE: EdgeSyncLLM? = null

        /**
         * Returns (or creates) the singleton EdgeSyncLLM instance.
         *
         * Example:
         *   val sdk = EdgeSyncLLM.getInstance(
         *       context = this,
         *       config = EdgeSyncLLM.Config(
         *           storeDir = "${filesDir}/edgecache",
         *           embeddingModelPath = "${filesDir}/models/all-MiniLM-L6-v2.ort",
         *           engine = "llamacpp",
         *       )
         *   )
         *   sdk.initialize(modelId.toJson())
         */
        fun getInstance(context: Context, config: Config): EdgeSyncLLM =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: EdgeSyncLLM(context.applicationContext, config).also {
                    INSTANCE = it
                }
            }
    }
}
