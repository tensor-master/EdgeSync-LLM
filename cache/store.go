// Package cache — Persistent fragment store backed by SQLite WAL.
//
// DESIGN
// ───────
// Fragments have two tiers of storage:
//
//   TIER 1 — IN-MEMORY (hot)
//     The HNSW index + a sync.Map of id → *KVFragment.
//     Sub-millisecond lookup. Lost on process restart.
//     Capacity: ~50-100 fragments before RAM pressure on Android (≈300-600 MB).
//
//   TIER 2 — PERSISTENT (warm)
//     SQLite WAL database at a configurable path (default: /data/edgecache/fragments.db).
//     Survives restart. Load time: ~5ms per fragment (disk read + deserialize).
//     Only fragments with HitCount >= HitThresholdPromote are persisted.
//
// SCHEMA
// ───────
// The existing schema.go (cache_entries, cache_deltas, job_metrics) stores
// prompt strings and response text — the old "naive cache" design.
// This file adds a NEW table `kv_fragments` that stores the actual tensor blobs.
// Both schemas coexist; this is backwards-compatible.
//
// SQLITE BLOB STORAGE STRATEGY
// ──────────────────────────────
// SQLite stores BLOBs inline up to page_size (default 4KB).
// KVFragment tensors are 6-24 MB — far above the page size.
// Strategy: store tensor blobs in a separate file alongside the DB,
// referenced by filename in the `blob_path` column.
// The DB row holds all metadata (token range, model, TTL, hit count).
// This avoids SQLite page fragmentation and enables mmap() on the blob file
// for zero-copy injection on Linux/Android.
//
// EVICTION
// ─────────
// On startup, expired fragments are purged from both the DB and blob files.
// During runtime, an LRU eviction runs when fragment count exceeds HitThresholdEvict.
// The eviction policy: delete the fragment with the lowest HitCount × recency score.
package cache

import (
	"database/sql"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3" // SQLite driver
)

// ─────────────────────────────────────────────────────────────────────────────
// Schema extension — kv_fragments table
// ─────────────────────────────────────────────────────────────────────────────

const createKVFragmentsTable = `
CREATE TABLE IF NOT EXISTS kv_fragments (
    id              TEXT PRIMARY KEY,       -- KVFragment.ID (hex string)
    model_hash      TEXT NOT NULL,          -- ModelID.Hash() — 8-char hex
    model_json      TEXT NOT NULL,          -- ModelID serialized as JSON
    token_start     INTEGER NOT NULL,
    token_end       INTEGER NOT NULL,
    layer_start     INTEGER NOT NULL,
    layer_end       INTEGER NOT NULL,
    layer_stride    INTEGER NOT NULL,
    token_ids_blob  BLOB NOT NULL,          -- []int32 packed as little-endian bytes
    content_hash    TEXT NOT NULL,
    embedding_blob  BLOB NOT NULL,          -- []float32 packed as little-endian bytes
    engine          TEXT NOT NULL,
    engine_version  TEXT NOT NULL,
    keys_path       TEXT NOT NULL,          -- path to Keys blob file
    values_path     TEXT NOT NULL,          -- path to Values blob file
    hit_count       INTEGER NOT NULL DEFAULT 0,
    created_at      DATETIME NOT NULL,
    expires_at      DATETIME NOT NULL,
    last_used_at    DATETIME NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kv_model_token
    ON kv_fragments(model_hash, token_start, token_end);

CREATE INDEX IF NOT EXISTS idx_kv_expires
    ON kv_fragments(expires_at);

CREATE INDEX IF NOT EXISTS idx_kv_hits
    ON kv_fragments(hit_count DESC);
`

// ─────────────────────────────────────────────────────────────────────────────
// FragmentStore — the persistent + in-memory cache layer
// ─────────────────────────────────────────────────────────────────────────────

// FragmentStore manages the two-tier fragment storage:
// in-memory sync.Map for hot fragments + SQLite WAL for persistent fragments.
type FragmentStore struct {
	db      *sql.DB
	blobDir string        // directory where tensor blob files are stored
	hot     sync.Map      // id string → *KVFragment (in-memory tier)
	mu      sync.RWMutex  // guards count and eviction
	count   int
}

// NewFragmentStore opens (or creates) the SQLite database at dbPath and
// initializes the kv_fragments schema. blobDir is where tensor blob files
// are written; it is created if it does not exist.
func NewFragmentStore(dbPath, blobDir string) (*FragmentStore, error) {
	if err := os.MkdirAll(filepath.Dir(dbPath), 0755); err != nil {
		return nil, fmt.Errorf("fragment store: cannot create db dir: %w", err)
	}
	if err := os.MkdirAll(blobDir, 0755); err != nil {
		return nil, fmt.Errorf("fragment store: cannot create blob dir: %w", err)
	}

	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_synchronous=NORMAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("fragment store: open db: %w", err)
	}

	if _, err := db.Exec(createKVFragmentsTable); err != nil {
		db.Close()
		return nil, fmt.Errorf("fragment store: create schema: %w", err)
	}

	store := &FragmentStore{db: db, blobDir: blobDir}

	// Purge expired fragments on startup
	if err := store.purgeExpired(); err != nil {
		// Non-fatal: log and continue
		fmt.Printf("fragment store: startup purge warning: %v\n", err)
	}

	return store, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Store — write a fragment to both tiers
// ─────────────────────────────────────────────────────────────────────────────

// Store saves a fragment to the in-memory hot cache.
// If the fragment has HitCount >= HitThresholdPromote, it is also persisted
// to SQLite + blob files.
//
// Thread-safe.
func (s *FragmentStore) Store(f *KVFragment) error {
	// Always store in hot tier
	s.hot.Store(f.ID, f)
	s.mu.Lock()
	s.count++
	shouldEvict := s.count > HitThresholdEvict
	s.mu.Unlock()

	if shouldEvict {
		go s.evictLRU()
	}

	// Persist to SQLite only if fragment qualifies
	if f.HitCount < HitThresholdPromote {
		return nil
	}

	return s.persist(f)
}

// persist writes the fragment metadata to SQLite and tensor blobs to disk.
func (s *FragmentStore) persist(f *KVFragment) error {
	// Write tensor blobs to files
	keysPath := filepath.Join(s.blobDir, f.ID+".keys.bin")
	valsPath := filepath.Join(s.blobDir, f.ID+".vals.bin")

	// Use atomic write (write-temp-rename) to prevent partial blobs on crash.
	// See cache/atomic_write.go for full rationale.
	if err := atomicWriteBlob(keysPath, f.Keys); err != nil {
		return fmt.Errorf("persist: write keys blob: %w", err)
	}
	if err := atomicWriteBlob(valsPath, f.Values); err != nil {
		// Keys blob already written — clean it up before returning error
		os.Remove(keysPath)
		return fmt.Errorf("persist: write vals blob: %w", err)
	}

	// Serialize model ID as JSON
	modelJSON, err := json.Marshal(f.Model)
	if err != nil {
		return fmt.Errorf("persist: marshal model: %w", err)
	}

	// Pack token IDs as little-endian bytes
	tokenBlob := make([]byte, len(f.TokenIDs)*4)
	for i, tok := range f.TokenIDs {
		tokenBlob[i*4] = byte(tok)
		tokenBlob[i*4+1] = byte(tok >> 8)
		tokenBlob[i*4+2] = byte(tok >> 16)
		tokenBlob[i*4+3] = byte(tok >> 24)
	}

	// Pack embedding as little-endian float32 bytes
	embBlob := float32SliceToBytesStore(f.EmbeddingVector)

	_, err = s.db.Exec(`
		INSERT OR REPLACE INTO kv_fragments
		(id, model_hash, model_json, token_start, token_end,
		 layer_start, layer_end, layer_stride,
		 token_ids_blob, content_hash, embedding_blob,
		 engine, engine_version, keys_path, values_path,
		 hit_count, created_at, expires_at, last_used_at)
		VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		f.ID, f.Model.Hash(), string(modelJSON),
		f.TokenStart, f.TokenEnd,
		f.LayerStart, f.LayerEnd, f.LayerStride,
		tokenBlob, f.ContentHash, embBlob,
		f.Engine, f.EngineVersion, keysPath, valsPath,
		f.HitCount,
		f.CreatedAt.UTC().Format(time.RFC3339),
		f.ExpiresAt.UTC().Format(time.RFC3339),
		f.LastUsedAt.UTC().Format(time.RFC3339),
	)
	return err
}

// ─────────────────────────────────────────────────────────────────────────────
// Get — retrieve a fragment by ID
// ─────────────────────────────────────────────────────────────────────────────

// Get retrieves a fragment by ID. Checks the hot tier first, then SQLite.
// Returns nil, nil if the fragment does not exist.
func (s *FragmentStore) Get(id string) (*KVFragment, error) {
	// Hot tier
	if v, ok := s.hot.Load(id); ok {
		f := v.(*KVFragment)
		if f.IsExpired() {
			s.hot.Delete(id)
			return nil, nil
		}
		f.RecordHit()
		return f, nil
	}

	// Cold tier — load from SQLite + blob files
	f, err := s.loadFromDB(id)
	if err != nil {
		return nil, err
	}
	if f == nil {
		return nil, nil
	}

	// Warm up the hot tier
	s.hot.Store(f.ID, f)
	f.RecordHit()

	// Update hit count in DB asynchronously
	go s.updateHitCount(f.ID, f.HitCount, f.LastUsedAt)

	return f, nil
}

// loadFromDB loads a fragment's metadata from SQLite and tensor blobs from disk.
func (s *FragmentStore) loadFromDB(id string) (*KVFragment, error) {
	row := s.db.QueryRow(`
		SELECT id, model_json, token_start, token_end,
		       layer_start, layer_end, layer_stride,
		       token_ids_blob, content_hash, embedding_blob,
		       engine, engine_version, keys_path, values_path,
		       hit_count, created_at, expires_at, last_used_at
		FROM kv_fragments WHERE id = ?`, id)

	var (
		modelJSON    string
		tokenBlob    []byte
		embBlob      []byte
		keysPath     string
		valsPath     string
		createdAtStr string
		expiresAtStr string
		lastUsedStr  string
		f            KVFragment
	)

	err := row.Scan(
		&f.ID, &modelJSON, &f.TokenStart, &f.TokenEnd,
		&f.LayerStart, &f.LayerEnd, &f.LayerStride,
		&tokenBlob, &f.ContentHash, &embBlob,
		&f.Engine, &f.EngineVersion, &keysPath, &valsPath,
		&f.HitCount, &createdAtStr, &expiresAtStr, &lastUsedStr,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("loadFromDB: scan: %w", err)
	}

	// Deserialize model
	if err := json.Unmarshal([]byte(modelJSON), &f.Model); err != nil {
		return nil, fmt.Errorf("loadFromDB: unmarshal model: %w", err)
	}

	// Deserialize timestamps
	f.CreatedAt, _ = time.Parse(time.RFC3339, createdAtStr)
	f.ExpiresAt, _ = time.Parse(time.RFC3339, expiresAtStr)
	f.LastUsedAt, _ = time.Parse(time.RFC3339, lastUsedStr)

	if f.IsExpired() {
		go s.deleteFromDB(id, keysPath, valsPath)
		return nil, nil
	}

	// Deserialize token IDs
	f.TokenIDs = make([]int32, len(tokenBlob)/4)
	for i := range f.TokenIDs {
		f.TokenIDs[i] = int32(tokenBlob[i*4]) |
			int32(tokenBlob[i*4+1])<<8 |
			int32(tokenBlob[i*4+2])<<16 |
			int32(tokenBlob[i*4+3])<<24
	}

	// Deserialize embedding
	f.EmbeddingVector = bytesToFloat32SliceStore(embBlob)

	// Load tensor blobs from disk (lazy: only Keys/Values, not hot-cached metadata)
	// Validate blob sizes before reading — catch truncated files from crashes.
	expectedBlobBytes := f.NumLayersCovered() * f.TokenSpan() * f.Model.NumKVHeads * f.Model.HeadDim * 4
	if err := validateBlobSize(keysPath, expectedBlobBytes); err != nil {
		// Corrupted blob — delete and treat as miss
		go s.deleteFromDB(f.ID, keysPath, valsPath)
		return nil, nil
	}
	if err := validateBlobSize(valsPath, expectedBlobBytes); err != nil {
		go s.deleteFromDB(f.ID, keysPath, valsPath)
		return nil, nil
	}

	f.Keys, err = os.ReadFile(keysPath)
	if err != nil {
		return nil, fmt.Errorf("loadFromDB: read keys blob %q: %w", keysPath, err)
	}
	f.Values, err = os.ReadFile(valsPath)
	if err != nil {
		return nil, fmt.Errorf("loadFromDB: read vals blob %q: %w", valsPath, err)
	}

	return &f, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// QueryByTokenRange — find fragments covering a token range for a model
// ─────────────────────────────────────────────────────────────────────────────

// QueryByTokenRange returns all non-expired fragments for the given model
// that overlap the token range [tokenStart, tokenEnd).
// Results are ordered by hit_count DESC (most reused first).
func (s *FragmentStore) QueryByTokenRange(modelHash string, tokenStart, tokenEnd int) ([]*KVFragment, error) {
	rows, err := s.db.Query(`
		SELECT id FROM kv_fragments
		WHERE model_hash = ?
		  AND token_start <= ? AND token_end >= ?
		  AND expires_at > ?
		ORDER BY hit_count DESC
		LIMIT 10`,
		modelHash, tokenEnd, tokenStart,
		time.Now().UTC().Format(time.RFC3339),
	)
	if err != nil {
		return nil, fmt.Errorf("QueryByTokenRange: %w", err)
	}
	defer rows.Close()

	var results []*KVFragment
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			continue
		}
		f, err := s.Get(id)
		if err != nil || f == nil {
			continue
		}
		results = append(results, f)
	}
	return results, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Maintenance
// ─────────────────────────────────────────────────────────────────────────────

// purgeExpired deletes all expired fragment rows and their blob files.
func (s *FragmentStore) purgeExpired() error {
	rows, err := s.db.Query(`
		SELECT id, keys_path, values_path FROM kv_fragments
		WHERE expires_at <= ?`,
		time.Now().UTC().Format(time.RFC3339),
	)
	if err != nil {
		return err
	}
	defer rows.Close()

	var toDelete []struct{ id, keysPath, valsPath string }
	for rows.Next() {
		var r struct{ id, keysPath, valsPath string }
		if err := rows.Scan(&r.id, &r.keysPath, &r.valsPath); err == nil {
			toDelete = append(toDelete, r)
		}
	}

	for _, r := range toDelete {
		s.deleteFromDB(r.id, r.keysPath, r.valsPath)
		s.hot.Delete(r.id)
	}
	return nil
}

func (s *FragmentStore) deleteFromDB(id, keysPath, valsPath string) {
	// Safe ordering: delete DB row FIRST, then files.
	// If killed after DB delete but before file delete, orphanSweep() cleans up.
	// Reverse ordering would leave DB rows pointing to missing files (hard error).
	s.db.Exec(`DELETE FROM kv_fragments WHERE id = ?`, id)
	os.Remove(keysPath)
	os.Remove(valsPath)
}

// evictLRU removes the least recently used fragment from the hot tier.
func (s *FragmentStore) evictLRU() {
	s.mu.Lock()
	defer s.mu.Unlock()

	var oldestID string
	var oldestTime time.Time

	s.hot.Range(func(key, val interface{}) bool {
		f := val.(*KVFragment)
		if oldestID == "" || f.LastUsedAt.Before(oldestTime) {
			oldestID = f.ID
			oldestTime = f.LastUsedAt
		}
		return true
	})

	if oldestID != "" {
		s.hot.Delete(oldestID)
		s.count--
	}
}

func (s *FragmentStore) updateHitCount(id string, count int, lastUsed time.Time) {
	s.db.Exec(`UPDATE kv_fragments SET hit_count = ?, last_used_at = ? WHERE id = ?`,
		count, lastUsed.UTC().Format(time.RFC3339), id)
}

// Close closes the SQLite database connection.
func (s *FragmentStore) Close() error {
	return s.db.Close()
}

// Stats returns a summary of the store state for monitoring.
func (s *FragmentStore) Stats() (hot int, persistent int, err error) {
	s.hot.Range(func(_, _ interface{}) bool { hot++; return true })
	row := s.db.QueryRow(`SELECT COUNT(*) FROM kv_fragments WHERE expires_at > ?`,
		time.Now().UTC().Format(time.RFC3339))
	err = row.Scan(&persistent)
	return
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization helpers (local to avoid naming collision with adapter package)
// ─────────────────────────────────────────────────────────────────────────────

func float32SliceToBytesStore(src []float32) []byte {
	out := make([]byte, len(src)*4)
	for i, v := range src {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func bytesToFloat32SliceStore(src []byte) []float32 {
	n := len(src) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(src[i*4:]))
	}
	return out
}
