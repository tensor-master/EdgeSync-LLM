package cache

import (
	"database/sql"
	"fmt"
	"time"
)

// SQL schema scripts for EdgeSync-LLM semantic cache storage.
// Leverages SQLite WAL (Write-Ahead Logging) to ensure <2ms lookups during concurrent operations.
const (
	// Enable WAL journaling, synchronous NORMAL, and busy timeout for high concurrent performance
	PRAGMA_SETUP = `
		PRAGMA journal_mode = WAL;
		PRAGMA synchronous = NORMAL;
		PRAGMA foreign_keys = ON;
		PRAGMA busy_timeout = 5000;
	`

	// Schema defining primary semantic entries containing prompt strings, float16 embedding vectors, and metadata
	CREATE_CACHE_ENTRIES_TABLE = `
		CREATE TABLE IF NOT EXISTS cache_entries (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			prompt TEXT NOT NULL UNIQUE,
			embedding BLOB NOT NULL,       -- Flat float16 or float32 array (384 dims, 768 or 1536 bytes)
			response TEXT NOT NULL,
			hit_count INTEGER DEFAULT 0,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			expires_at DATETIME NOT NULL
		);
	`

	// Schema defining differential updates (deltas) for PARTIAL hit state reconstruction
	CREATE_CACHE_DELTAS_TABLE = `
		CREATE TABLE IF NOT EXISTS cache_deltas (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			parent_entry_id INTEGER NOT NULL,
			divergence_idx INTEGER NOT NULL, -- index in characters or words where divergence started
			delta_response TEXT NOT NULL,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY(parent_entry_id) REFERENCES cache_entries(id) ON DELETE CASCADE
		);
	`

	// Schema defining execution job performance and energy telemetry metrics
	CREATE_JOB_METRICS_TABLE = `
		CREATE TABLE IF NOT EXISTS job_metrics (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
			prompt TEXT NOT NULL,
			cache_status TEXT NOT NULL,      -- EXACT, PARTIAL, MISS
			lookup_time_ms REAL NOT NULL,
			inference_time_ms REAL NOT NULL,
			npu_reduction_pct REAL NOT NULL, -- percentage of NPU reduction
			energy_used_mah REAL NOT NULL,   -- energy usage captured from Android power interface
			tokens_generated INTEGER NOT NULL
		);
	`

	// Optimization Indexes to accelerate lookup speeds and analytics query execution
	CREATE_INDEXES = `
		CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache_entries(expires_at);
		CREATE INDEX IF NOT EXISTS idx_cache_hits ON cache_entries(hit_count DESC);
		CREATE INDEX IF NOT EXISTS idx_deltas_parent_id ON cache_deltas(parent_entry_id);
		CREATE INDEX IF NOT EXISTS idx_metrics_status ON job_metrics(cache_status);
	`
)

// CacheEntry maps rows from the `cache_entries` table.
type CacheEntry struct {
	ID        int64
	Prompt    string
	Embedding []byte // Raw float16/float32 array representation
	Response  string
	HitCount  int
	CreatedAt time.Time
	ExpiresAt time.Time
}

// CacheDelta maps rows from the `cache_deltas` table.
type CacheDelta struct {
	ID            int64
	ParentEntryID int64
	DivergenceIdx int
	DeltaResponse string
	CreatedAt     time.Time
}

// JobMetric maps rows from the `job_metrics` table.
type JobMetric struct {
	ID               int64
	Timestamp        time.Time
	Prompt           string
	CacheStatus      string // "EXACT", "PARTIAL", "MISS"
	LookupTimeMs     float64
	InferenceTimeMs   float64
	NpuReductionPct  float64
	EnergyUsedMah    float64
	TokensGenerated  int
}

// InitDatabase executes SQL statements to configure SQLite WAL and provision database structures.
func InitDatabase(db *sql.DB) error {
	// Apply performance pragmas first
	if _, err := db.Exec(PRAGMA_SETUP); err != nil {
		return fmt.Errorf("failed to apply SQLite pragmas: %w", err)
	}

	// Create tables in correct dependency order
	tables := []struct {
		name string
		ddl  string
	}{
		{"cache_entries", CREATE_CACHE_ENTRIES_TABLE},
		{"cache_deltas", CREATE_CACHE_DELTAS_TABLE},
		{"job_metrics", CREATE_JOB_METRICS_TABLE},
	}

	for _, t := range tables {
		if _, err := db.Exec(t.ddl); err != nil {
			return fmt.Errorf("failed to create table %s: %w", t.name, err)
		}
	}

	// Create indexes
	if _, err := db.Exec(CREATE_INDEXES); err != nil {
		return fmt.Errorf("failed to create cache indexes: %w", err)
	}

	return nil
}
