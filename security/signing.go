// Package security — Cache poisoning prevention via HMAC payload signing.
//
// PROBLEM: Semantic cache poisoning
// ────────────────────────────────────
// EdgeSync-LLM supports differential sync: KV fragment deltas can be received
// from a server or peer node and injected into the local cache. If these payloads
// are not authenticated, an attacker can:
//
//   1. INJECT MALICIOUS FRAGMENTS: craft a fragment with a high similarity score
//      to a legitimate prompt, but containing tensors that produce adversarial
//      outputs (e.g. always generate a specific response regardless of context).
//
//   2. REPLAY OLD FRAGMENTS: inject an expired or outdated fragment to serve
//      stale responses, bypassing fresh inference.
//
//   3. CROSS-MODEL POLLUTION: inject a fragment from model A into a cache
//      expecting model B. The model hash check in CanInject() catches this,
//      but only if the ModelID in the fragment header is trusted.
//
// SOLUTION: HMAC-SHA256 signing
// ───────────────────────────────
// Every fragment received from an external source must carry an HMAC-SHA256
// signature over its canonical payload. The signing key is:
//   - For server-delivered fragments: a per-deployment shared secret
//     (set via EDGESYNC_SIGNING_KEY environment variable or Android Keystore)
//   - For peer-to-peer sync: a per-session negotiated key (out of scope here)
//
// The canonical payload that is signed:
//   FragmentID | ModelHash | TokenStart | TokenEnd | LayerStart | LayerEnd |
//   LayerStride | ContentHash | ExpiresAt (Unix timestamp)
//
// Tensor data (Keys/Values) is NOT signed directly — it would be too expensive
// (~50ms for a 24MB SHA-256 on Cortex-A55). Instead, we sign the ContentHash,
// which is SHA-256 of the TokenIDs. A valid ContentHash implies valid tensors
// (the model is deterministic: same tokens → same KV tensors).
//
// REPLAY PROTECTION
// ──────────────────
// The signature includes ExpiresAt. An attacker cannot extend the TTL of a
// captured fragment without invalidating the signature.
// A nonce (random bytes in the signed payload) would provide stronger replay
// protection but requires server coordination. Not implemented in this MVP.
package security

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"time"

	"github.com/bossandboss/EdgeSync-LLM/cache"
)

// ─────────────────────────────────────────────────────────────────────────────
// Signing key management
// ─────────────────────────────────────────────────────────────────────────────

// SigningKeySource defines where the HMAC signing key comes from.
type SigningKeySource int

const (
	// KeySourceEnv reads the key from the EDGESYNC_SIGNING_KEY environment variable.
	// Value must be a 64-character hex string (32 bytes = 256-bit key).
	// Suitable for server-side deployments.
	KeySourceEnv SigningKeySource = iota

	// KeySourceAndroidKeystore uses Android Keystore API for key storage.
	// The key is generated on first run and never leaves the secure enclave.
	// Suitable for on-device fragment validation (self-signed fragments).
	// Note: Android Keystore access is via JNI; the key bytes are never
	// exposed to Go — signing is done in Kotlin using javax.crypto.Mac.
	KeySourceAndroidKeystore

	// KeySourceExplicit uses a caller-provided key (testing only).
	KeySourceExplicit
)

const (
	// envVarSigningKey is the environment variable for the signing key.
	envVarSigningKey = "EDGESYNC_SIGNING_KEY"

	// SignatureLen is the length of a hex-encoded HMAC-SHA256 signature.
	SignatureLen = 64 // 32 bytes × 2 hex chars
)

// Signer manages fragment signing and verification.
type Signer struct {
	key []byte
}

// NewSignerFromEnv creates a Signer using the key from EDGESYNC_SIGNING_KEY.
// Returns an error if the variable is missing or malformed.
func NewSignerFromEnv() (*Signer, error) {
	hexKey := os.Getenv(envVarSigningKey)
	if hexKey == "" {
		return nil, fmt.Errorf(
			"signing key not set: export %s=<64-hex-char key> or use NewSignerFromKey()",
			envVarSigningKey,
		)
	}
	key, err := hex.DecodeString(hexKey)
	if err != nil {
		return nil, fmt.Errorf("signing key: invalid hex: %w", err)
	}
	if len(key) < 16 {
		return nil, fmt.Errorf("signing key: minimum 16 bytes (128-bit), got %d", len(key))
	}
	return &Signer{key: key}, nil
}

// NewSignerFromKey creates a Signer from explicit key bytes (testing/embedding use).
func NewSignerFromKey(key []byte) (*Signer, error) {
	if len(key) < 16 {
		return nil, fmt.Errorf("signing key: minimum 16 bytes, got %d", len(key))
	}
	keyCopy := make([]byte, len(key))
	copy(keyCopy, key)
	return &Signer{key: keyCopy}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Signed fragment envelope
// ─────────────────────────────────────────────────────────────────────────────

// SignedFragment wraps a KVFragment with its HMAC signature.
// This is the wire format for fragments received from external sources.
type SignedFragment struct {
	Fragment  *cache.KVFragment
	Signature string // hex-encoded HMAC-SHA256, SignatureLen chars
}

// ─────────────────────────────────────────────────────────────────────────────
// Canonical payload construction
// ─────────────────────────────────────────────────────────────────────────────

// canonicalPayload builds the deterministic byte sequence that is HMAC-signed.
//
// Fields included (in fixed order to ensure determinism across platforms):
//   FragmentID (length-prefixed UTF-8)
//   ModelHash  (8-char hex, fixed length)
//   TokenStart (int32 little-endian)
//   TokenEnd   (int32 little-endian)
//   LayerStart (int32 little-endian)
//   LayerEnd   (int32 little-endian)
//   LayerStride (int32 little-endian)
//   ContentHash (64-char hex SHA-256, fixed length)
//   MerkleRoot (64-char hex SHA-256, fixed length — see merkle.go)
//   ExpiresAt  (int64 Unix timestamp, little-endian)
//   Engine     (length-prefixed UTF-8)
//
// Raw tensor data (Keys/Values) is intentionally excluded from direct signing —
// signing 24MB of float32 data takes ~50ms on Cortex-A55. Instead, MerkleRoot
// (computed cheaply per-layer, see security/merkle.go) commits to the tensor
// content: any single-layer corruption changes the root, and the HMAC over
// the root then transitively protects every layer's bytes. ContentHash alone
// (SHA-256 of TokenIDs) only proves the *input* was unmodified, not that the
// *computed tensors* match — MerkleRoot closes that gap.
//
// Fragments whose tensor layout doesn't support per-layer splitting (e.g. the
// ONNX header-prefixed wire format) get MerkleRoot = "" and degrade gracefully
// to ContentHash-only integrity, matching pre-Merkle behavior.
func canonicalPayload(f *cache.KVFragment) []byte {
	buf := make([]byte, 0, 320)

	// FragmentID (length-prefixed)
	buf = appendLenPrefixed(buf, []byte(f.ID))

	// ModelHash (fixed 8 bytes)
	buf = append(buf, []byte(f.Model.Hash())...)

	// Integer fields (4 bytes each, little-endian)
	var tmp [8]byte
	for _, v := range []int{f.TokenStart, f.TokenEnd, f.LayerStart, f.LayerEnd, f.LayerStride} {
		binary.LittleEndian.PutUint32(tmp[:4], uint32(v))
		buf = append(buf, tmp[:4]...)
	}

	// ContentHash (fixed 64 bytes)
	buf = append(buf, []byte(f.ContentHash)...)

	// MerkleRoot (length-prefixed; empty for layouts that don't support per-layer split)
	buf = appendLenPrefixed(buf, []byte(fragmentMerkleRootHex(f)))

	// ExpiresAt (8 bytes, Unix timestamp)
	binary.LittleEndian.PutUint64(tmp[:], uint64(f.ExpiresAt.Unix()))
	buf = append(buf, tmp[:]...)

	// Engine (length-prefixed)
	buf = appendLenPrefixed(buf, []byte(f.Engine))

	return buf
}

// fragmentMerkleRootHex computes the Merkle root for a fragment's tensor blocks.
// Returns "" if the tree cannot be built (e.g. non-uniform layout like ONNX's
// header-prefixed format) — Sign/Verify degrade gracefully to ContentHash-only
// integrity in that case.
func fragmentMerkleRootHex(f *cache.KVFragment) string {
	tree, err := BuildMerkleTree(f)
	if err != nil {
		return ""
	}
	return tree.RootHex()
}

func appendLenPrefixed(dst, src []byte) []byte {
	var lenBuf [4]byte
	binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(src)))
	dst = append(dst, lenBuf[:]...)
	dst = append(dst, src...)
	return dst
}

// ─────────────────────────────────────────────────────────────────────────────
// Sign and Verify
// ─────────────────────────────────────────────────────────────────────────────

// Sign computes an HMAC-SHA256 signature for the fragment and returns a
// SignedFragment ready for transmission.
func (s *Signer) Sign(f *cache.KVFragment) (*SignedFragment, error) {
	if f == nil {
		return nil, fmt.Errorf("sign: fragment is nil")
	}

	payload := canonicalPayload(f)
	mac := hmac.New(sha256.New, s.key)
	mac.Write(payload)
	sig := hex.EncodeToString(mac.Sum(nil))

	return &SignedFragment{
		Fragment:  f,
		Signature: sig,
	}, nil
}

// Verify checks the HMAC signature of a SignedFragment.
// Returns nil if the signature is valid, or a descriptive error otherwise.
//
// Also performs additional semantic checks:
//   - Fragment must not be expired
//   - ContentHash must match re-computed hash of TokenIDs
//   - Model hash must be non-empty
func (s *Signer) Verify(sf *SignedFragment) error {
	if sf == nil || sf.Fragment == nil {
		return fmt.Errorf("verify: nil fragment")
	}
	if len(sf.Signature) != SignatureLen {
		return fmt.Errorf("verify: signature length %d, expected %d", len(sf.Signature), SignatureLen)
	}

	// Recompute expected signature
	payload := canonicalPayload(sf.Fragment)
	mac := hmac.New(sha256.New, s.key)
	mac.Write(payload)
	expected := hex.EncodeToString(mac.Sum(nil))

	// Constant-time comparison to prevent timing attacks
	sigBytes, err := hex.DecodeString(sf.Signature)
	if err != nil {
		return fmt.Errorf("verify: malformed signature hex: %w", err)
	}
	expectedBytes, _ := hex.DecodeString(expected)

	if !hmac.Equal(sigBytes, expectedBytes) {
		return fmt.Errorf("verify: HMAC signature mismatch — fragment may be tampered or from wrong key")
	}

	// Semantic checks after signature validation
	f := sf.Fragment

	if f.IsExpired() {
		return fmt.Errorf("verify: fragment %q is expired (expired at %s)", f.ID, f.ExpiresAt.Format(time.RFC3339))
	}

	if f.Model.Hash() == "" {
		return fmt.Errorf("verify: fragment has empty model hash")
	}

	// Verify ContentHash matches TokenIDs
	if err := verifyContentHash(f); err != nil {
		return fmt.Errorf("verify: %w", err)
	}

	return nil
}

// verifyContentHash recomputes the ContentHash from TokenIDs and compares.
// A mismatch means the TokenIDs were altered after signing (or the fragment
// was constructed incorrectly).
func verifyContentHash(f *cache.KVFragment) error {
	if len(f.TokenIDs) == 0 {
		return fmt.Errorf("fragment has no TokenIDs")
	}
	hashInput := make([]byte, len(f.TokenIDs)*4)
	for i, tok := range f.TokenIDs {
		binary.LittleEndian.PutUint32(hashInput[i*4:], uint32(tok))
	}
	h := sha256.Sum256(hashInput)
	recomputed := hex.EncodeToString(h[:])
	if recomputed != f.ContentHash {
		return fmt.Errorf("ContentHash mismatch: stored=%q, recomputed=%q (TokenIDs may be altered)",
			f.ContentHash, recomputed)
	}
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// UnsignedPolicy — defines what to do with fragments that have no signature
// ─────────────────────────────────────────────────────────────────────────────

// UnsignedPolicy controls how the cache handles fragments without a signature.
// Used for backwards compatibility during migration.
type UnsignedPolicy int

const (
	// UnsignedReject rejects all unsigned fragments. Use in production.
	UnsignedReject UnsignedPolicy = iota

	// UnsignedAcceptLocal accepts unsigned fragments only if they were produced
	// by this process (engine == local engine name). Rejects remote unsigned fragments.
	UnsignedAcceptLocal

	// UnsignedAcceptAll accepts all fragments regardless of signature.
	// Use only for development / benchmarking. Never use in production.
	UnsignedAcceptAll
)

// VerifyOrPolicy checks a signed fragment according to the given policy.
// If signer is nil, behavior is governed by policy alone.
func VerifyOrPolicy(signer *Signer, sf *SignedFragment, policy UnsignedPolicy, localEngine string) error {
	if signer == nil {
		switch policy {
		case UnsignedReject:
			return fmt.Errorf("no signer configured and policy=UnsignedReject")
		case UnsignedAcceptLocal:
			if sf.Fragment != nil && sf.Fragment.Engine != localEngine {
				return fmt.Errorf("no signer configured: remote fragment from engine %q rejected by UnsignedAcceptLocal policy",
					sf.Fragment.Engine)
			}
			return nil
		case UnsignedAcceptAll:
			return nil
		}
	}

	if sf.Signature == "" {
		switch policy {
		case UnsignedReject:
			return fmt.Errorf("fragment %q has no signature and policy=UnsignedReject", sf.Fragment.ID)
		case UnsignedAcceptLocal:
			if sf.Fragment.Engine != localEngine {
				return fmt.Errorf("unsigned remote fragment rejected")
			}
			return nil
		case UnsignedAcceptAll:
			return nil
		}
	}

	return signer.Verify(sf)
}
