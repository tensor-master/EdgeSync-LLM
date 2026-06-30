package security

import (
	"testing"

	"github.com/bossandboss/EdgeSync-LLM/cache"
)

// makeTestFragmentForMerkle builds on the package's existing makeTestFragment
// (security_test.go) but fills the tensor bytes with non-zero, non-uniform
// content — an all-zero tensor would make single-byte corruption tests less
// meaningful, and a varying pattern better exercises the hashing path.
func makeTestFragmentForMerkle(t *testing.T) *cache.KVFragment {
	t.Helper()
	f := makeTestFragment("merkle-test-" + t.Name())
	for i := range f.Keys {
		f.Keys[i] = byte((i * 37) % 256)
	}
	for i := range f.Values {
		f.Values[i] = byte((i*53 + 7) % 256)
	}
	return f
}

// ─────────────────────────────────────────────────────────────────────────────
// BuildMerkleTree tests
// ─────────────────────────────────────────────────────────────────────────────

func TestBuildMerkleTree_Deterministic(t *testing.T) {
	f := makeTestFragmentForMerkle(t)

	tree1, err := BuildMerkleTree(f)
	if err != nil {
		t.Fatalf("BuildMerkleTree: %v", err)
	}
	tree2, err := BuildMerkleTree(f)
	if err != nil {
		t.Fatalf("BuildMerkleTree (2nd): %v", err)
	}

	if tree1.RootHex() != tree2.RootHex() {
		t.Error("BuildMerkleTree should be deterministic for the same fragment")
	}
}

func TestBuildMerkleTree_DifferentDataDifferentRoot(t *testing.T) {
	f1 := makeTestFragmentForMerkle(t)
	f2 := makeTestFragmentForMerkle(t)
	f2.Keys[0] ^= 0xFF

	tree1, _ := BuildMerkleTree(f1)
	tree2, _ := BuildMerkleTree(f2)

	if tree1.RootHex() == tree2.RootHex() {
		t.Error("single-byte tensor change should produce a different Merkle root")
	}
}

func TestBuildMerkleTree_NilFragment(t *testing.T) {
	_, err := BuildMerkleTree(nil)
	if err == nil {
		t.Error("expected error for nil fragment")
	}
}

func TestBuildMerkleTree_OddLayerCount(t *testing.T) {
	// 7 captured layers (odd) to exercise the self-pad path in buildTreeFromLeaves.
	model := cache.ModelID{
		Architecture: "qwen", Name: "test", Quantization: "Q4_K_M",
		ContextLength: 4096, HeadDim: 64, NumKVHeads: 8, NumLayers: 24,
	}
	tokens := 64
	tokenIDs := make([]int32, tokens)
	for i := range tokenIDs {
		tokenIDs[i] = int32(i + 1)
	}
	numLayers := 7 // layers 0,2,4,6,8,10,12 with stride 2, LayerEnd=14
	keys := make([]byte, numLayers*tokens*8*64*4)
	vals := make([]byte, numLayers*tokens*8*64*4)
	for i := range keys {
		keys[i] = byte(i % 256)
	}
	for i := range vals {
		vals[i] = byte((i + 1) % 256)
	}

	f, err := cache.NewFragment("odd-layer-test", model, 0, tokens, 0, 14, 2,
		keys, vals, tokenIDs, nil, "llamacpp", "b3117", cache.DefaultTTLSession)
	if err != nil {
		t.Fatalf("setup NewFragment: %v", err)
	}

	tree, err := BuildMerkleTree(f)
	if err != nil {
		t.Fatalf("BuildMerkleTree: %v", err)
	}
	if len(tree.Leaves) != 7 {
		t.Errorf("want 7 leaves, got %d", len(tree.Leaves))
	}
	if tree.RootHex() == "" {
		t.Error("root should not be empty for odd layer count")
	}
}

func TestBuildMerkleTree_NonUniformLayout(t *testing.T) {
	f := makeTestFragmentForMerkle(t)
	f.Keys = f.Keys[:len(f.Keys)-3] // not evenly divisible by layer count

	_, err := BuildMerkleTree(f)
	if err == nil {
		t.Error("expected error for non-uniform tensor layout")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Proof generation and verification
// ─────────────────────────────────────────────────────────────────────────────

func TestGenerateAndVerifyProof_ValidLeaf(t *testing.T) {
	f := makeTestFragmentForMerkle(t)
	tree, err := BuildMerkleTree(f)
	if err != nil {
		t.Fatalf("BuildMerkleTree: %v", err)
	}

	for i := 0; i < len(tree.Leaves); i++ {
		proof, err := tree.GenerateProof(i)
		if err != nil {
			t.Fatalf("GenerateProof(%d): %v", i, err)
		}
		if !VerifyProof(tree.Leaves[i], proof, tree.Root()) {
			t.Errorf("VerifyProof failed for valid leaf %d", i)
		}
	}
}

func TestGenerateProof_OutOfRange(t *testing.T) {
	f := makeTestFragmentForMerkle(t)
	tree, _ := BuildMerkleTree(f)

	if _, err := tree.GenerateProof(-1); err == nil {
		t.Error("expected error for negative leaf index")
	}
	if _, err := tree.GenerateProof(len(tree.Leaves)); err == nil {
		t.Error("expected error for leaf index >= len(Leaves)")
	}
}

func TestVerifyProof_TamperedLeaf(t *testing.T) {
	f := makeTestFragmentForMerkle(t)
	tree, _ := BuildMerkleTree(f)

	proof, _ := tree.GenerateProof(3)
	var tamperedHash [32]byte
	copy(tamperedHash[:], tree.Leaves[3][:])
	tamperedHash[0] ^= 0xFF

	if VerifyProof(tamperedHash, proof, tree.Root()) {
		t.Error("VerifyProof should fail for a tampered leaf hash")
	}
}

func TestVerifyProof_WrongRoot(t *testing.T) {
	f := makeTestFragmentForMerkle(t)
	tree, _ := BuildMerkleTree(f)

	proof, _ := tree.GenerateProof(0)
	var wrongRoot [32]byte
	wrongRoot[0] = 0x01

	if VerifyProof(tree.Leaves[0], proof, wrongRoot) {
		t.Error("VerifyProof should fail against a wrong root")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// VerifyFragmentIntegrity tests
// ─────────────────────────────────────────────────────────────────────────────

func TestVerifyFragmentIntegrity_Valid(t *testing.T) {
	f := makeTestFragmentForMerkle(t)
	tree, _ := BuildMerkleTree(f)

	if err := VerifyFragmentIntegrity(f, tree.RootHex()); err != nil {
		t.Errorf("VerifyFragmentIntegrity should pass for unmodified fragment: %v", err)
	}
}

func TestVerifyFragmentIntegrity_Corrupted(t *testing.T) {
	f := makeTestFragmentForMerkle(t)
	tree, _ := BuildMerkleTree(f)
	expectedRoot := tree.RootHex()

	f.Values[100] ^= 0xFF

	if err := VerifyFragmentIntegrity(f, expectedRoot); err == nil {
		t.Error("VerifyFragmentIntegrity should fail for corrupted tensor data")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// VerifyLayerIntegrity tests
// ─────────────────────────────────────────────────────────────────────────────

func TestVerifyLayerIntegrity_UncorruptedLayer(t *testing.T) {
	f := makeTestFragmentForMerkle(t)
	originalTree, _ := BuildMerkleTree(f)

	ok, err := VerifyLayerIntegrity(f, 5, originalTree)
	if err != nil {
		t.Fatalf("VerifyLayerIntegrity: %v", err)
	}
	if !ok {
		t.Error("expected layer 5 to verify successfully")
	}
}

func TestVerifyLayerIntegrity_CorruptedSpecificLayer(t *testing.T) {
	f := makeTestFragmentForMerkle(t)
	originalTree, _ := BuildMerkleTree(f)

	numLayers := f.NumLayersCovered()
	keysPerLayer := len(f.Keys) / numLayers
	f.Keys[5*keysPerLayer] ^= 0xFF // corrupt layer index 5 specifically

	ok5, _ := VerifyLayerIntegrity(f, 5, originalTree)
	if ok5 {
		t.Error("expected layer 5 verification to fail after corruption")
	}

	ok0, _ := VerifyLayerIntegrity(f, 0, originalTree)
	if !ok0 {
		t.Error("expected layer 0 to still verify (it was not corrupted)")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Integration: Sign/Verify with MerkleRoot embedded (signing.go interaction)
// ─────────────────────────────────────────────────────────────────────────────

func TestSignVerify_DetectsTensorCorruptionViaMerkleRoot(t *testing.T) {
	key := make([]byte, 32)
	signer, _ := NewSignerFromKey(key)

	f := makeTestFragmentForMerkle(t)
	sf, err := signer.Sign(f)
	if err != nil {
		t.Fatalf("Sign: %v", err)
	}

	if err := signer.Verify(sf); err != nil {
		t.Errorf("Verify should pass for unmodified fragment: %v", err)
	}

	// Corrupt a tensor byte WITHOUT touching TokenIDs — ContentHash alone
	// would NOT catch this; this is exactly the gap MerkleRoot closes.
	sf.Fragment.Keys[42] ^= 0xFF

	if err := signer.Verify(sf); err == nil {
		t.Error("Verify should fail when tensor bytes are corrupted but TokenIDs are untouched")
	}
}
