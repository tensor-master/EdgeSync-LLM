// docs/examples/onnx_session.go
//
// Complete implementation of the ONNXSession interface using
// github.com/yalue/onnxruntime_go for desktop benchmarking.
//
// To run:
//   go get github.com/yalue/onnxruntime_go
//   go run docs/examples/onnx_session.go
package main

import (
	"context"
	"fmt"
	"log"

	"react-example/adapter"
	"react-example/cache"
)

// ORTSession implements adapter.ONNXSession against onnxruntime_go.
// For production Android, replace with com.microsoft.onnxruntime (Java/Kotlin).
type ORTSession struct {
	modelPath string
	model     cache.ModelID
	// In a real implementation:
	// session  *ort.Session
	// tokenizer *tokenizers.Tokenizer
}

func NewORTSession(modelPath string, model cache.ModelID) *ORTSession {
	return &ORTSession{modelPath: modelPath, model: model}
}

// RunPrefill executes a prefill pass and returns present_key_values for all layers.
// Returns [num_layers]keys and [num_layers]values as flat float32 slices.
//
// Real implementation:
//
//	inputTensor, _ := ort.NewTensor(shape, tokenIDs)
//	outputs, _ := session.Run([]string{"input_ids"}, []*ort.Tensor{inputTensor})
//	// outputs contains logits + present.N.key + present.N.value for N in 0..NumLayers-1
func (s *ORTSession) RunPrefill(tokenIDs []int32) (keys [][]float32, values [][]float32, err error) {
	numLayers := s.model.NumLayers
	seqLen := len(tokenIDs)
	H := s.model.NumKVHeads
	D := s.model.HeadDim
	floatsPerLayer := H * seqLen * D

	// Stub: return deterministic zeros (replace with real ORT call)
	keys = make([][]float32, numLayers)
	values = make([][]float32, numLayers)
	for i := 0; i < numLayers; i++ {
		keys[i] = make([]float32, floatsPerLayer)
		values[i] = make([]float32, floatsPerLayer)
		// In production: keys[i] = outputs["present."+strconv.Itoa(i)+".key"].GetData()
	}
	return keys, values, nil
}

// RunWithPast executes generation with past_key_values.
// pastKeys, pastValues: [num_layers][]float32, each [1 × heads × past_len × dim]
func (s *ORTSession) RunWithPast(
	pastKeys, pastValues [][]float32,
	newTokenIDs []int32,
	maxNewTokens int,
) (string, int, error) {
	if pastKeys == nil {
		// Cold start: no past KV
		return fmt.Sprintf("[cold generation, %d new tokens, max %d]",
			len(newTokenIDs), maxNewTokens), len(newTokenIDs), nil
	}
	pastLen := 0
	if len(pastKeys) > 0 {
		// Infer past sequence length from tensor shape: past_keys[layer] is [H × past_len × D]
		pastLen = len(pastKeys[0]) / (s.model.NumKVHeads * s.model.HeadDim)
	}
	return fmt.Sprintf("[warm generation from pos %d, %d new tokens, max %d]",
		pastLen, len(newTokenIDs), maxNewTokens), maxNewTokens, nil
}

func (s *ORTSession) Tokenize(text string) ([]int32, error) {
	// Real: use tokenizers.Tokenizer loaded from tokenizer.json
	// Stub: return fake token IDs
	ids := make([]int32, len(text)/4+1)
	for i := range ids {
		ids[i] = int32(i + 100)
	}
	return ids, nil
}

func (s *ORTSession) Close() error {
	// Real: session.Destroy()
	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Example usage
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	model := cache.ModelID{
		Architecture:  "qwen",
		Name:          "Qwen2.5-0.5B",
		Quantization:  "Q4_K_M",
		ContextLength: 4096,
		HeadDim:       64,
		NumKVHeads:    8,
		NumLayers:     24,
	}

	// 1. Create ONNX adapter
	session := NewORTSession("./qwen2.5-0.5b-onnx/model.onnx", model)
	onnxAdapter := adapter.NewONNXAdapter(session, model)
	adapter.Register(onnxAdapter)

	ctx := context.Background()
	prompt := "explain write-ahead logging in sqlite for high concurrency"

	// 2. Tokenize
	tokenIDs, err := onnxAdapter.Tokenize(ctx, prompt)
	if err != nil {
		log.Fatalf("tokenize: %v", err)
	}
	fmt.Printf("Tokenized: %d tokens\n", len(tokenIDs))

	// 3. Extract fragment (simulated embedding)
	embedding := make([]float32, 384)
	for i := range embedding {
		embedding[i] = float32(i) / 384.0
	}

	fragment, err := onnxAdapter.ExtractFragment(
		ctx, tokenIDs,
		0, model.NumLayers, cache.FragmentLayerStride,
		embedding,
	)
	if err != nil {
		log.Fatalf("extract: %v", err)
	}
	fmt.Printf("Extracted fragment: ID=%s, size=%d KB, TTL=%s\n",
		fragment.ID,
		fragment.SizeBytes()/1024,
		fragment.ExpiresAt.Sub(fragment.CreatedAt).Round(1e9),
	)

	// 4. Simulate a second request — inject fragment and generate
	err = onnxAdapter.InjectFragment(ctx, fragment)
	if err != nil {
		log.Fatalf("inject: %v", err)
	}

	text, count, err := onnxAdapter.Generate(ctx, prompt+" and its performance implications", fragment.TokenEnd, 200)
	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	fmt.Printf("Generated %d tokens: %s\n", count, text)

	// 5. Cross-engine reshape demo
	fmt.Println("\n--- Cross-engine reshape ---")
	reshaped, err := adapter.ReshapeForEngine(fragment, "llamacpp")
	if err != nil {
		fmt.Printf("Reshape onnx→llamacpp: %v\n", err)
	} else {
		fmt.Printf("Reshaped to llamacpp: engine=%s, size=%d KB\n",
			reshaped.Engine, reshaped.SizeBytes()/1024)
	}
}
