package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"testing"
)

// --- test types ---

type scoreResult struct {
	Score  int    `json:"score"`
	Reason string `json:"reason"`
}

func (s scoreResult) Validate() error {
	if s.Score < 0 || s.Score > 10 {
		return fmt.Errorf("score must be 0-10, got %d", s.Score)
	}
	if s.Reason == "" {
		return fmt.Errorf("reason must not be empty")
	}
	return nil
}

type alwaysValid struct {
	Value string `json:"value"`
}

func (a alwaysValid) Validate() error { return nil }

// --- tests ---

func TestExtractValidJSON(t *testing.T) {
	result := scoreResult{Score: 7, Reason: "good quality"}
	jsonBytes, _ := json.Marshal(result)

	mp := newMockProvider()
	mp.responses["test-model"] = ProviderResponse{
		Text:   string(jsonBytes),
		Model:  "test-model",
		Tokens: TokenUsage{TotalTokens: 10},
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	spec := PromptSpec{ID: "test/extract", Role: "test"}

	typed, err := Extract[scoreResult](context.Background(), rt, spec, map[string]any{"query": "eval"}, ExtractOptions[scoreResult]{})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	if typed.Value.Score != 7 {
		t.Fatalf("Score = %d, want 7", typed.Value.Score)
	}
	if typed.Value.Reason != "good quality" {
		t.Fatalf("Reason = %q, want %q", typed.Value.Reason, "good quality")
	}
}

func TestExtractWithMarkdownFences(t *testing.T) {
	result := alwaysValid{Value: "hello"}
	jsonBytes, _ := json.Marshal(result)
	wrappedJSON := "```json\n" + string(jsonBytes) + "\n```"

	mp := newMockProvider()
	mp.responses["test-model"] = ProviderResponse{
		Text:   wrappedJSON,
		Model:  "test-model",
		Tokens: TokenUsage{TotalTokens: 10},
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	spec := PromptSpec{ID: "test/extract_md", Role: "test"}

	typed, err := Extract[alwaysValid](context.Background(), rt, spec, map[string]any{"query": "test"}, ExtractOptions[alwaysValid]{})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	if typed.Value.Value != "hello" {
		t.Fatalf("Value = %q, want %q", typed.Value.Value, "hello")
	}
}

func TestExtractValidationFailureTriggersRetry(t *testing.T) {
	var callCount atomic.Int32

	vp := &validationRetryProvider{
		callCount: &callCount,
		responses: []string{
			`{"score": 15, "reason": "too high"}`,   // fails validation (>10)
			`{"score": 7, "reason": "now correct"}`, // passes validation
		},
	}

	rt := newTestRuntime(vp, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	spec := PromptSpec{ID: "test/validate", Role: "test"}

	typed, err := Extract[scoreResult](context.Background(), rt, spec, map[string]any{"query": "eval"}, ExtractOptions[scoreResult]{})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	if typed.Value.Score != 7 {
		t.Fatalf("Score = %d, want 7 (after retry)", typed.Value.Score)
	}
	// Should have called provider at least 2 times (first fail + second success).
	if callCount.Load() < 2 {
		t.Fatalf("callCount = %d, want >= 2", callCount.Load())
	}
}

func TestExtractInvalidJSONAfterRetries(t *testing.T) {
	mp := newMockProvider()
	mp.responses["test-model"] = ProviderResponse{
		Text:   "this is not json at all",
		Model:  "test-model",
		Tokens: TokenUsage{TotalTokens: 5},
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	spec := PromptSpec{ID: "test/bad_json", Role: "test"}

	_, err := Extract[scoreResult](context.Background(), rt, spec, map[string]any{"query": "eval"}, ExtractOptions[scoreResult]{})
	if err == nil {
		t.Fatal("expected error for invalid JSON after all retries")
	}
}

func TestExtractValidationFailsAfterAllRetries(t *testing.T) {
	// All responses fail validation.
	mp := newMockProvider()
	mp.responses["test-model"] = ProviderResponse{
		Text:   `{"score": 15, "reason": "always too high"}`,
		Model:  "test-model",
		Tokens: TokenUsage{TotalTokens: 5},
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	spec := PromptSpec{ID: "test/always_invalid", Role: "test"}

	_, err := Extract[scoreResult](context.Background(), rt, spec, map[string]any{"query": "eval"}, ExtractOptions[scoreResult]{})
	if err == nil {
		t.Fatal("expected error when validation fails after all retries")
	}
}

func TestExtractEnsembleUsesMergeFunc(t *testing.T) {
	mp := newMockProvider()
	mp.responses["model-a"] = ProviderResponse{
		Text:   `{"score": 3, "reason": "from a"}`,
		Model:  "model-a",
		Tokens: TokenUsage{TotalTokens: 11},
	}
	mp.responses["model-b"] = ProviderResponse{
		Text:   `{"score": 9, "reason": "from b"}`,
		Model:  "model-b",
		Tokens: TokenUsage{TotalTokens: 13},
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "model-a"},
		Orchestration: map[string]RoleConfig{
			"ensemble_extract": {
				Mode:   modePtr(ModeEnsemble),
				Models: []string{"model-a", "model-b"},
			},
		},
	})

	spec := PromptSpec{ID: "test/ensemble_extract", Role: "test"}

	var merged atomic.Int32
	typed, err := Extract[scoreResult](context.Background(), rt, spec, map[string]any{"query": "eval"}, ExtractOptions[scoreResult]{
		Merge: func(items []TypedResponse[scoreResult]) (scoreResult, error) {
			merged.Add(1)
			if len(items) != 2 {
				return scoreResult{}, fmt.Errorf("len(items) = %d, want 2", len(items))
			}
			return scoreResult{
				Score:  9,
				Reason: "merged winner",
			}, nil
		},
	})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	if merged.Load() != 1 {
		t.Fatalf("Merge called %d times, want 1", merged.Load())
	}
	if typed.Value.Score != 9 {
		t.Fatalf("Score = %d, want 9", typed.Value.Score)
	}
	if typed.Value.Reason != "merged winner" {
		t.Fatalf("Reason = %q, want %q", typed.Value.Reason, "merged winner")
	}
	if len(typed.Response.ModeTrace) != 2 {
		t.Fatalf("ModeTrace len = %d, want 2", len(typed.Response.ModeTrace))
	}
}

func TestExtractChallengeUsesChallengerOutput(t *testing.T) {
	mp := newMockProvider()
	mp.responses["gen-model"] = ProviderResponse{
		Text:   `{"score": 2, "reason": "initial answer"}`,
		Model:  "gen-model",
		Tokens: TokenUsage{TotalTokens: 10},
	}
	mp.responses["chal-model"] = ProviderResponse{
		Text:   `{"score": 8, "reason": "challenger final answer"}`,
		Model:  "chal-model",
		Tokens: TokenUsage{TotalTokens: 14},
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "gen-model"},
		Orchestration: map[string]RoleConfig{
			"challenge_extract": {
				Mode:                  modePtr(ModeChallenge),
				Models:                []string{"gen-model", "chal-model"},
				ChallengeInstructions: strPtr("Challenge this response."),
			},
		},
	})

	spec := PromptSpec{ID: "test/challenge_extract", Role: "test"}

	typed, err := Extract[scoreResult](context.Background(), rt, spec, map[string]any{"query": "eval"}, ExtractOptions[scoreResult]{})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	if typed.Value.Score != 8 {
		t.Fatalf("Score = %d, want 8 from challenger", typed.Value.Score)
	}
	if typed.Value.Reason != "challenger final answer" {
		t.Fatalf("Reason = %q, want challenger output", typed.Value.Reason)
	}
	if len(typed.Response.ModeTrace) != 2 {
		t.Fatalf("ModeTrace len = %d, want 2", len(typed.Response.ModeTrace))
	}
	if typed.Response.Text != `{"score": 8, "reason": "challenger final answer"}` {
		t.Fatalf("Response.Text = %q, want challenger JSON only", typed.Response.Text)
	}
}

func TestExtractChallengeFallsBackToGeneratorOutputWhenChallengerFails(t *testing.T) {
	mp := newMockProvider()
	mp.responses["gen-model"] = ProviderResponse{
		Text:   `{"score":2,"reason":"generator fallback"}`,
		Model:  "gen-model",
		Tokens: TokenUsage{TotalTokens: 10},
	}
	mp.errors["chal-model"] = context.DeadlineExceeded

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "gen-model"},
		Orchestration: map[string]RoleConfig{
			"challenge_extract": {
				Mode:                  modePtr(ModeChallenge),
				Models:                []string{"gen-model", "chal-model"},
				ChallengeInstructions: strPtr("Challenge this response."),
			},
		},
	})

	spec := PromptSpec{ID: "test/challenge_extract", Role: "test"}

	typed, err := Extract[scoreResult](context.Background(), rt, spec, map[string]any{"query": "eval"}, ExtractOptions[scoreResult]{})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	if typed.Value.Score != 2 {
		t.Fatalf("Score = %d, want generator fallback score 2", typed.Value.Score)
	}
	if typed.Value.Reason != "generator fallback" {
		t.Fatalf("Reason = %q, want generator fallback", typed.Value.Reason)
	}
	if typed.Response.Text != `{"score":2,"reason":"generator fallback"}` {
		t.Fatalf("Response.Text = %q, want generator JSON", typed.Response.Text)
	}
	if len(typed.Response.ModeTrace) != 1 || typed.Response.ModeTrace[0].Role != "generator" {
		t.Fatalf("ModeTrace = %+v, want generator-only fallback trace", typed.Response.ModeTrace)
	}
}

func TestExtractStampsResolvedRoleSeparatelyFromModel(t *testing.T) {
	mp := newMockProvider()
	mp.responses["resolved-model"] = ProviderResponse{
		Text:   `{"score":5,"reason":"ok"}`,
		Model:  "resolved-model",
		Tokens: TokenUsage{TotalTokens: 10},
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "default-model"},
		Orchestration: map[string]RoleConfig{
			"score_role": {
				Model: strPtr("resolved-model"),
			},
		},
		PromptRoles: map[string]string{
			"score_prompt": "score_role",
		},
	})

	typed, err := Extract[scoreResult](
		context.Background(),
		rt,
		PromptSpec{ID: "test/score_prompt"},
		map[string]any{"query": "eval"},
		ExtractOptions[scoreResult]{},
	)
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	if typed.Response.Role != "score_role" {
		t.Fatalf("Response.Role = %q, want %q", typed.Response.Role, "score_role")
	}
	if typed.Response.Model != "resolved-model" {
		t.Fatalf("Response.Model = %q, want %q", typed.Response.Model, "resolved-model")
	}
	if typed.Response.Role == typed.Response.Model {
		t.Fatalf("Response.Role and Response.Model should remain distinct, both = %q", typed.Response.Role)
	}
}

func TestMergeByMedianSelectsMedianValue(t *testing.T) {
	merge := MergeByMedian(func(v scoreResult) float64 { return float64(v.Score) })

	got, err := merge([]TypedResponse[scoreResult]{
		{Value: scoreResult{Score: 9, Reason: "high"}},
		{Value: scoreResult{Score: 3, Reason: "low"}},
		{Value: scoreResult{Score: 5, Reason: "median"}},
	})
	if err != nil {
		t.Fatalf("MergeByMedian() error = %v", err)
	}
	if got.Score != 5 {
		t.Fatalf("Score = %d, want 5", got.Score)
	}
}

func TestMergeByMajoritySelectsWinningBucket(t *testing.T) {
	merge := MergeByMajority(func(v scoreResult) string {
		if v.Score >= 7 {
			return "high"
		}
		return "low"
	})

	got, err := merge([]TypedResponse[scoreResult]{
		{Value: scoreResult{Score: 8, Reason: "first high"}},
		{Value: scoreResult{Score: 2, Reason: "low"}},
		{Value: scoreResult{Score: 7, Reason: "second high"}},
	})
	if err != nil {
		t.Fatalf("MergeByMajority() error = %v", err)
	}
	if got.Score != 8 {
		t.Fatalf("Score = %d, want 8 (first majority item)", got.Score)
	}
}

func TestRepairJSON(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "plain json",
			input: `{"key": "value"}`,
			want:  `{"key": "value"}`,
		},
		{
			name:  "markdown fenced",
			input: "```json\n{\"key\": \"value\"}\n```",
			want:  `{"key": "value"}`,
		},
		{
			name:  "markdown fenced no language",
			input: "```\n{\"key\": \"value\"}\n```",
			want:  `{"key": "value"}`,
		},
		{
			name:  "with whitespace",
			input: "  \n{\"key\": \"value\"}\n  ",
			want:  `{"key": "value"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := repairJSON(tt.input)
			if got != tt.want {
				t.Fatalf("repairJSON() = %q, want %q", got, tt.want)
			}
		})
	}
}

func modePtr(m Mode) *Mode { return &m }

func strPtr(s string) *string { return &s }

// --- helper provider ---

// validationRetryProvider returns different responses on successive calls.
type validationRetryProvider struct {
	callCount *atomic.Int32
	responses []string
}

func (v *validationRetryProvider) Name() string { return "validation-retry" }

func (v *validationRetryProvider) Call(ctx context.Context, req ProviderRequest) (ProviderResponse, error) {
	idx := int(v.callCount.Add(1)) - 1
	text := v.responses[len(v.responses)-1] // default to last
	if idx < len(v.responses) {
		text = v.responses[idx]
	}
	return ProviderResponse{
		Text:   text,
		Model:  req.Model,
		Tokens: TokenUsage{TotalTokens: 10},
	}, nil
}
