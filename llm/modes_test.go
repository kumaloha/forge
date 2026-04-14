package llm

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"
)

// mockProvider is a configurable mock for testing modes.
type mockProvider struct {
	name      string
	responses map[string]ProviderResponse // keyed by model name
	errors    map[string]error            // keyed by model name
	callCount atomic.Int32
	delay     time.Duration
}

func newMockProvider() *mockProvider {
	return &mockProvider{
		name:      "mock",
		responses: make(map[string]ProviderResponse),
		errors:    make(map[string]error),
	}
}

func (m *mockProvider) Name() string { return m.name }

func (m *mockProvider) Call(ctx context.Context, req ProviderRequest) (ProviderResponse, error) {
	m.callCount.Add(1)

	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return ProviderResponse{}, ctx.Err()
		}
	}

	if err, ok := m.errors[req.Model]; ok {
		return ProviderResponse{}, err
	}

	if resp, ok := m.responses[req.Model]; ok {
		return resp, nil
	}

	// Default response.
	return ProviderResponse{
		Text:  fmt.Sprintf("response from %s", req.Model),
		Model: req.Model,
		Tokens: TokenUsage{
			PromptTokens:     10,
			CompletionTokens: 20,
			TotalTokens:      30,
		},
	}, nil
}

func newTestRuntime(p Provider, cfg LLMConfig) *Runtime {
	return NewRuntime(RuntimeConfig{
		Provider:    p,
		LLMConfig:   cfg,
		Concurrency: 4,
	})
}

type capturingProvider struct {
	req ProviderRequest
}

func (p *capturingProvider) Name() string { return "capturing" }

func (p *capturingProvider) Call(_ context.Context, req ProviderRequest) (ProviderResponse, error) {
	p.req = req
	return ProviderResponse{
		Text:  "captured",
		Model: req.Model,
	}, nil
}

func TestExecuteSingle(t *testing.T) {
	mp := newMockProvider()
	mp.responses["qwen3-max"] = ProviderResponse{
		Text:   "single response",
		Model:  "qwen3-max",
		Tokens: TokenUsage{PromptTokens: 5, CompletionTokens: 10, TotalTokens: 15},
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "qwen3-max"},
	})

	rc := rt.config.Resolve("test")
	built := BuiltPrompt{System: "sys", User: "usr"}

	resp, err := rt.executeSingle(context.Background(), rc, built, nil)
	if err != nil {
		t.Fatalf("executeSingle() error = %v", err)
	}
	if resp.Text != "single response" {
		t.Fatalf("Text = %q, want %q", resp.Text, "single response")
	}
	if len(resp.ModeTrace) != 1 || resp.ModeTrace[0].Mode != ModeSingle {
		t.Fatalf("ModeTrace = %+v, want single step", resp.ModeTrace)
	}
}

func TestChatSinglePassesTemperature(t *testing.T) {
	cp := &capturingProvider{}
	rt := newTestRuntime(cp, LLMConfig{
		Default: DefaultConfig{Model: "qwen3-max"},
	})

	_, err := rt.chatSingle(
		context.Background(),
		"qwen3-max",
		BuiltPrompt{System: "sys", User: "usr"},
		false,
		false,
		0.65,
		nil,
	)
	if err != nil {
		t.Fatalf("chatSingle() error = %v", err)
	}
	if cp.req.Temperature != 0.65 {
		t.Fatalf("Temperature = %f, want 0.65", cp.req.Temperature)
	}
}

func TestRuntimeCallPassesImageParts(t *testing.T) {
	cp := &capturingProvider{}
	rt := newTestRuntime(cp, LLMConfig{
		Default: DefaultConfig{Model: "qwen3-max"},
	})

	_, err := rt.Call(context.Background(), ProviderRequest{
		Model:  "qwen3-max",
		System: "sys",
		UserParts: []ContentPart{
			{Type: "image_url", ImageURL: "data:image/png;base64,abc"},
			{Type: "text", Text: "describe"},
		},
	})
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}
	if len(cp.req.UserParts) != 2 {
		t.Fatalf("UserParts = %#v, want 2 parts", cp.req.UserParts)
	}
	if cp.req.UserParts[0].Type != "image_url" || cp.req.UserParts[0].ImageURL != "data:image/png;base64,abc" {
		t.Fatalf("first part = %#v", cp.req.UserParts[0])
	}
}

func TestChatStampsResolvedRoleSeparatelyFromModel(t *testing.T) {
	mp := newMockProvider()
	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "default-model"},
		Orchestration: map[string]RoleConfig{
			"narrative_verdict": {
				Model: strPtr("resolved-model"),
			},
		},
		PromptRoles: map[string]string{
			"role_test": "narrative_verdict",
		},
	})

	resp, err := rt.Chat(context.Background(), PromptSpec{ID: "test/role_test"}, map[string]any{"query": "hello"})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}
	if resp.Role != "narrative_verdict" {
		t.Fatalf("Response.Role = %q, want %q", resp.Role, "narrative_verdict")
	}
	if resp.Model != "resolved-model" {
		t.Fatalf("Response.Model = %q, want %q", resp.Model, "resolved-model")
	}
	if resp.Role == resp.Model {
		t.Fatalf("Response.Role and Response.Model should remain distinct, both = %q", resp.Role)
	}
}

func TestExecuteEnsembleCollectsAll(t *testing.T) {
	mp := newMockProvider()
	mp.responses["model-a"] = ProviderResponse{Text: "answer A", Model: "model-a", Tokens: TokenUsage{TotalTokens: 10}}
	mp.responses["model-b"] = ProviderResponse{Text: "answer B", Model: "model-b", Tokens: TokenUsage{TotalTokens: 20}}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "model-a"},
	})

	mode := ModeEnsemble
	rc := ResolvedConfig{
		Model:  "model-a",
		Mode:   mode,
		Models: []string{"model-a", "model-b"},
	}
	built := BuiltPrompt{System: "sys", User: "usr"}

	resp, err := rt.executeEnsemble(context.Background(), rc, built, nil)
	if err != nil {
		t.Fatalf("executeEnsemble() error = %v", err)
	}

	// Both responses should be merged.
	if resp.Text == "" {
		t.Fatal("Text should not be empty")
	}
	if len(resp.ModeTrace) != 2 {
		t.Fatalf("ModeTrace len = %d, want 2", len(resp.ModeTrace))
	}
	if resp.Tokens.TotalTokens != 30 {
		t.Fatalf("TotalTokens = %d, want 30", resp.Tokens.TotalTokens)
	}
}

func TestExecuteEnsemblePartialFailure(t *testing.T) {
	mp := newMockProvider()
	mp.responses["model-a"] = ProviderResponse{Text: "answer A", Model: "model-a"}
	mp.errors["model-b"] = &APIError{StatusCode: 400, Body: "bad request"}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "model-a"},
	})

	rc := ResolvedConfig{
		Mode:   ModeEnsemble,
		Models: []string{"model-a", "model-b"},
	}
	built := BuiltPrompt{System: "sys", User: "usr"}

	resp, err := rt.executeEnsemble(context.Background(), rc, built, nil)
	if err != nil {
		t.Fatalf("executeEnsemble() error = %v (should succeed with partial results)", err)
	}

	if len(resp.ModeTrace) != 1 {
		t.Fatalf("ModeTrace len = %d, want 1 (only model-a succeeded)", len(resp.ModeTrace))
	}
}

func TestExecuteCascadeStopsAtFirstSuccess(t *testing.T) {
	mp := newMockProvider()
	mp.errors["model-a"] = &APIError{StatusCode: 500, Body: "error"}
	mp.responses["model-b"] = ProviderResponse{Text: "fallback", Model: "model-b"}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "model-a"},
	})

	rc := ResolvedConfig{
		Mode:   ModeCascade,
		Models: []string{"model-a", "model-b"},
	}
	built := BuiltPrompt{System: "sys", User: "usr"}

	resp, err := rt.executeCascade(context.Background(), rc, built, nil)
	if err != nil {
		t.Fatalf("executeCascade() error = %v", err)
	}
	if resp.Text != "fallback" {
		t.Fatalf("Text = %q, want %q", resp.Text, "fallback")
	}
	if len(resp.ModeTrace) != 1 {
		t.Fatalf("ModeTrace len = %d, want 1", len(resp.ModeTrace))
	}
	if resp.ModeTrace[0].Model != "model-b" {
		t.Fatalf("ModeTrace[0].Model = %q, want model-b", resp.ModeTrace[0].Model)
	}
}

func TestExecuteCascadeAllFail(t *testing.T) {
	mp := newMockProvider()
	mp.errors["model-a"] = &APIError{StatusCode: 400, Body: "bad"}
	mp.errors["model-b"] = &APIError{StatusCode: 400, Body: "bad"}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "model-a"},
	})

	rc := ResolvedConfig{
		Mode:   ModeCascade,
		Models: []string{"model-a", "model-b"},
	}
	built := BuiltPrompt{System: "sys", User: "usr"}

	_, err := rt.executeCascade(context.Background(), rc, built, nil)
	if err == nil {
		t.Fatal("expected error when all cascade models fail")
	}
}

func TestExecuteChallengeChains(t *testing.T) {
	mp := newMockProvider()
	mp.responses["gen-model"] = ProviderResponse{Text: "generated answer", Model: "gen-model", Tokens: TokenUsage{TotalTokens: 10}}
	mp.responses["chal-model"] = ProviderResponse{Text: "challenged answer", Model: "chal-model", Tokens: TokenUsage{TotalTokens: 15}}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "gen-model"},
	})

	rc := ResolvedConfig{
		Mode:                  ModeChallenge,
		Models:                []string{"gen-model", "chal-model"},
		ChallengeInstructions: "Challenge this response.",
	}
	built := BuiltPrompt{System: "sys", User: "usr"}

	resp, err := rt.executeChallenge(context.Background(), rc, built, nil)
	if err != nil {
		t.Fatalf("executeChallenge() error = %v", err)
	}

	if len(resp.ModeTrace) != 2 {
		t.Fatalf("ModeTrace len = %d, want 2", len(resp.ModeTrace))
	}
	if resp.ModeTrace[0].Role != "generator" {
		t.Fatalf("ModeTrace[0].Role = %q, want generator", resp.ModeTrace[0].Role)
	}
	if resp.ModeTrace[1].Role != "challenger" {
		t.Fatalf("ModeTrace[1].Role = %q, want challenger", resp.ModeTrace[1].Role)
	}
	if resp.Tokens.TotalTokens != 25 {
		t.Fatalf("TotalTokens = %d, want 25", resp.Tokens.TotalTokens)
	}
	// Both texts should be present.
	if resp.Text == "" {
		t.Fatal("Text should not be empty")
	}
}

func TestExecuteChallengeFallsBackWhenChallengerFails(t *testing.T) {
	mp := newMockProvider()
	mp.responses["gen-model"] = ProviderResponse{
		Text:   "generated answer",
		Model:  "gen-model",
		Tokens: TokenUsage{TotalTokens: 10},
	}
	mp.errors["chal-model"] = context.DeadlineExceeded

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "gen-model"},
	})

	rc := ResolvedConfig{
		Mode:                  ModeChallenge,
		Models:                []string{"gen-model", "chal-model"},
		ChallengeInstructions: "Challenge this response.",
	}

	resp, err := rt.executeChallenge(context.Background(), rc, BuiltPrompt{System: "sys", User: "usr"}, nil)
	if err != nil {
		t.Fatalf("executeChallenge() error = %v", err)
	}
	if resp.Text != "generated answer" {
		t.Fatalf("Text = %q, want generator fallback", resp.Text)
	}
	if resp.Model != "gen-model" {
		t.Fatalf("Model = %q, want gen-model", resp.Model)
	}
	if len(resp.ModeTrace) != 1 || resp.ModeTrace[0].Role != "generator" {
		t.Fatalf("ModeTrace = %+v, want generator-only fallback trace", resp.ModeTrace)
	}
	if len(resp.Attempts) < 2 {
		t.Fatalf("Attempts len = %d, want challenger failures preserved", len(resp.Attempts))
	}
	if resp.Attempts[len(resp.Attempts)-1].Error == "" {
		t.Fatalf("expected final attempt to record challenger failure, got %+v", resp.Attempts)
	}
}

func TestExecuteChallengeSingleModelErrors(t *testing.T) {
	mp := newMockProvider()
	mp.responses["only-one"] = ProviderResponse{Text: "single response", Model: "only-one", Tokens: TokenUsage{TotalTokens: 10}}
	rt := newTestRuntime(mp, LLMConfig{Default: DefaultConfig{Model: "only-one"}})

	rc := ResolvedConfig{
		Mode:   ModeChallenge,
		Model:  "only-one",
		Models: []string{"only-one"},
	}
	built := BuiltPrompt{System: "sys", User: "usr"}

	_, err := rt.executeChallenge(context.Background(), rc, built, nil)
	if err == nil {
		t.Fatal("expected error for single-model challenge config")
	}
}

func TestExecuteDebateRounds(t *testing.T) {
	mp := newMockProvider()
	// All models use the same mock, distinguished by model name.
	mp.responses["bull-model"] = ProviderResponse{Text: "bull argument", Model: "bull-model", Tokens: TokenUsage{TotalTokens: 10}}
	mp.responses["bear-model"] = ProviderResponse{Text: "bear argument", Model: "bear-model", Tokens: TokenUsage{TotalTokens: 10}}
	mp.responses["judge-model"] = ProviderResponse{Text: "verdict", Model: "judge-model", Tokens: TokenUsage{TotalTokens: 15}}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "bull-model"},
	})

	rc := ResolvedConfig{
		Mode:   ModeDebate,
		Rounds: 2,
		Roles: map[string]ResolvedConfig{
			"bull":  {Model: "bull-model"},
			"bear":  {Model: "bear-model"},
			"judge": {Model: "judge-model"},
		},
	}
	built := BuiltPrompt{System: "sys", User: "usr"}

	resp, err := rt.executeDebate(context.Background(), rc, built, nil)
	if err != nil {
		t.Fatalf("executeDebate() error = %v", err)
	}

	// 2 rounds * (bull + bear) + judge = 5 steps.
	if len(resp.ModeTrace) != 5 {
		t.Fatalf("ModeTrace len = %d, want 5 (2 rounds of bull+bear + judge)", len(resp.ModeTrace))
	}

	// Final text should be the judge's verdict.
	if resp.Text != "verdict" {
		t.Fatalf("Text = %q, want %q", resp.Text, "verdict")
	}

	// Total tokens: 10*4 (2 rounds * 2 roles) + 15 (judge) = 55
	if resp.Tokens.TotalTokens != 55 {
		t.Fatalf("TotalTokens = %d, want 55", resp.Tokens.TotalTokens)
	}
}

func TestExecuteDebateMissingRoles(t *testing.T) {
	mp := newMockProvider()
	rt := newTestRuntime(mp, LLMConfig{})

	rc := ResolvedConfig{
		Mode:   ModeDebate,
		Rounds: 1,
		Roles: map[string]ResolvedConfig{
			"bull": {Model: "bull-model"},
			// Missing bear and judge.
		},
	}
	built := BuiltPrompt{System: "sys", User: "usr"}

	_, err := rt.executeDebate(context.Background(), rc, built, nil)
	if err == nil {
		t.Fatal("expected error when debate is missing required roles")
	}
}

func TestBoundedConcurrency(t *testing.T) {
	// Use concurrency limit of 2 and run ensemble with 4 models.
	// Verify that at most 2 calls run concurrently.
	var maxConcurrent atomic.Int32
	var current atomic.Int32

	mp := &mockProvider{
		name:      "mock",
		responses: make(map[string]ProviderResponse),
		errors:    make(map[string]error),
	}

	// Override Call to track concurrency.
	type concurrencyProvider struct {
		Provider
	}
	cp := &trackingProvider{
		inner:         mp,
		current:       &current,
		maxConcurrent: &maxConcurrent,
	}

	rt := NewRuntime(RuntimeConfig{
		Provider:    cp,
		Concurrency: 2,
	})

	rc := ResolvedConfig{
		Mode:   ModeEnsemble,
		Models: []string{"m1", "m2", "m3", "m4"},
	}
	built := BuiltPrompt{System: "sys", User: "usr"}

	resp, err := rt.executeEnsemble(context.Background(), rc, built, nil)
	if err != nil {
		t.Fatalf("executeEnsemble() error = %v", err)
	}
	if len(resp.ModeTrace) != 4 {
		t.Fatalf("ModeTrace len = %d, want 4", len(resp.ModeTrace))
	}

	// Verify semaphore limited concurrency to 2.
	if maxConcurrent.Load() > 2 {
		t.Fatalf("maxConcurrent = %d, want <= 2", maxConcurrent.Load())
	}
}

// trackingProvider wraps a provider and tracks concurrency.
type trackingProvider struct {
	inner         Provider
	current       *atomic.Int32
	maxConcurrent *atomic.Int32
}

func (tp *trackingProvider) Name() string { return tp.inner.Name() }

func (tp *trackingProvider) Call(ctx context.Context, req ProviderRequest) (ProviderResponse, error) {
	c := tp.current.Add(1)
	for {
		old := tp.maxConcurrent.Load()
		if c <= old || tp.maxConcurrent.CompareAndSwap(old, c) {
			break
		}
	}

	time.Sleep(50 * time.Millisecond) // simulate work
	tp.current.Add(-1)

	return tp.inner.Call(ctx, req)
}

func TestChatDispatchesByMode(t *testing.T) {
	mp := newMockProvider()

	mode := ModeEnsemble
	merge := MergeUnion
	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "default-model"},
		Orchestration: map[string]RoleConfig{
			"test_role": {
				Mode:   &mode,
				Models: []string{"m1", "m2"},
				Merge:  &merge,
			},
		},
	})

	spec := PromptSpec{
		ID:   "some/path/test_role",
		Role: "test",
	}

	resp, err := rt.Chat(context.Background(), spec, map[string]any{"query": "hello"})
	if err != nil {
		t.Fatalf("Chat() error = %v", err)
	}

	// Should dispatch to ensemble (2 models).
	if len(resp.ModeTrace) != 2 {
		t.Fatalf("ModeTrace len = %d, want 2 (ensemble with 2 models)", len(resp.ModeTrace))
	}
}

func TestRetryOnServerError(t *testing.T) {
	callCount := 0
	mp := &countingProvider{
		callCount: &callCount,
		failFirst: 2,
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	rc := rt.config.Resolve("test")
	built := BuiltPrompt{System: "sys", User: "usr"}

	resp, err := rt.executeSingle(context.Background(), rc, built, nil)
	if err != nil {
		t.Fatalf("executeSingle() error = %v (should succeed after retries)", err)
	}
	if resp.Text != "success" {
		t.Fatalf("Text = %q, want %q", resp.Text, "success")
	}
	if len(resp.Attempts) != 3 {
		t.Fatalf("Attempts = %d, want 3 (2 failures + 1 success)", len(resp.Attempts))
	}
}

func TestRetryHonorsCustomMaxAttempts(t *testing.T) {
	callCount := 0
	mp := &countingProvider{
		callCount: &callCount,
		failFirst: 5,
	}

	rt := NewRuntime(RuntimeConfig{
		Provider:    mp,
		LLMConfig:   LLMConfig{Default: DefaultConfig{Model: "test-model"}},
		Concurrency: 4,
		MaxAttempts: 1,
	})

	rc := rt.config.Resolve("test")
	built := BuiltPrompt{System: "sys", User: "usr"}

	resp, err := rt.executeSingle(context.Background(), rc, built, nil)
	if err == nil {
		t.Fatal("executeSingle() error = nil, want failure after one allowed attempt")
	}
	if len(resp.Attempts) != 1 {
		t.Fatalf("Attempts = %d, want 1 when MaxAttempts is set to 1", len(resp.Attempts))
	}
	if callCount != 1 {
		t.Fatalf("callCount = %d, want 1 when MaxAttempts is set to 1", callCount)
	}
}

// countingProvider fails the first N calls with a retryable 500, then succeeds.
type countingProvider struct {
	callCount *int
	failFirst int
}

func (cp *countingProvider) Name() string { return "counting" }

func (cp *countingProvider) Call(ctx context.Context, req ProviderRequest) (ProviderResponse, error) {
	*cp.callCount++
	if *cp.callCount <= cp.failFirst {
		return ProviderResponse{}, &APIError{StatusCode: 500, Body: "error"}
	}
	return ProviderResponse{
		Text:   "success",
		Model:  req.Model,
		Tokens: TokenUsage{TotalTokens: 10},
	}, nil
}
