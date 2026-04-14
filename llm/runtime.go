package llm

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Runtime is the main LLM orchestration engine. It holds a provider, prompt
// loader, config, and a bounded-concurrency semaphore.
type Runtime struct {
	provider    Provider
	prompts     *PromptLoader
	config      LLMConfig
	sem         chan struct{}
	maxAttempts int
}

// RuntimeConfig configures a new Runtime.
type RuntimeConfig struct {
	Provider    Provider
	Prompts     *PromptLoader
	LLMConfig   LLMConfig
	Concurrency int // max parallel LLM calls, default 4
	MaxAttempts int // retry budget for a single provider call path, default 3
}

// NewRuntime creates a Runtime from the given configuration.
func NewRuntime(cfg RuntimeConfig) *Runtime {
	conc := cfg.Concurrency
	if conc <= 0 {
		conc = 4
	}
	maxAttempts := cfg.MaxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 3
	}
	return &Runtime{
		provider:    cfg.Provider,
		prompts:     cfg.Prompts,
		config:      cfg.LLMConfig,
		sem:         make(chan struct{}, conc),
		maxAttempts: maxAttempts,
	}
}

// Chat sends a prompt and returns the raw text response. It resolves the
// role config from the prompt spec's ID to determine mode/model, then
// dispatches to the appropriate mode handler.
func (r *Runtime) Chat(ctx context.Context, spec PromptSpec, vars map[string]any) (Response, error) {
	return r.chatWithSchema(ctx, spec, vars, nil)
}

// Call sends a raw provider request through the runtime retry/concurrency path.
// This is useful for callers that need provider-native capabilities such as
// multimodal image inputs while still relying on forge retry semantics.
func (r *Runtime) Call(ctx context.Context, req ProviderRequest) (Response, error) {
	if r == nil {
		return Response{}, fmt.Errorf("llm: runtime is nil")
	}
	return r.callSingle(ctx, req)
}

// chatWithSchema is the internal entry that supports an optional JSON schema.
// Used by both Chat (schema=nil) and Extract (schema set).
func (r *Runtime) chatWithSchema(ctx context.Context, spec PromptSpec, vars map[string]any, schema *Schema) (Response, error) {
	built := spec.Build(vars, nil)

	roleKey := r.config.ResolvePromptRole(spec.ID)
	rc := r.config.Resolve(roleKey)

	var resp Response
	var err error
	switch rc.Mode {
	case ModeEnsemble:
		resp, err = r.executeEnsemble(ctx, rc, built, schema)
	case ModeCascade:
		resp, err = r.executeCascade(ctx, rc, built, schema)
	case ModeChallenge:
		resp, err = r.executeChallenge(ctx, rc, built, schema)
	case ModeDebate:
		resp, err = r.executeDebate(ctx, rc, built, schema)
	default:
		resp, err = r.executeSingle(ctx, rc, built, schema)
	}

	// Stamp audit fields on every response.
	stampResponseMetadata(&resp, spec.ID, roleKey)
	return resp, err
}

// chatSingle is the core single-call implementation with retry logic.
// Max 2 retries (3 total attempts). Retries on 5xx, timeout, 429.
// Exponential backoff: 1s, 2s.
func (r *Runtime) chatSingle(ctx context.Context, model string, built BuiltPrompt, search, thinking bool, temp float64, schema *Schema) (Response, error) {
	return r.callSingle(ctx, ProviderRequest{
		Model:       model,
		System:      built.System,
		User:        built.User,
		Temperature: temp,
		Search:      search,
		Thinking:    thinking,
		JSONSchema:  schema,
	})
}

func (r *Runtime) callSingle(ctx context.Context, req ProviderRequest) (Response, error) {
	maxAttempts := r.maxAttempts
	if maxAttempts <= 0 {
		maxAttempts = 3
	}
	backoff := [...]time.Duration{time.Second, 2 * time.Second}

	var attempts []AttemptTrace
	var lastErr error

	for i := range maxAttempts {
		start := time.Now()

		pr, err := r.callProvider(ctx, req)

		elapsed := time.Since(start)

		trace := AttemptTrace{
			Model:   req.Model,
			Latency: elapsed,
		}

		if err != nil {
			trace.Error = err.Error()
			attempts = append(attempts, trace)

			var apiErr *APIError
			if errors.As(err, &apiErr) && !apiErr.Retryable() {
				return Response{Attempts: attempts}, err
			}

			lastErr = err
			if i < maxAttempts-1 {
				select {
				case <-ctx.Done():
					return Response{Attempts: attempts}, ctx.Err()
				case <-time.After(backoff[minInt(i, len(backoff)-1)]):
				}
			}
			continue
		}

		trace.Tokens = pr.Tokens
		trace.Truncated = pr.Truncated
		attempts = append(attempts, trace)

		resp := Response{
			Text:     pr.Text,
			Model:    pr.Model,
			Attempts: attempts,
			Latency:  elapsed,
			Tokens:   pr.Tokens,
		}

		if pr.Refusal != nil {
			resp.Refusal = pr.Refusal
			return resp, fmt.Errorf("llm: model refused: %s", pr.Refusal.Reason)
		}

		return resp, nil
	}

	return Response{Attempts: attempts}, fmt.Errorf("llm: all %d attempts failed: %w", maxAttempts, lastErr)
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// callProvider acquires the semaphore, calls the provider, and releases.
func (r *Runtime) callProvider(ctx context.Context, req ProviderRequest) (ProviderResponse, error) {
	select {
	case r.sem <- struct{}{}:
	case <-ctx.Done():
		return ProviderResponse{}, ctx.Err()
	}
	defer func() { <-r.sem }()

	return r.provider.Call(ctx, req)
}

func stampResponseMetadata(resp *Response, promptID, roleKey string) {
	resp.PromptID = promptID
	resp.Role = roleKey
}
