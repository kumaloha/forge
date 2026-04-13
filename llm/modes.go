package llm

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"
)

// executeSingle runs a single model call.
func (r *Runtime) executeSingle(ctx context.Context, rc ResolvedConfig, built BuiltPrompt, schema *Schema) (Response, error) {
	resp, err := r.chatSingle(ctx, rc.Model, built, rc.Search, rc.Thinking, rc.Temperature, schema)
	if err != nil {
		return resp, err
	}
	resp.ModeTrace = []ModeStep{{
		Mode:    ModeSingle,
		Role:    "single",
		Model:   resp.Model,
		Latency: resp.Latency,
		Tokens:  resp.Tokens,
	}}
	return resp, nil
}

// executeEnsemble runs N models in parallel and merges results.
func (r *Runtime) executeEnsemble(ctx context.Context, rc ResolvedConfig, built BuiltPrompt, schema *Schema) (Response, error) {
	responses, aggregate, err := r.collectEnsembleResponses(ctx, rc, built, schema)
	if err != nil {
		return aggregate, err
	}

	texts := make([]string, 0, len(responses))
	for _, resp := range responses {
		texts = append(texts, resp.Text)
	}
	aggregate.Text = strings.Join(texts, "\n---\n")
	return aggregate, nil
}

func (r *Runtime) collectEnsembleResponses(ctx context.Context, rc ResolvedConfig, built BuiltPrompt, schema *Schema) ([]Response, Response, error) {
	models := rc.Models
	if len(models) == 0 {
		return nil, Response{}, fmt.Errorf("llm: ensemble mode requires models")
	}

	type result struct {
		resp Response
		err  error
	}

	results := make([]result, len(models))
	var wg sync.WaitGroup

	for i, model := range models {
		wg.Add(1)
		go func(idx int, m string) {
			defer wg.Done()
			resp, err := r.chatSingle(ctx, m, built, rc.Search, rc.Thinking, rc.Temperature, schema)
			results[idx] = result{resp: resp, err: err}
		}(i, model)
	}
	wg.Wait()

	// Collect successful responses.
	var responses []Response
	var allAttempts []AttemptTrace
	var steps []ModeStep
	var totalTokens TokenUsage
	var totalLatency time.Duration
	var lastModel string
	var errs []error

	for i, res := range results {
		allAttempts = append(allAttempts, res.resp.Attempts...)
		if res.err != nil {
			errs = append(errs, fmt.Errorf("model %s: %w", models[i], res.err))
			continue
		}
		responses = append(responses, res.resp)
		steps = append(steps, ModeStep{
			Mode:    ModeEnsemble,
			Role:    fmt.Sprintf("ensemble_%d", i),
			Model:   res.resp.Model,
			Latency: res.resp.Latency,
			Tokens:  res.resp.Tokens,
		})
		totalTokens.PromptTokens += res.resp.Tokens.PromptTokens
		totalTokens.CompletionTokens += res.resp.Tokens.CompletionTokens
		totalTokens.TotalTokens += res.resp.Tokens.TotalTokens
		if res.resp.Latency > totalLatency {
			totalLatency = res.resp.Latency
		}
		lastModel = res.resp.Model
	}

	aggregate := Response{
		Model:     lastModel,
		Attempts:  allAttempts,
		Latency:   totalLatency,
		Tokens:    totalTokens,
		ModeTrace: steps,
	}

	if len(responses) == 0 {
		return nil, aggregate, fmt.Errorf("llm: all ensemble models failed: %v", errs)
	}

	return responses, aggregate, nil
}

// executeCascade tries models sequentially, returning the first success.
func (r *Runtime) executeCascade(ctx context.Context, rc ResolvedConfig, built BuiltPrompt, schema *Schema) (Response, error) {
	models := rc.Models
	if len(models) == 0 {
		models = []string{rc.Model}
	}

	var allAttempts []AttemptTrace
	var lastErr error

	for i, model := range models {
		resp, err := r.chatSingle(ctx, model, built, rc.Search, rc.Thinking, rc.Temperature, schema)
		allAttempts = append(allAttempts, resp.Attempts...)

		if err == nil {
			resp.Attempts = allAttempts
			resp.ModeTrace = []ModeStep{{
				Mode:    ModeCascade,
				Role:    fmt.Sprintf("cascade_%d", i),
				Model:   resp.Model,
				Latency: resp.Latency,
				Tokens:  resp.Tokens,
			}}
			return resp, nil
		}
		lastErr = err
	}

	return Response{Attempts: allAttempts}, fmt.Errorf("llm: all cascade models failed: %w", lastErr)
}

// executeChallenge runs a generator (model A) then a challenger (model B).
// If the challenger fails after the generator succeeds, it falls back to the
// generator response while preserving challenger attempt traces.
func (r *Runtime) executeChallenge(ctx context.Context, rc ResolvedConfig, built BuiltPrompt, schema *Schema) (Response, error) {
	_, _, aggregate, err := r.runChallengePair(ctx, rc, built, schema)
	return aggregate, err
}

func (r *Runtime) runChallengePair(ctx context.Context, rc ResolvedConfig, built BuiltPrompt, schema *Schema) (Response, Response, Response, error) {
	models := rc.Models
	if len(models) < 2 {
		return Response{}, Response{}, Response{}, fmt.Errorf("llm: challenge mode requires exactly 2 models")
	}

	genResp, err := r.chatSingle(ctx, models[0], built, rc.Search, rc.Thinking, rc.Temperature, schema)
	if err != nil {
		return genResp, Response{}, genResp, fmt.Errorf("llm: challenge generator failed: %w", err)
	}

	challengeInstructions := rc.ChallengeInstructions
	if challengeInstructions == "" {
		challengeInstructions = "Review the following response critically. Identify any errors, logical flaws, missing considerations, or unsupported claims. Then provide an improved response."
	}

	challengePrompt := BuiltPrompt{
		System: built.System,
		User: fmt.Sprintf("%s\n\n--- Original Response ---\n%s\n\n--- Challenge Instructions ---\n%s",
			built.User, genResp.Text, challengeInstructions),
	}

	chalResp, err := r.chatSingle(ctx, models[1], challengePrompt, rc.Search, rc.Thinking, rc.Temperature, schema)
	if err != nil {
		fallback := genResp
		fallback.Attempts = append(append([]AttemptTrace(nil), genResp.Attempts...), chalResp.Attempts...)
		fallback.ModeTrace = []ModeStep{
			{Mode: ModeChallenge, Role: "generator", Model: genResp.Model, Latency: genResp.Latency, Tokens: genResp.Tokens},
		}
		return genResp, Response{}, fallback, nil
	}

	allAttempts := append(genResp.Attempts, chalResp.Attempts...)
	steps := []ModeStep{
		{Mode: ModeChallenge, Role: "generator", Model: genResp.Model, Latency: genResp.Latency, Tokens: genResp.Tokens},
		{Mode: ModeChallenge, Role: "challenger", Model: chalResp.Model, Latency: chalResp.Latency, Tokens: chalResp.Tokens},
	}

	totalTokens := TokenUsage{
		PromptTokens:     genResp.Tokens.PromptTokens + chalResp.Tokens.PromptTokens,
		CompletionTokens: genResp.Tokens.CompletionTokens + chalResp.Tokens.CompletionTokens,
		TotalTokens:      genResp.Tokens.TotalTokens + chalResp.Tokens.TotalTokens,
	}

	aggregate := Response{
		Text:      genResp.Text + "\n---\n" + chalResp.Text,
		Model:     chalResp.Model,
		Attempts:  allAttempts,
		Latency:   genResp.Latency + chalResp.Latency,
		Tokens:    totalTokens,
		ModeTrace: steps,
	}
	return genResp, chalResp, aggregate, nil
}

// executeDebate runs bull/bear/judge rounds.
func (r *Runtime) executeDebate(ctx context.Context, rc ResolvedConfig, built BuiltPrompt, schema *Schema) (Response, error) {
	bullRC, hasBull := rc.Roles["bull"]
	bearRC, hasBear := rc.Roles["bear"]
	judgeRC, hasJudge := rc.Roles["judge"]

	if !hasBull || !hasBear || !hasJudge {
		return Response{}, fmt.Errorf("llm: debate mode requires bull, bear, judge sub-roles")
	}

	rounds := rc.Rounds
	if rounds <= 0 {
		rounds = 1
	}

	var steps []ModeStep
	var allAttempts []AttemptTrace
	var totalTokens TokenUsage
	var totalLatency time.Duration

	var bullText, bearText string

	for round := range rounds {
		// Bull argues.
		bullPrompt := built
		if round > 0 {
			bullPrompt = BuiltPrompt{
				System: built.System,
				User: fmt.Sprintf("%s\n\n--- Bear's argument (round %d) ---\n%s\n\nProvide your counter-argument.",
					built.User, round, bearText),
			}
		}

		bullResp, err := r.chatSingle(ctx, bullRC.Model, bullPrompt, bullRC.Search, bullRC.Thinking, bullRC.Temperature, nil)
		if err != nil {
			return Response{Attempts: allAttempts}, fmt.Errorf("llm: debate bull round %d failed: %w", round+1, err)
		}
		bullText = bullResp.Text
		allAttempts = append(allAttempts, bullResp.Attempts...)
		steps = append(steps, ModeStep{
			Mode: ModeDebate, Role: fmt.Sprintf("bull_round_%d", round+1),
			Model: bullResp.Model, Latency: bullResp.Latency, Tokens: bullResp.Tokens,
		})
		totalTokens = addTokens(totalTokens, bullResp.Tokens)
		totalLatency += bullResp.Latency

		// Bear argues against bull.
		bearPrompt := BuiltPrompt{
			System: built.System,
			User: fmt.Sprintf("%s\n\n--- Bull's argument (round %d) ---\n%s\n\nProvide your counter-argument.",
				built.User, round+1, bullText),
		}

		bearResp, err := r.chatSingle(ctx, bearRC.Model, bearPrompt, bearRC.Search, bearRC.Thinking, bearRC.Temperature, nil)
		if err != nil {
			return Response{Attempts: allAttempts}, fmt.Errorf("llm: debate bear round %d failed: %w", round+1, err)
		}
		bearText = bearResp.Text
		allAttempts = append(allAttempts, bearResp.Attempts...)
		steps = append(steps, ModeStep{
			Mode: ModeDebate, Role: fmt.Sprintf("bear_round_%d", round+1),
			Model: bearResp.Model, Latency: bearResp.Latency, Tokens: bearResp.Tokens,
		})
		totalTokens = addTokens(totalTokens, bearResp.Tokens)
		totalLatency += bearResp.Latency
	}

	// Judge synthesizes.
	judgePrompt := BuiltPrompt{
		System: built.System,
		User: fmt.Sprintf("%s\n\n--- Bull's final argument ---\n%s\n\n--- Bear's final argument ---\n%s\n\nSynthesize both perspectives into a balanced verdict.",
			built.User, bullText, bearText),
	}

	judgeResp, err := r.chatSingle(ctx, judgeRC.Model, judgePrompt, judgeRC.Search, judgeRC.Thinking, judgeRC.Temperature, schema)
	if err != nil {
		return Response{Attempts: allAttempts}, fmt.Errorf("llm: debate judge failed: %w", err)
	}
	allAttempts = append(allAttempts, judgeResp.Attempts...)
	steps = append(steps, ModeStep{
		Mode: ModeDebate, Role: "judge",
		Model: judgeResp.Model, Latency: judgeResp.Latency, Tokens: judgeResp.Tokens,
	})
	totalTokens = addTokens(totalTokens, judgeResp.Tokens)
	totalLatency += judgeResp.Latency

	return Response{
		Text:      judgeResp.Text,
		Model:     judgeResp.Model,
		Attempts:  allAttempts,
		Latency:   totalLatency,
		Tokens:    totalTokens,
		ModeTrace: steps,
	}, nil
}

func addTokens(a, b TokenUsage) TokenUsage {
	return TokenUsage{
		PromptTokens:     a.PromptTokens + b.PromptTokens,
		CompletionTokens: a.CompletionTokens + b.CompletionTokens,
		TotalTokens:      a.TotalTokens + b.TotalTokens,
	}
}
