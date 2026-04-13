package llm

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestFacetRunnerRunWithSpec(t *testing.T) {
	mp := newMockProvider()
	mp.responses["test-model"] = ProviderResponse{
		Text:  `{"score": 7, "reason": "facet ok"}`,
		Model: "test-model",
	}

	rt := newTestRuntime(mp, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	runner := NewFacetRunnerWithSpec[scoreResult](rt, PromptSpec{ID: "test/facet"})
	got, err := runner.Run(context.Background(), map[string]any{"input": "hello"}, ExtractOptions[scoreResult]{})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got.Value.Score != 7 {
		t.Fatalf("Score = %d, want 7", got.Value.Score)
	}
}

func TestNewDebateRunnerLoadsPromptIDsAndBuildsJudgeVars(t *testing.T) {
	root := t.TempDir()
	writePrompt := func(id string) {
		fullPath := filepath.Join(root, "prompts", filepath.FromSlash(id)+".yaml")
		if err := os.MkdirAll(filepath.Dir(fullPath), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		content := "id: " + id + "\nrole: test\n"
		if err := os.WriteFile(fullPath, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}
	writePrompt("test/debate/bull")
	writePrompt("test/debate/bear")
	writePrompt("test/debate/judge")

	cp := &capturingSequenceProvider{responses: []ProviderResponse{
		{Text: `{"score": 4, "reason": "bull"}`, Model: "test-model"},
		{Text: `{"score": 6, "reason": "bear"}`, Model: "test-model"},
		{Text: `{"score": 8, "reason": "judge"}`, Model: "test-model"},
	}}
	rt := newTestRuntime(cp, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	runner, err := NewDebateRunner[scoreResult, scoreResult, scoreResult](
		rt,
		NewPromptLoaderFromDir(filepath.Join(root, "prompts")),
		"test/debate/bull",
		"test/debate/bear",
		"test/debate/judge",
	)
	if err != nil {
		t.Fatalf("NewDebateRunner() error = %v", err)
	}

	got, err := runner.Run(context.Background(), map[string]any{"input": "base prompt"}, func(base map[string]any, bull, bear scoreResult) map[string]any {
		base["input"] = base["input"].(string) + "\njudge sees bull=" + bull.Reason + " bear=" + bear.Reason
		return base
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got.Bull.Value.Score != 4 || got.Bear.Value.Score != 6 || got.Judge.Value.Score != 8 {
		t.Fatalf("unexpected scores = %+v", got)
	}
	if len(cp.requests) != 3 {
		t.Fatalf("request count = %d, want 3", len(cp.requests))
	}
	if strings.Contains(cp.requests[0].User, "judge sees") || strings.Contains(cp.requests[1].User, "judge sees") {
		t.Fatalf("bull/bear prompt should not contain judge vars: %+v", cp.requests)
	}
	if !strings.Contains(cp.requests[2].User, "judge sees bull=bull bear=bear") {
		t.Fatalf("judge prompt missing derived vars: %q", cp.requests[2].User)
	}
}

func TestTextDebateRunnerUsesRawBullBearTextForJudge(t *testing.T) {
	root := t.TempDir()
	writePrompt := func(id string) {
		fullPath := filepath.Join(root, "prompts", filepath.FromSlash(id)+".yaml")
		if err := os.MkdirAll(filepath.Dir(fullPath), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		content := "id: " + id + "\nrole: test\n"
		if err := os.WriteFile(fullPath, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}
	writePrompt("test/textdebate/bull")
	writePrompt("test/textdebate/bear")
	writePrompt("test/textdebate/judge")

	cp := &capturingSequenceProvider{responses: []ProviderResponse{
		{Text: "bull says up", Model: "test-model"},
		{Text: "bear says down", Model: "test-model"},
		{Text: `{"score": 5, "reason": "judge ok"}`, Model: "test-model"},
	}}
	rt := newTestRuntime(cp, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	runner, err := NewTextDebateRunner[scoreResult](
		rt,
		NewPromptLoaderFromDir(filepath.Join(root, "prompts")),
		"test/textdebate/bull",
		"test/textdebate/bear",
		"test/textdebate/judge",
	)
	if err != nil {
		t.Fatalf("NewTextDebateRunner() error = %v", err)
	}

	got, err := runner.Run(context.Background(), map[string]any{"input": "base"}, func(base map[string]any, bull, bear Response) map[string]any {
		base["input"] = base["input"].(string) + "\nB=" + bull.Text + "\nR=" + bear.Text
		return base
	})
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got.Judge.Value.Score != 5 {
		t.Fatalf("Judge score = %d, want 5", got.Judge.Value.Score)
	}
	if !strings.Contains(cp.requests[2].User, "B=bull says up") || !strings.Contains(cp.requests[2].User, "R=bear says down") {
		t.Fatalf("judge prompt missing raw bull/bear text: %q", cp.requests[2].User)
	}
}

func TestRunNamedTextDebatesParallel(t *testing.T) {
	root := t.TempDir()
	writePrompt := func(id string) {
		fullPath := filepath.Join(root, "prompts", filepath.FromSlash(id)+".yaml")
		if err := os.MkdirAll(filepath.Dir(fullPath), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		content := "id: " + id + "\nrole: " + id + "\n"
		if err := os.WriteFile(fullPath, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}
	for _, id := range []string{
		"test/group/a/bull", "test/group/a/bear", "test/group/a/judge",
		"test/group/b/bull", "test/group/b/bear", "test/group/b/judge",
	} {
		writePrompt(id)
	}

	rt := newTestRuntime(&routingProvider{}, LLMConfig{
		Default: DefaultConfig{Model: "test-model"},
	})

	runnerA, err := NewTextDebateRunner[scoreResult](rt, NewPromptLoaderFromDir(filepath.Join(root, "prompts")), "test/group/a/bull", "test/group/a/bear", "test/group/a/judge")
	if err != nil {
		t.Fatalf("runnerA error = %v", err)
	}
	runnerB, err := NewTextDebateRunner[scoreResult](rt, NewPromptLoaderFromDir(filepath.Join(root, "prompts")), "test/group/b/bull", "test/group/b/bear", "test/group/b/judge")
	if err != nil {
		t.Fatalf("runnerB error = %v", err)
	}

	got, err := RunNamedTextDebatesParallel(context.Background(), map[string]any{"input": "base"}, func(base map[string]any, bull, bear Response) map[string]any {
		base["input"] = base["input"].(string) + "\n" + bull.Text + "\n" + bear.Text
		return base
	}, []NamedTextDebate[scoreResult]{
		{Name: "a", Runner: runnerA},
		{Name: "b", Runner: runnerB},
	})
	if err != nil {
		t.Fatalf("RunNamedTextDebatesParallel() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("len(got) = %d, want 2", len(got))
	}
	if got["a"].Judge.Value.Score != 1 {
		t.Fatalf("unexpected a judge score: %+v", got["a"].Judge.Value)
	}
	if got["b"].Judge.Value.Score != 2 {
		t.Fatalf("unexpected b judge score: %+v", got["b"].Judge.Value)
	}
}

type capturingSequenceProvider struct {
	responses []ProviderResponse
	requests  []ProviderRequest
	mu        sync.Mutex
}

func (p *capturingSequenceProvider) Name() string { return "capturing-sequence" }

func (p *capturingSequenceProvider) Call(_ context.Context, req ProviderRequest) (ProviderResponse, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.requests = append(p.requests, req)
	if len(p.requests) > len(p.responses) {
		return ProviderResponse{}, nil
	}
	return p.responses[len(p.requests)-1], nil
}

type routingProvider struct{}

func (p *routingProvider) Name() string { return "routing" }

func (p *routingProvider) Call(_ context.Context, req ProviderRequest) (ProviderResponse, error) {
	system := req.System
	switch {
	case strings.Contains(system, "test/group/a/bull"):
		return ProviderResponse{Text: "a bull", Model: req.Model}, nil
	case strings.Contains(system, "test/group/a/bear"):
		return ProviderResponse{Text: "a bear", Model: req.Model}, nil
	case strings.Contains(system, "test/group/a/judge"):
		return ProviderResponse{Text: `{"score": 1, "reason": "a judge"}`, Model: req.Model}, nil
	case strings.Contains(system, "test/group/b/bull"):
		return ProviderResponse{Text: "b bull", Model: req.Model}, nil
	case strings.Contains(system, "test/group/b/bear"):
		return ProviderResponse{Text: "b bear", Model: req.Model}, nil
	case strings.Contains(system, "test/group/b/judge"):
		return ProviderResponse{Text: `{"score": 2, "reason": "b judge"}`, Model: req.Model}, nil
	default:
		return ProviderResponse{Text: `{"score": 0, "reason": "default"}`, Model: req.Model}, nil
	}
}
