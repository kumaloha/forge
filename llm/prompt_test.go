package llm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func testPromptRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()

	writePrompt := func(id, content string) {
		path := filepath.Join(root, "prompts", filepath.FromSlash(id)+".yaml")
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatalf("MkdirAll() error = %v", err)
		}
		if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
			t.Fatalf("WriteFile() error = %v", err)
		}
	}

	writePrompt("cognition/fundamental/comprehensibility_gate", `id: cognition/fundamental/comprehensibility_gate
version: v1
role: You are explaining a business model.
output_format: Output JSON
`)

	writePrompt("portfolio/view/synthesize", `id: portfolio/view/synthesize
version: v1
role: You are a portfolio synthesizer.
definitions: some definitions
criteria: some criteria
boundaries: some boundaries
few_shots:
  - input: example input
    output: example output
    reason: example reason
output_format: Output JSON
`)

	return root
}

func TestLoadRealPrompt(t *testing.T) {
	root := testPromptRoot(t)
	loader := NewPromptLoaderFromDir(filepath.Join(root, "prompts"))

	spec, err := loader.Load("cognition/fundamental/comprehensibility_gate")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if spec.ID != "cognition/fundamental/comprehensibility_gate" {
		t.Fatalf("ID = %q, want cognition/fundamental/comprehensibility_gate", spec.ID)
	}
	if spec.Version != "v1" {
		t.Fatalf("Version = %q, want v1", spec.Version)
	}
	if !strings.Contains(spec.Role, "business model") {
		t.Fatalf("Role should mention business model, got %q", spec.Role)
	}
	if !strings.Contains(spec.OutputFormat, "JSON") {
		t.Fatalf("OutputFormat should mention JSON, got %q", spec.OutputFormat)
	}
}

func TestLoadPromptWithAllFields(t *testing.T) {
	root := testPromptRoot(t)
	loader := NewPromptLoaderFromDir(filepath.Join(root, "prompts"))

	spec, err := loader.Load("portfolio/view/synthesize")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if spec.Definitions == "" {
		t.Fatal("expected non-empty Definitions")
	}
	if spec.Criteria == "" {
		t.Fatal("expected non-empty Criteria")
	}
	if spec.Boundaries == "" {
		t.Fatal("expected non-empty Boundaries")
	}
	if len(spec.FewShots) == 0 {
		t.Fatal("expected at least one few-shot example")
	}
	if spec.FewShots[0].Input == "" {
		t.Fatal("expected non-empty few-shot input")
	}
	if spec.FewShots[0].Output == "" {
		t.Fatal("expected non-empty few-shot output")
	}
	if spec.FewShots[0].Reason == "" {
		t.Fatal("expected non-empty few-shot reason")
	}
}

func TestLoadInvalidIDReturnsError(t *testing.T) {
	root := testPromptRoot(t)
	loader := NewPromptLoaderFromDir(filepath.Join(root, "prompts"))

	_, err := loader.Load("nonexistent/prompt/id")
	if err == nil {
		t.Fatal("expected error for nonexistent prompt, got nil")
	}
}

func TestLoadIDMismatchReturnsError(t *testing.T) {
	dir := t.TempDir()
	promptDir := filepath.Join(dir, "prompts", "test")
	if err := os.MkdirAll(promptDir, 0o755); err != nil {
		t.Fatalf("MkdirAll() error = %v", err)
	}
	content := `id: wrong/id
version: v1
role: "test"
`
	if err := os.WriteFile(filepath.Join(promptDir, "mismatch.yaml"), []byte(content), 0o644); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	loader := NewPromptLoaderFromDir(filepath.Join(dir, "prompts"))
	_, err := loader.Load("test/mismatch")
	if err == nil {
		t.Fatal("expected error for id mismatch, got nil")
	}
	if !strings.Contains(err.Error(), "id mismatch") {
		t.Fatalf("error should mention id mismatch, got %v", err)
	}
}

func TestLoaderCachesBetweenCalls(t *testing.T) {
	root := testPromptRoot(t)
	loader := NewPromptLoaderFromDir(filepath.Join(root, "prompts"))

	spec1, err := loader.Load("cognition/fundamental/comprehensibility_gate")
	if err != nil {
		t.Fatalf("first Load() error = %v", err)
	}

	spec2, err := loader.Load("cognition/fundamental/comprehensibility_gate")
	if err != nil {
		t.Fatalf("second Load() error = %v", err)
	}

	// Both should return equivalent specs (cache hit)
	if spec1.ID != spec2.ID || spec1.Version != spec2.Version || spec1.Role != spec2.Role {
		t.Fatal("cached result differs from first load")
	}

	// Verify cache is populated
	loader.mu.RLock()
	_, cached := loader.cache["cognition/fundamental/comprehensibility_gate"]
	loader.mu.RUnlock()
	if !cached {
		t.Fatal("expected prompt to be cached after Load()")
	}
}

func TestBuildSystemPromptOmitsEmptySections(t *testing.T) {
	spec := PromptSpec{
		Role:         "You are a test assistant.",
		OutputFormat: "Output JSON: {}",
		// Definitions, Criteria, Boundaries are empty
	}

	built := spec.Build(nil, nil)

	if !strings.Contains(built.System, "test assistant") {
		t.Fatalf("system should contain role, got %q", built.System)
	}
	if !strings.Contains(built.System, "Output JSON") {
		t.Fatalf("system should contain output format, got %q", built.System)
	}
	// Should be exactly role + output_format separated by double newline
	parts := strings.Split(built.System, "\n\n")
	if len(parts) != 2 {
		t.Fatalf("expected 2 system sections, got %d: %q", len(parts), built.System)
	}
}

func TestBuildSystemPromptAllSections(t *testing.T) {
	spec := PromptSpec{
		Role:         "You are a role.",
		Definitions:  "DEFINITIONS here.",
		Criteria:     "CRITERIA here.",
		Boundaries:   "BOUNDARIES here.",
		OutputFormat: "FORMAT here.",
	}

	built := spec.Build(nil, nil)
	parts := strings.Split(built.System, "\n\n")
	if len(parts) != 5 {
		t.Fatalf("expected 5 system sections, got %d", len(parts))
	}
	if parts[0] != "You are a role." {
		t.Fatalf("first section = %q, want role", parts[0])
	}
	if parts[4] != "FORMAT here." {
		t.Fatalf("last section = %q, want format", parts[4])
	}
}

func TestBuildUserWithFewShots(t *testing.T) {
	spec := PromptSpec{
		Role: "role",
		FewShots: []FewShot{
			{Input: "q1", Output: "a1", Reason: "r1"},
			{Input: "q2", Output: "a2"},
		},
	}

	built := spec.Build(map[string]any{"query": "my question"}, nil)

	if !strings.Contains(built.User, "Example 1:") {
		t.Fatalf("user should contain Example 1, got %q", built.User)
	}
	if !strings.Contains(built.User, "Input: q1") {
		t.Fatalf("user should contain first input")
	}
	if !strings.Contains(built.User, "Reason: r1") {
		t.Fatalf("user should contain reason for first example")
	}
	if !strings.Contains(built.User, "Example 2:") {
		t.Fatalf("user should contain Example 2")
	}
	// second example has no reason
	if strings.Contains(built.User, "Reason: \n") {
		t.Fatal("should not contain empty reason line")
	}
	if !strings.Contains(built.User, "my question") {
		t.Fatalf("user should contain query")
	}
}

func TestBuildUserWithMemory(t *testing.T) {
	spec := PromptSpec{Role: "role"}
	memory := []MemoryBlock{
		{Input: "prev-in", Output: "prev-out", Note: "failed validation"},
		{Input: "prev-in-2", Output: "prev-out-2"},
	}

	built := spec.Build(map[string]any{"input": "new input"}, memory)

	if !strings.Contains(built.User, "Previous attempts:") {
		t.Fatalf("user should contain memory header, got %q", built.User)
	}
	if !strings.Contains(built.User, "- Input: prev-in") {
		t.Fatalf("user should contain memory input")
	}
	if !strings.Contains(built.User, "  Note: failed validation") {
		t.Fatalf("user should contain memory note")
	}
	if !strings.Contains(built.User, "new input") {
		t.Fatalf("user should contain input variable")
	}
}

func TestBuildPrefersQueryOverInput(t *testing.T) {
	spec := PromptSpec{Role: "role"}

	built := spec.Build(map[string]any{"query": "the query", "input": "the input"}, nil)

	if !strings.Contains(built.User, "the query") {
		t.Fatalf("should use query when both provided, got %q", built.User)
	}
	// "the input" should NOT appear because query takes precedence
	if strings.Contains(built.User, "the input") {
		t.Fatalf("should not contain input when query is present")
	}
}

func TestBuildEmptyVarsProducesEmptyUser(t *testing.T) {
	spec := PromptSpec{Role: "role"}
	built := spec.Build(nil, nil)
	if built.User != "" {
		t.Fatalf("expected empty user, got %q", built.User)
	}
}
