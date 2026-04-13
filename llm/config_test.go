package llm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestLoadLLMConfigFromRealFile(t *testing.T) {
	root := testLLMConfigRoot(t)
	cfg, err := loadTestLLMConfig(root)
	if err != nil {
		t.Fatalf("LoadLLMConfig() error = %v", err)
	}

	if cfg.Provider != "dashscope" {
		t.Fatalf("Provider = %q, want dashscope", cfg.Provider)
	}
	if cfg.APIBase == "" {
		t.Fatal("APIBase should not be empty")
	}
	if cfg.Default.Model != "qwen3-max" {
		t.Fatalf("Default.Model = %q, want qwen3-max", cfg.Default.Model)
	}
	if cfg.Default.Temperature != 0 {
		t.Fatalf("Default.Temperature = %v, want 0", cfg.Default.Temperature)
	}
	if cfg.Default.Search != false {
		t.Fatal("Default.Search should be false")
	}
	if cfg.Default.Thinking != false {
		t.Fatal("Default.Thinking should be false")
	}
	if len(cfg.Orchestration) == 0 {
		t.Fatal("Orchestration should not be empty")
	}
}

func TestResolveExistingRole(t *testing.T) {
	root := testLLMConfigRoot(t)
	cfg, err := loadTestLLMConfig(root)
	if err != nil {
		t.Fatalf("LoadLLMConfig() error = %v", err)
	}

	resolved := cfg.Resolve("category_classifier")

	if resolved.Model != "qwen3-max" {
		t.Fatalf("Model = %q, want qwen3-max", resolved.Model)
	}
	// category_classifier doesn't set search, so it should inherit default (false)
	if resolved.Search != false {
		t.Fatal("Search should be false (inherited from default)")
	}
	// category_classifier doesn't set mode, so it should be single
	if resolved.Mode != ModeSingle {
		t.Fatalf("Mode = %q, want single", resolved.Mode)
	}
}

func TestResolveNonexistentRoleFallsBackToDefaults(t *testing.T) {
	root := testLLMConfigRoot(t)
	cfg, err := loadTestLLMConfig(root)
	if err != nil {
		t.Fatalf("LoadLLMConfig() error = %v", err)
	}

	resolved := cfg.Resolve("nonexistent_role")

	if resolved.Model != cfg.Default.Model {
		t.Fatalf("Model = %q, want default %q", resolved.Model, cfg.Default.Model)
	}
	if resolved.Mode != ModeSingle {
		t.Fatalf("Mode = %q, want single", resolved.Mode)
	}
	if resolved.Temperature != cfg.Default.Temperature {
		t.Fatalf("Temperature = %v, want default %v", resolved.Temperature, cfg.Default.Temperature)
	}
	if resolved.Rounds != 1 {
		t.Fatalf("Rounds = %d, want 1", resolved.Rounds)
	}
}

func TestResolveChallengeRole(t *testing.T) {
	root := testLLMConfigRoot(t)
	cfg, err := loadTestLLMConfig(root)
	if err != nil {
		t.Fatalf("LoadLLMConfig() error = %v", err)
	}

	resolved := cfg.Resolve("protagonist_discoverer")

	if resolved.Mode != ModeChallenge {
		t.Fatalf("Mode = %q, want challenge", resolved.Mode)
	}
	if len(resolved.Models) != 2 {
		t.Fatalf("Models len = %d, want 2", len(resolved.Models))
	}
	if resolved.Search != true {
		t.Fatal("Search should be true for protagonist_discoverer")
	}
	if resolved.ChallengeInstructions == "" {
		t.Fatal("ChallengeInstructions should not be empty")
	}
}

func TestResolveDebateRole(t *testing.T) {
	root := testLLMConfigRoot(t)
	cfg, err := loadTestLLMConfig(root)
	if err != nil {
		t.Fatalf("LoadLLMConfig() error = %v", err)
	}

	resolved := cfg.Resolve("narrative_verdict")

	if resolved.Mode != ModeDebate {
		t.Fatalf("Mode = %q, want debate", resolved.Mode)
	}
	if resolved.Rounds != 2 {
		t.Fatalf("Rounds = %d, want 2", resolved.Rounds)
	}
	if len(resolved.Roles) != 3 {
		t.Fatalf("Roles len = %d, want 3 (bull, bear, judge)", len(resolved.Roles))
	}

	bull, ok := resolved.Roles["bull"]
	if !ok {
		t.Fatal("expected bull sub-role")
	}
	if bull.Model != "qwen3-max" {
		t.Fatalf("bull.Model = %q, want qwen3-max", bull.Model)
	}
	if bull.Search != true {
		t.Fatal("bull.Search should be true")
	}

	judge, ok := resolved.Roles["judge"]
	if !ok {
		t.Fatal("expected judge sub-role")
	}
	if judge.Thinking != true {
		t.Fatal("judge.Thinking should be true")
	}
	// judge doesn't set search, should inherit default (false)
	if judge.Search != false {
		t.Fatal("judge.Search should be false (inherited from default)")
	}
}

func TestResolveEnsembleRole(t *testing.T) {
	root := testLLMConfigRoot(t)
	cfg, err := loadTestLLMConfig(root)
	if err != nil {
		t.Fatalf("LoadLLMConfig() error = %v", err)
	}

	resolved := cfg.Resolve("claim_extractor")

	if resolved.Mode != ModeEnsemble {
		t.Fatalf("Mode = %q, want ensemble", resolved.Mode)
	}
	if resolved.Merge != MergeUnion {
		t.Fatalf("Merge = %q, want union", resolved.Merge)
	}
	if len(resolved.Models) != 2 {
		t.Fatalf("Models len = %d, want 2", len(resolved.Models))
	}
}

func TestResolveCascadeRole(t *testing.T) {
	root := testLLMConfigRoot(t)
	cfg, err := loadTestLLMConfig(root)
	if err != nil {
		t.Fatalf("LoadLLMConfig() error = %v", err)
	}

	resolved := cfg.Resolve("expand_judge")

	if resolved.Mode != ModeCascade {
		t.Fatalf("Mode = %q, want cascade", resolved.Mode)
	}
	if len(resolved.Models) != 2 {
		t.Fatalf("Models len = %d, want 2", len(resolved.Models))
	}
}

func TestResolvePointerFieldOverlay(t *testing.T) {
	// Test that nil pointer fields in RoleConfig mean "use default"
	// while non-nil fields override.
	cfg := LLMConfig{
		Default: DefaultConfig{
			Model:       "default-model",
			Search:      false,
			Temperature: 0,
			Thinking:    false,
		},
		Orchestration: map[string]RoleConfig{
			"search_only": {
				Search: boolPtr(true),
				// Model, Temperature, Thinking are nil -> inherit defaults
			},
		},
	}

	resolved := cfg.Resolve("search_only")

	if resolved.Model != "default-model" {
		t.Fatalf("Model = %q, want default-model (nil means inherit)", resolved.Model)
	}
	if resolved.Search != true {
		t.Fatal("Search should be true (explicitly set)")
	}
	if resolved.Temperature != 0 {
		t.Fatalf("Temperature = %v, want 0 (nil means inherit)", resolved.Temperature)
	}
	if resolved.Thinking != false {
		t.Fatal("Thinking should be false (nil means inherit)")
	}
}

func TestResolvePromptMapsReflexivityRoles(t *testing.T) {
	root := testLLMConfigRoot(t)
	cfg, err := loadTestLLMConfig(root)
	if err != nil {
		t.Fatalf("LoadLLMConfig() error = %v", err)
	}

	tests := []struct {
		promptID   string
		wantMode   Mode
		wantSearch bool
	}{
		// chain_profile → supply_chain_mapper (challenge mode with search)
		{"cognition/reflexivity/chain_profile", ModeChallenge, true},
		// protagonist_identification → protagonist_discoverer (challenge mode with search)
		{"cognition/reflexivity/protagonist_identification", ModeChallenge, true},
		// vulnerability_assessment → risk_assessment (debate mode)
		{"cognition/reflexivity/vulnerability_assessment", ModeDebate, false},
	}

	for _, tt := range tests {
		rc := cfg.ResolvePrompt(tt.promptID)
		if rc.Mode != tt.wantMode {
			t.Errorf("ResolvePrompt(%q).Mode = %q, want %q", tt.promptID, rc.Mode, tt.wantMode)
		}
		if rc.Search != tt.wantSearch {
			t.Errorf("ResolvePrompt(%q).Search = %v, want %v", tt.promptID, rc.Search, tt.wantSearch)
		}
	}
}

func TestResolvePromptFallsBackToSuffix(t *testing.T) {
	root := testLLMConfigRoot(t)
	cfg, err := loadTestLLMConfig(root)
	if err != nil {
		t.Fatalf("LoadLLMConfig() error = %v", err)
	}

	// category_classifier is a direct match (no prompt_roles mapping needed)
	rc := cfg.ResolvePrompt("cognition/fundamental/category_classifier")
	if rc.Mode != ModeSingle {
		t.Errorf("Mode = %q, want single", rc.Mode)
	}
	if rc.Model != "qwen3-max" {
		t.Errorf("Model = %q, want qwen3-max", rc.Model)
	}
}

func TestLoadLLMConfigRejectsChallengeWithoutTwoModels(t *testing.T) {
	root := t.TempDir()
	writeEngineParams(t, root, `
llm:
  provider: dashscope
  api_base: https://example.com
  default:
    model: qwen3-max
    search: false
    temperature: 0
    thinking: false
  orchestration:
    bad_challenge:
      mode: challenge
      model: qwen3-max
`)

	_, err := loadTestLLMConfig(root)
	if err == nil {
		t.Fatal("expected error for single-model challenge config")
	}
	if !strings.Contains(err.Error(), "challenge") {
		t.Fatalf("error = %v, want challenge validation error", err)
	}
}

func TestLoadLLMConfigRejectsEnsembleWithoutModels(t *testing.T) {
	root := t.TempDir()
	writeEngineParams(t, root, `
llm:
  provider: dashscope
  api_base: https://example.com
  default:
    model: qwen3-max
    search: false
    temperature: 0
    thinking: false
  orchestration:
    bad_ensemble:
      mode: ensemble
`)

	_, err := loadTestLLMConfig(root)
	if err == nil {
		t.Fatal("expected error for ensemble config without models")
	}
	if !strings.Contains(err.Error(), "ensemble") {
		t.Fatalf("error = %v, want ensemble validation error", err)
	}
}

func writeEngineParams(t *testing.T, root, content string) {
	t.Helper()

	path := filepath.Join(root, "config", "engine_params.yaml")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(strings.TrimSpace(content)+"\n"), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func boolPtr(b bool) *bool { return &b }

func testLLMConfigRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	writeEngineParams(t, root, `
llm:
  provider: dashscope
  api_base: https://example.com
  default:
    model: qwen3-max
    search: false
    temperature: 0
    thinking: false
  prompt_roles:
    chain_profile: supply_chain_mapper
    protagonist_identification: protagonist_discoverer
    vulnerability_assessment: risk_assessment
  orchestration:
    category_classifier:
      model: qwen3-max
    supply_chain_mapper:
      mode: challenge
      models: ["qwen3-max", "qwen3.6-plus"]
      search: true
      challenge_instructions: map the chain
    protagonist_discoverer:
      mode: challenge
      models: ["qwen3-max", "qwen3.6-plus"]
      search: true
      challenge_instructions: challenge it
    narrative_verdict:
      mode: debate
      rounds: 2
      roles:
        bull: { model: "qwen3-max", search: true }
        bear: { model: "qwen3.6-plus", search: true }
        judge: { model: "qwen3.6-plus", thinking: true }
    claim_extractor:
      mode: ensemble
      models: ["qwen3-max", "qwen3.6-plus"]
      merge: union
    expand_judge:
      mode: cascade
      models: ["qwen3-max", "qwen3.6-plus"]
    risk_assessment:
      mode: debate
      roles:
        bull: { model: "qwen3-max" }
        bear: { model: "qwen3.6-plus" }
        judge: { model: "qwen3.6-plus" }
`)
	return root
}

type testEngineParams struct {
	LLM LLMConfig `yaml:"llm"`
}

func loadTestLLMConfig(projectRoot string) (LLMConfig, error) {
	data, err := os.ReadFile(filepath.Join(projectRoot, "config", "engine_params.yaml"))
	if err != nil {
		return LLMConfig{}, err
	}
	var params testEngineParams
	if err := yaml.Unmarshal(data, &params); err != nil {
		return LLMConfig{}, err
	}
	if err := params.LLM.Validate(); err != nil {
		return LLMConfig{}, err
	}
	return params.LLM, nil
}
