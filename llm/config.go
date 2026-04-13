package llm

import (
	"fmt"
	"strings"
)

// SubRoleConfig defines LLM settings for a sub-role within a debate.
// Pointer fields: nil means "inherit from defaults".
type SubRoleConfig struct {
	Model       *string  `yaml:"model"`
	Search      *bool    `yaml:"search"`
	Temperature *float64 `yaml:"temperature"`
	Thinking    *bool    `yaml:"thinking"`
}

// RoleConfig defines LLM settings for a specific orchestration role.
// Pointer fields: nil means "inherit from defaults".
type RoleConfig struct {
	Model                 *string                  `yaml:"model"`
	Mode                  *Mode                    `yaml:"mode"`
	Models                []string                 `yaml:"models"`
	Merge                 *MergeStrategy           `yaml:"merge"`
	Search                *bool                    `yaml:"search"`
	Temperature           *float64                 `yaml:"temperature"`
	Thinking              *bool                    `yaml:"thinking"`
	Rounds                *int                     `yaml:"rounds"`
	ChallengeInstructions *string                  `yaml:"challenge_instructions"`
	Roles                 map[string]SubRoleConfig `yaml:"roles"`
}

// DefaultConfig holds the baseline LLM settings that all roles inherit from.
type DefaultConfig struct {
	Model       string  `yaml:"model"`
	Search      bool    `yaml:"search"`
	Temperature float64 `yaml:"temperature"`
	Thinking    bool    `yaml:"thinking"`
}

// LLMConfig is the top-level LLM configuration parsed from engine_params.yaml.
type LLMConfig struct {
	Provider      string                `yaml:"provider"`
	APIBase       string                `yaml:"api_base"`
	Default       DefaultConfig         `yaml:"default"`
	Orchestration map[string]RoleConfig `yaml:"orchestration"`
	// PromptRoles maps prompt ID suffixes to orchestration keys when they differ.
	// Example: "protagonist_identification" → "protagonist_discoverer"
	PromptRoles map[string]string `yaml:"prompt_roles"`
}

// ResolvedConfig is the fully-resolved configuration for a role, with all
// defaults applied. No pointer fields -- every value is concrete.
type ResolvedConfig struct {
	Model                 string
	Mode                  Mode
	Models                []string
	Merge                 MergeStrategy
	Search                bool
	Temperature           float64
	Thinking              bool
	Rounds                int
	ChallengeInstructions string
	Roles                 map[string]ResolvedConfig
}

// ResolvePromptRole maps a prompt ID to its orchestration role key.
// It first checks PromptRoles for an explicit mapping, then falls back to
// using the last path segment of the prompt ID as the role key.
func (c LLMConfig) ResolvePromptRole(promptID string) string {
	parts := strings.Split(promptID, "/")
	suffix := parts[len(parts)-1]

	// Check explicit prompt→role mapping first.
	if role, ok := c.PromptRoles[suffix]; ok {
		return role
	}
	return suffix
}

// ResolvePrompt maps a prompt ID to its orchestration role and resolves config.
func (c LLMConfig) ResolvePrompt(promptID string) ResolvedConfig {
	return c.Resolve(c.ResolvePromptRole(promptID))
}

// Resolve returns the effective config for a role, overlaying role-specific
// settings on top of defaults. If the role is not found in Orchestration,
// the result is pure defaults.
func (c LLMConfig) Resolve(role string) ResolvedConfig {
	base := ResolvedConfig{
		Model:       c.Default.Model,
		Mode:        ModeSingle,
		Search:      c.Default.Search,
		Temperature: c.Default.Temperature,
		Thinking:    c.Default.Thinking,
		Rounds:      1,
	}

	rc, ok := c.Orchestration[role]
	if !ok {
		return base
	}

	return c.applyRole(rc, base)
}

// applyRole overlays a RoleConfig onto a base ResolvedConfig.
func (c LLMConfig) applyRole(rc RoleConfig, base ResolvedConfig) ResolvedConfig {
	if rc.Model != nil {
		base.Model = *rc.Model
	}
	if rc.Mode != nil {
		base.Mode = *rc.Mode
	}
	if rc.Models != nil {
		base.Models = rc.Models
	}
	if rc.Merge != nil {
		base.Merge = *rc.Merge
	}
	if rc.Search != nil {
		base.Search = *rc.Search
	}
	if rc.Temperature != nil {
		base.Temperature = *rc.Temperature
	}
	if rc.Thinking != nil {
		base.Thinking = *rc.Thinking
	}
	if rc.Rounds != nil {
		base.Rounds = *rc.Rounds
	}
	if rc.ChallengeInstructions != nil {
		base.ChallengeInstructions = *rc.ChallengeInstructions
	}

	if len(rc.Roles) > 0 {
		base.Roles = make(map[string]ResolvedConfig, len(rc.Roles))
		for name, sub := range rc.Roles {
			resolved := ResolvedConfig{
				Model:       c.Default.Model,
				Mode:        ModeSingle,
				Search:      c.Default.Search,
				Temperature: c.Default.Temperature,
				Thinking:    c.Default.Thinking,
				Rounds:      1,
			}
			if sub.Model != nil {
				resolved.Model = *sub.Model
			}
			if sub.Search != nil {
				resolved.Search = *sub.Search
			}
			if sub.Temperature != nil {
				resolved.Temperature = *sub.Temperature
			}
			if sub.Thinking != nil {
				resolved.Thinking = *sub.Thinking
			}
			base.Roles[name] = resolved
		}
	}

	return base
}

// Validate checks mode-specific configuration invariants that should fail fast
// during config load rather than silently degrading at runtime.
func (c LLMConfig) Validate() error {
	if strings.TrimSpace(c.Default.Model) == "" {
		return fmt.Errorf("llm config: default.model must not be empty")
	}

	for role, rc := range c.Orchestration {
		if rc.Mode == nil {
			continue
		}

		switch *rc.Mode {
		case ModeEnsemble:
			if len(rc.Models) == 0 {
				return fmt.Errorf("llm config: role %q with mode=ensemble requires models", role)
			}
		case ModeChallenge:
			if len(rc.Models) != 2 {
				return fmt.Errorf("llm config: role %q with mode=challenge requires exactly 2 models", role)
			}
		}
	}

	return nil
}
