package llm

import "time"

// Mode describes how an orchestration step invokes LLMs.
type Mode string

const (
	ModeSingle    Mode = "single"
	ModeEnsemble  Mode = "ensemble"
	ModeCascade   Mode = "cascade"
	ModeChallenge Mode = "challenge"
	ModeDebate    Mode = "debate"
)

// MergeStrategy determines how ensemble results are combined.
type MergeStrategy string

const (
	MergeUnion MergeStrategy = "union"
	MergeVote  MergeStrategy = "vote"
)

// TokenUsage tracks token consumption for a single LLM call.
type TokenUsage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// AttemptTrace records one LLM call attempt, including retries.
type AttemptTrace struct {
	Model     string
	Latency   time.Duration
	Tokens    TokenUsage
	Error     string // empty if successful
	Truncated bool   // true if response was truncated
}

// ModeStep records one step in multi-step execution (ensemble, debate, etc.).
type ModeStep struct {
	Mode    Mode
	Role    string // e.g. "bull", "bear", "judge", "generator", "challenger"
	Model   string
	Latency time.Duration
	Tokens  TokenUsage
}

// RefusalInfo captures a model refusal (safety filter, policy, etc.).
type RefusalInfo struct {
	Reason string
}

// Response is the audit envelope for every LLM interaction.
// It tracks the final result plus full provenance: every attempt,
// every step in multi-model orchestration, and any refusal.
type Response struct {
	PromptID  string // prompt asset ID that produced this response
	Role      string // orchestration role used (maps to config key)
	Text      string
	Model     string
	Attempts  []AttemptTrace
	Latency   time.Duration
	Tokens    TokenUsage
	ModeTrace []ModeStep
	Refusal   *RefusalInfo
}

// TypedResponse wraps Response with a parsed structured value.
type TypedResponse[T any] struct {
	Response
	Value T
}
