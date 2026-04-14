package llm

import "context"

// Schema represents a JSON Schema for structured output.
type Schema struct {
	Name       string
	Properties map[string]any
	Required   []string
}

type ContentPart struct {
	Type     string
	Text     string
	ImageURL string
}

// ProviderRequest is what we send to an LLM provider.
type ProviderRequest struct {
	Model       string
	System      string
	User        string
	UserParts   []ContentPart
	Temperature float64
	Search      bool    // Qwen-specific: enable_search
	Thinking    bool    // Qwen-specific: enable_thinking
	JSONSchema  *Schema // nil = free-form text response
}

// ProviderResponse is what we get back from the provider.
type ProviderResponse struct {
	Text      string
	Model     string
	Tokens    TokenUsage
	Refusal   *RefusalInfo
	Truncated bool
}

// Provider is the interface for LLM API calls.
type Provider interface {
	Name() string
	Call(ctx context.Context, req ProviderRequest) (ProviderResponse, error)
}
