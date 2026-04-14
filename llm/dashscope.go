package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// Dashscope implements Provider for Alibaba's Dashscope API (OpenAI-compatible).
type Dashscope struct {
	apiBase string
	apiKey  string
	client  *http.Client
}

// DashscopeOption configures a Dashscope provider.
type DashscopeOption func(*Dashscope)

// WithAPIBase overrides the default Dashscope API base URL.
func WithAPIBase(base string) DashscopeOption {
	return func(d *Dashscope) { d.apiBase = base }
}

// WithAPIKey overrides the API key (default: DASHSCOPE_API_KEY env var).
func WithAPIKey(key string) DashscopeOption {
	return func(d *Dashscope) { d.apiKey = key }
}

// WithTimeout sets the HTTP client timeout (default: 120s).
func WithTimeout(t time.Duration) DashscopeOption {
	return func(d *Dashscope) { d.client.Timeout = t }
}

// NewDashscope creates a Dashscope provider. If no API key is provided via
// options, it reads DASHSCOPE_API_KEY from the environment.
func NewDashscope(opts ...DashscopeOption) (*Dashscope, error) {
	d := &Dashscope{
		apiBase: "https://dashscope.aliyuncs.com/compatible-mode/v1",
		client:  &http.Client{Timeout: 120 * time.Second},
	}
	if raw := os.Getenv("DASHSCOPE_TIMEOUT_SECONDS"); raw != "" {
		seconds, err := strconv.Atoi(strings.TrimSpace(raw))
		if err != nil || seconds <= 0 {
			return nil, fmt.Errorf("llm.Dashscope: invalid DASHSCOPE_TIMEOUT_SECONDS %q", raw)
		}
		d.client.Timeout = time.Duration(seconds) * time.Second
	}
	for _, opt := range opts {
		opt(d)
	}
	if d.apiKey == "" {
		d.apiKey = os.Getenv("DASHSCOPE_API_KEY")
	}
	if d.apiKey == "" {
		return nil, fmt.Errorf("llm.Dashscope: DASHSCOPE_API_KEY not set")
	}
	return d, nil
}

func (d *Dashscope) Name() string { return "dashscope" }

// Call sends a chat completion request to the Dashscope API.
func (d *Dashscope) Call(ctx context.Context, req ProviderRequest) (ProviderResponse, error) {
	body, err := d.buildRequestBody(req)
	if err != nil {
		return ProviderResponse{}, fmt.Errorf("llm.Dashscope: build request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, d.apiBase+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return ProviderResponse{}, fmt.Errorf("llm.Dashscope: new request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+d.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := d.client.Do(httpReq)
	if err != nil {
		return ProviderResponse{}, fmt.Errorf("llm.Dashscope: do request: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return ProviderResponse{}, fmt.Errorf("llm.Dashscope: read response: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		return ProviderResponse{}, &APIError{
			StatusCode: httpResp.StatusCode,
			Body:       string(respBody),
		}
	}

	return d.parseResponse(respBody)
}

// APIError represents a non-200 response from the API.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("llm.Dashscope: HTTP %d: %s", e.StatusCode, e.Body)
}

// Retryable returns true if the error is worth retrying (5xx, 429).
func (e *APIError) Retryable() bool {
	return e.StatusCode == http.StatusTooManyRequests || e.StatusCode >= 500
}

// buildRequestBody constructs the JSON request body for Dashscope.
func (d *Dashscope) buildRequestBody(req ProviderRequest) ([]byte, error) {
	systemContent := req.System
	userContent := req.User
	userParts := append([]ContentPart(nil), req.UserParts...)
	if req.JSONSchema != nil && !containsJSONHint(systemContent, userContent, partsText(userParts)) {
		if len(userParts) > 0 {
			userParts = append(userParts, ContentPart{Type: "text", Text: "Return valid JSON only."})
		} else {
			userContent = strings.TrimSpace(userContent + "\n\nReturn valid JSON only.")
		}
	}

	messages := make([]chatMessage, 0, 2)
	if strings.TrimSpace(systemContent) != "" {
		messages = append(messages, chatMessage{Role: "system", Content: systemContent})
	}
	if len(userParts) > 0 {
		messages = append(messages, chatMessage{Role: "user", Content: contentPartsToWire(userParts)})
	} else {
		messages = append(messages, chatMessage{Role: "user", Content: userContent})
	}

	body := chatRequest{
		Model:    req.Model,
		Messages: messages,
		Stream:   false,
	}

	// Temperature: Dashscope requires > 0 for some models; we pass it as-is
	// and let the API handle validation.
	body.Temperature = &req.Temperature

	// Qwen-specific flags go at top level for Dashscope.
	if req.Search {
		body.EnableSearch = &req.Search
	}
	if req.Thinking {
		body.EnableThinking = &req.Thinking
	}

	// Structured output via JSON schema.
	if req.JSONSchema != nil {
		schemaObj := map[string]any{
			"type":       "object",
			"properties": req.JSONSchema.Properties,
		}
		if len(req.JSONSchema.Required) > 0 {
			schemaObj["required"] = req.JSONSchema.Required
		}
		body.ResponseFormat = &responseFormat{
			Type: "json_schema",
			JSONSchema: &jsonSchemaSpec{
				Name:   req.JSONSchema.Name,
				Schema: schemaObj,
			},
		}
	}

	return json.Marshal(body)
}

func containsJSONHint(parts ...string) bool {
	for _, part := range parts {
		if strings.Contains(strings.ToLower(part), "json") {
			return true
		}
	}
	return false
}

func partsText(parts []ContentPart) string {
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if strings.TrimSpace(part.Text) != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

func contentPartsToWire(parts []ContentPart) []chatMessagePart {
	out := make([]chatMessagePart, 0, len(parts))
	for _, part := range parts {
		switch strings.TrimSpace(part.Type) {
		case "image_url":
			out = append(out, chatMessagePart{
				Type:     "image_url",
				ImageURL: &chatImageURL{URL: part.ImageURL},
			})
		default:
			out = append(out, chatMessagePart{
				Type: "text",
				Text: part.Text,
			})
		}
	}
	return out
}

// parseResponse extracts a ProviderResponse from the raw JSON.
func (d *Dashscope) parseResponse(data []byte) (ProviderResponse, error) {
	var resp chatResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return ProviderResponse{}, fmt.Errorf("llm.Dashscope: parse response: %w", err)
	}

	if len(resp.Choices) == 0 {
		return ProviderResponse{}, fmt.Errorf("llm.Dashscope: no choices in response")
	}

	choice := resp.Choices[0]
	pr := ProviderResponse{
		Text:  choice.Message.Content,
		Model: resp.Model,
		Tokens: TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
		Truncated: choice.FinishReason == "length",
	}

	if choice.Message.Refusal != "" {
		pr.Refusal = &RefusalInfo{Reason: choice.Message.Refusal}
	}

	return pr, nil
}

// --- JSON wire types ---

type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type chatMessagePart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text,omitempty"`
	ImageURL *chatImageURL `json:"image_url,omitempty"`
}

type chatImageURL struct {
	URL string `json:"url"`
}

type responseFormat struct {
	Type       string          `json:"type"`
	JSONSchema *jsonSchemaSpec `json:"json_schema,omitempty"`
}

type jsonSchemaSpec struct {
	Name   string         `json:"name"`
	Schema map[string]any `json:"schema"`
}

type chatRequest struct {
	Model          string          `json:"model"`
	Messages       []chatMessage   `json:"messages"`
	Temperature    *float64        `json:"temperature,omitempty"`
	Stream         bool            `json:"stream"`
	EnableSearch   *bool           `json:"enable_search,omitempty"`
	EnableThinking *bool           `json:"enable_thinking,omitempty"`
	ResponseFormat *responseFormat `json:"response_format,omitempty"`
}

type chatResponse struct {
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
}

type chatChoice struct {
	Message      chatRespMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type chatRespMessage struct {
	Content string `json:"content"`
	Refusal string `json:"refusal"`
}

type chatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
