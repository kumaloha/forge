package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return fn(req)
}

func newHTTPResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func newDashscopeTestProvider(t *testing.T, transport roundTripFunc, opts ...DashscopeOption) *Dashscope {
	t.Helper()

	allOpts := append([]DashscopeOption{
		WithAPIBase("https://dashscope.test"),
		WithAPIKey("test-key"),
	}, opts...)

	d, err := NewDashscope(allOpts...)
	if err != nil {
		t.Fatalf("NewDashscope() error = %v", err)
	}
	d.client.Transport = transport
	return d
}

func TestDashscopeRequestBody(t *testing.T) {
	var captured map[string]any

	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)

		return newHTTPResponse(http.StatusOK, `{
			"model": "qwen3-max",
			"choices": [
				{"message": {"content": "hello"}, "finish_reason": "stop"}
			],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`), nil
	})

	_, err := d.Call(context.Background(), ProviderRequest{
		Model:       "qwen3-max",
		System:      "You are a helper.",
		User:        "Hello",
		Temperature: 0,
		Search:      true,
		Thinking:    true,
	})
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}

	// Verify request body structure.
	if captured["model"] != "qwen3-max" {
		t.Fatalf("model = %v, want qwen3-max", captured["model"])
	}
	if captured["stream"] != false {
		t.Fatalf("stream = %v, want false", captured["stream"])
	}

	msgs, ok := captured["messages"].([]any)
	if !ok || len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %v", captured["messages"])
	}

	sysMsg := msgs[0].(map[string]any)
	if sysMsg["role"] != "system" || sysMsg["content"] != "You are a helper." {
		t.Fatalf("system message = %v", sysMsg)
	}

	userMsg := msgs[1].(map[string]any)
	if userMsg["role"] != "user" || userMsg["content"] != "Hello" {
		t.Fatalf("user message = %v", userMsg)
	}

	if captured["enable_search"] != true {
		t.Fatalf("enable_search = %v, want true", captured["enable_search"])
	}
	if captured["enable_thinking"] != true {
		t.Fatalf("enable_thinking = %v, want true", captured["enable_thinking"])
	}
}

func TestDashscopeRequestBodyOmitsDisabledFlags(t *testing.T) {
	var captured map[string]any

	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)

		return newHTTPResponse(http.StatusOK, `{
			"model": "qwen3-max",
			"choices": [
				{"message": {"content": "ok"}, "finish_reason": "stop"}
			],
			"usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
		}`), nil
	})

	d.Call(context.Background(), ProviderRequest{
		Model:  "qwen3-max",
		System: "sys",
		User:   "usr",
	})

	// search and thinking should be omitted when false.
	if _, exists := captured["enable_search"]; exists {
		t.Fatalf("enable_search should be omitted when false, got %v", captured["enable_search"])
	}
	if _, exists := captured["enable_thinking"]; exists {
		t.Fatalf("enable_thinking should be omitted when false, got %v", captured["enable_thinking"])
	}
}

func TestDashscopeRequestWithJSONSchema(t *testing.T) {
	var captured map[string]any

	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)

		return newHTTPResponse(http.StatusOK, `{
			"model": "qwen3-max",
			"choices": [
				{"message": {"content": "{\"score\": 5}"}, "finish_reason": "stop"}
			],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`), nil
	})

	d.Call(context.Background(), ProviderRequest{
		Model:  "qwen3-max",
		System: "sys",
		User:   "usr",
		JSONSchema: &Schema{
			Name:       "ScoreResult",
			Properties: map[string]any{"score": map[string]any{"type": "integer"}},
			Required:   []string{"score"},
		},
	})

	rf, ok := captured["response_format"].(map[string]any)
	if !ok {
		t.Fatalf("expected response_format in request, got %v", captured["response_format"])
	}
	if rf["type"] != "json_schema" {
		t.Fatalf("response_format.type = %v, want json_schema", rf["type"])
	}
	js, ok := rf["json_schema"].(map[string]any)
	if !ok {
		t.Fatalf("expected json_schema in response_format")
	}
	if js["name"] != "ScoreResult" {
		t.Fatalf("json_schema.name = %v, want ScoreResult", js["name"])
	}
}

func TestDashscopeRequestWithJSONSchemaAddsJSONHint(t *testing.T) {
	var captured map[string]any

	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		body, _ := io.ReadAll(r.Body)
		json.Unmarshal(body, &captured)

		return newHTTPResponse(http.StatusOK, `{
			"model": "qwen3-max",
			"choices": [
				{"message": {"content": "{\"score\": 5}"}, "finish_reason": "stop"}
			],
			"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
		}`), nil
	})

	_, err := d.Call(context.Background(), ProviderRequest{
		Model:  "qwen3-max",
		System: "Judge the evidence carefully.",
		User:   "Return the score.",
		JSONSchema: &Schema{
			Name:       "ScoreResult",
			Properties: map[string]any{"score": map[string]any{"type": "integer"}},
			Required:   []string{"score"},
		},
	})
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}

	msgs, ok := captured["messages"].([]any)
	if !ok || len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %v", captured["messages"])
	}

	foundJSON := false
	for _, raw := range msgs {
		msg, _ := raw.(map[string]any)
		content, _ := msg["content"].(string)
		if strings.Contains(strings.ToLower(content), "json") {
			foundJSON = true
		}
	}
	if !foundJSON {
		t.Fatalf("expected at least one message to contain a json hint, got %v", captured["messages"])
	}
}

func TestDashscopeResponseParsing(t *testing.T) {
	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusOK, `{
			"model": "qwen3-max",
			"choices": [
				{"message": {"content": "response text"}, "finish_reason": "stop"}
			],
			"usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
		}`), nil
	})
	resp, err := d.Call(context.Background(), ProviderRequest{Model: "qwen3-max", System: "s", User: "u"})
	if err != nil {
		t.Fatalf("Call() error = %v", err)
	}

	if resp.Text != "response text" {
		t.Fatalf("Text = %q, want %q", resp.Text, "response text")
	}
	if resp.Model != "qwen3-max" {
		t.Fatalf("Model = %q, want qwen3-max", resp.Model)
	}
	if resp.Tokens.PromptTokens != 10 || resp.Tokens.CompletionTokens != 20 || resp.Tokens.TotalTokens != 30 {
		t.Fatalf("Tokens = %+v, want 10/20/30", resp.Tokens)
	}
	if resp.Truncated {
		t.Fatal("expected Truncated = false")
	}
	if resp.Refusal != nil {
		t.Fatalf("expected no refusal, got %+v", resp.Refusal)
	}
}

func TestDashscopeRefusal(t *testing.T) {
	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusOK, `{
			"model": "qwen3-max",
			"choices": [
				{"message": {"content": "", "refusal": "content policy violation"}, "finish_reason": "stop"}
			],
			"usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
		}`), nil
	})
	resp, _ := d.Call(context.Background(), ProviderRequest{Model: "m", System: "s", User: "u"})

	if resp.Refusal == nil {
		t.Fatal("expected refusal info")
	}
	if resp.Refusal.Reason != "content policy violation" {
		t.Fatalf("Refusal.Reason = %q, want %q", resp.Refusal.Reason, "content policy violation")
	}
}

func TestDashscopeTruncation(t *testing.T) {
	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusOK, `{
			"model": "qwen3-max",
			"choices": [
				{"message": {"content": "partial..."}, "finish_reason": "length"}
			],
			"usage": {"prompt_tokens": 10, "completion_tokens": 100, "total_tokens": 110}
		}`), nil
	})
	resp, _ := d.Call(context.Background(), ProviderRequest{Model: "m", System: "s", User: "u"})

	if !resp.Truncated {
		t.Fatal("expected Truncated = true for finish_reason=length")
	}
}

func TestDashscopeRetryOn5xx(t *testing.T) {
	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusInternalServerError, `{"error": "internal"}`), nil
	})

	// The Dashscope provider itself doesn't retry; retry is in Runtime.chatSingle.
	// Here we test that 5xx produces a retryable APIError.
	_, err := d.Call(context.Background(), ProviderRequest{Model: "m", System: "s", User: "u"})
	if err == nil {
		t.Fatal("expected error on 500")
	}

	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if !apiErr.Retryable() {
		t.Fatal("500 should be retryable")
	}
}

func TestDashscopeNoRetryOn4xx(t *testing.T) {
	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusBadRequest, `{"error": "bad request"}`), nil
	})
	_, err := d.Call(context.Background(), ProviderRequest{Model: "m", System: "s", User: "u"})
	if err == nil {
		t.Fatal("expected error on 400")
	}

	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if apiErr.Retryable() {
		t.Fatal("400 should not be retryable")
	}
}

func TestDashscope429IsRetryable(t *testing.T) {
	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		return newHTTPResponse(http.StatusTooManyRequests, `{"error": "rate limited"}`), nil
	})
	_, err := d.Call(context.Background(), ProviderRequest{Model: "m", System: "s", User: "u"})

	apiErr, ok := err.(*APIError)
	if !ok {
		t.Fatalf("expected *APIError, got %T", err)
	}
	if !apiErr.Retryable() {
		t.Fatal("429 should be retryable")
	}
}

func TestDashscopeAuthHeader(t *testing.T) {
	var gotAuth string

	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		gotAuth = r.Header.Get("Authorization")
		return newHTTPResponse(http.StatusOK, `{
			"model": "m",
			"choices": [
				{"message": {"content": "ok"}, "finish_reason": "stop"}
			],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
		}`), nil
	}, WithAPIKey("my-secret-key"))
	d.Call(context.Background(), ProviderRequest{Model: "m", System: "s", User: "u"})

	if gotAuth != "Bearer my-secret-key" {
		t.Fatalf("Authorization = %q, want %q", gotAuth, "Bearer my-secret-key")
	}
}

func TestNewDashscopeRequiresAPIKey(t *testing.T) {
	// Unset the env var temporarily for this test.
	t.Setenv("DASHSCOPE_API_KEY", "")
	_, err := NewDashscope()
	if err == nil {
		t.Fatal("expected error when no API key provided")
	}
}

func TestNewDashscopeUsesEnvTimeoutOverride(t *testing.T) {
	t.Setenv("DASHSCOPE_API_KEY", "test-key")
	t.Setenv("DASHSCOPE_TIMEOUT_SECONDS", "45")

	d, err := NewDashscope()
	if err != nil {
		t.Fatalf("NewDashscope() error = %v", err)
	}
	if got := d.client.Timeout; got != 45*time.Second {
		t.Fatalf("client.Timeout = %s, want 45s", got)
	}
}

func TestDashscopeTimeout(t *testing.T) {
	d := newDashscopeTestProvider(t, func(r *http.Request) (*http.Response, error) {
		select {
		case <-time.After(200 * time.Millisecond):
			return newHTTPResponse(http.StatusOK, `{
				"model": "m",
				"choices": [
					{"message": {"content": "ok"}, "finish_reason": "stop"}
				],
				"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
			}`), nil
		case <-r.Context().Done():
			return nil, r.Context().Err()
		}
	}, WithAPIKey("key"), WithTimeout(50*time.Millisecond))

	_, err := d.Call(context.Background(), ProviderRequest{Model: "m", System: "s", User: "u"})
	if err == nil {
		t.Fatal("expected timeout error")
	}
}
