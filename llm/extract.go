package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// MergeFunc combines multiple typed responses into one value.
type MergeFunc[T any] func(items []TypedResponse[T]) (T, error)

// ExtractOptions configures structured extraction.
type ExtractOptions[T any] struct {
	Merge MergeFunc[T] // nil = single mode, no merge needed
}

// Extract calls the LLM with a JSON schema constraint, parses the response
// into T, and validates it. T must implement Validate() error.
//
// This is a package-level generic function because Go does not support
// parameterized methods on structs.
func Extract[T interface{ Validate() error }](
	ctx context.Context,
	rt *Runtime,
	spec PromptSpec,
	vars map[string]any,
	opts ExtractOptions[T],
) (TypedResponse[T], error) {
	schema := SchemaFrom[T]()

	const maxValidationRetries = 2

	// We may need to modify the prompt for validation retries, so build
	// the prompt once and track the original user text.
	built := spec.Build(vars, nil)
	roleKey := rt.config.ResolvePromptRole(spec.ID)
	rc := rt.config.Resolve(roleKey)

	for attempt := range maxValidationRetries + 1 {
		typed, err := extractOnce(ctx, rt, roleKey, spec.ID, rc, built, &schema, opts)
		if err != nil {
			var retryErr *retryableExtractError
			if errors.As(err, &retryErr) {
				if attempt < maxValidationRetries {
					built = appendRetryHint(built, retryErr.hint)
					continue
				}
				return typed, fmt.Errorf("llm.Extract: %s after %d attempts: %w", retryErr.kind, attempt+1, retryErr.err)
			}
			return typed, err
		}

		return typed, nil
	}

	// Should not reach here, but just in case.
	return TypedResponse[T]{}, fmt.Errorf("llm.Extract: exhausted retries")
}

type retryableExtractError struct {
	kind string
	hint string
	err  error
}

func (e *retryableExtractError) Error() string { return e.err.Error() }

func (e *retryableExtractError) Unwrap() error { return e.err }

func extractOnce[T interface{ Validate() error }](
	ctx context.Context,
	rt *Runtime,
	roleKey string,
	promptID string,
	rc ResolvedConfig,
	built BuiltPrompt,
	schema *Schema,
	opts ExtractOptions[T],
) (TypedResponse[T], error) {
	switch rc.Mode {
	case ModeEnsemble:
		return extractEnsemble(ctx, rt, roleKey, promptID, rc, built, schema, opts)
	case ModeChallenge:
		return extractChallenge[T](ctx, rt, roleKey, promptID, rc, built, schema)
	case ModeCascade:
		resp, err := rt.executeCascade(ctx, rc, built, schema)
		stampResponseMetadata(&resp, promptID, roleKey)
		if err != nil {
			return TypedResponse[T]{Response: resp}, err
		}
		return parseTypedResponse[T](resp)
	case ModeDebate:
		resp, err := rt.executeDebate(ctx, rc, built, schema)
		stampResponseMetadata(&resp, promptID, roleKey)
		if err != nil {
			return TypedResponse[T]{Response: resp}, err
		}
		return parseTypedResponse[T](resp)
	default:
		resp, err := rt.executeSingle(ctx, rc, built, schema)
		stampResponseMetadata(&resp, promptID, roleKey)
		if err != nil {
			return TypedResponse[T]{Response: resp}, err
		}
		return parseTypedResponse[T](resp)
	}
}

func extractEnsemble[T interface{ Validate() error }](
	ctx context.Context,
	rt *Runtime,
	roleKey string,
	promptID string,
	rc ResolvedConfig,
	built BuiltPrompt,
	schema *Schema,
	opts ExtractOptions[T],
) (TypedResponse[T], error) {
	responses, aggregate, err := rt.collectEnsembleResponses(ctx, rc, built, schema)
	stampResponseMetadata(&aggregate, promptID, roleKey)
	if err != nil {
		return TypedResponse[T]{Response: aggregate}, err
	}

	items := make([]TypedResponse[T], 0, len(responses))
	for _, resp := range responses {
		stampResponseMetadata(&resp, promptID, roleKey)
		typed, err := parseTypedResponse[T](resp)
		if err != nil {
			typed.Response = aggregate
			return typed, err
		}
		items = append(items, typed)
	}

	switch {
	case len(items) == 0:
		return TypedResponse[T]{Response: aggregate}, fmt.Errorf("llm.Extract: ensemble returned no successful typed responses")
	case len(items) == 1 && opts.Merge == nil:
		return TypedResponse[T]{Response: aggregate, Value: items[0].Value}, nil
	case opts.Merge == nil:
		return TypedResponse[T]{Response: aggregate}, fmt.Errorf("llm.Extract: ensemble mode requires MergeFunc for %d responses", len(items))
	}

	merged, err := opts.Merge(items)
	if err != nil {
		return TypedResponse[T]{Response: aggregate}, fmt.Errorf("llm.Extract: merge failed: %w", err)
	}
	return TypedResponse[T]{Response: aggregate, Value: merged}, nil
}

func extractChallenge[T interface{ Validate() error }](
	ctx context.Context,
	rt *Runtime,
	roleKey string,
	promptID string,
	rc ResolvedConfig,
	built BuiltPrompt,
	schema *Schema,
) (TypedResponse[T], error) {
	genResp, challenger, aggregate, err := rt.runChallengePair(ctx, rc, built, schema)
	stampResponseMetadata(&genResp, promptID, roleKey)
	stampResponseMetadata(&challenger, promptID, roleKey)
	stampResponseMetadata(&aggregate, promptID, roleKey)
	if err != nil {
		return TypedResponse[T]{Response: aggregate}, err
	}

	finalResp := aggregate
	if strings.TrimSpace(challenger.Text) != "" {
		finalResp.Text = challenger.Text
		finalResp.Model = challenger.Model
	} else {
		finalResp.Text = genResp.Text
		finalResp.Model = genResp.Model
	}
	return parseTypedResponse[T](finalResp)
}

func parseTypedResponse[T interface{ Validate() error }](resp Response) (TypedResponse[T], error) {
	text := repairJSON(resp.Text)

	var value T
	if err := json.Unmarshal([]byte(text), &value); err != nil {
		return TypedResponse[T]{Response: resp}, &retryableExtractError{
			kind: "JSON parse failed",
			hint: fmt.Sprintf("JSON parse error: %v. Please return valid JSON matching the schema.", err),
			err:  err,
		}
	}

	if err := value.Validate(); err != nil {
		return TypedResponse[T]{Response: resp, Value: value}, &retryableExtractError{
			kind: "validation failed",
			hint: fmt.Sprintf("Validation error: %v. Please fix and return valid JSON.", err),
			err:  err,
		}
	}

	return TypedResponse[T]{Response: resp, Value: value}, nil
}

// MergeByMedian selects the response whose projected value is the median.
func MergeByMedian[T any](project func(T) float64) MergeFunc[T] {
	return func(items []TypedResponse[T]) (T, error) {
		var zero T
		if len(items) == 0 {
			return zero, fmt.Errorf("llm.MergeByMedian: no items")
		}

		sorted := append([]TypedResponse[T](nil), items...)
		sort.Slice(sorted, func(i, j int) bool {
			return project(sorted[i].Value) < project(sorted[j].Value)
		})

		return sorted[(len(sorted)-1)/2].Value, nil
	}
}

// MergeByMajority selects the first response belonging to the majority bucket.
func MergeByMajority[T any, K comparable](project func(T) K) MergeFunc[T] {
	return func(items []TypedResponse[T]) (T, error) {
		var zero T
		if len(items) == 0 {
			return zero, fmt.Errorf("llm.MergeByMajority: no items")
		}

		counts := make(map[K]int, len(items))
		bestCount := 0
		var winner K
		for _, item := range items {
			key := project(item.Value)
			counts[key]++
			if counts[key] > bestCount {
				bestCount = counts[key]
				winner = key
			}
		}

		for _, item := range items {
			if project(item.Value) == winner {
				return item.Value, nil
			}
		}

		return zero, fmt.Errorf("llm.MergeByMajority: winner not found")
	}
}

// repairJSON attempts to clean common LLM artifacts from JSON output:
// - Strip markdown code fences (```json ... ```)
// - Trim leading/trailing whitespace
func repairJSON(text string) string {
	text = strings.TrimSpace(text)

	// Strip markdown code fences.
	if strings.HasPrefix(text, "```") {
		// Remove opening fence (possibly with language hint).
		if idx := strings.Index(text, "\n"); idx != -1 {
			text = text[idx+1:]
		}
		// Remove closing fence.
		if idx := strings.LastIndex(text, "```"); idx != -1 {
			text = text[:idx]
		}
		text = strings.TrimSpace(text)
	}

	return text
}

// appendRetryHint appends a retry hint to the user prompt.
func appendRetryHint(built BuiltPrompt, hint string) BuiltPrompt {
	return BuiltPrompt{
		System: built.System,
		User:   built.User + "\n\n" + hint,
	}
}
