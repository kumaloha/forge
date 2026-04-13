package llm

import (
	"context"
	"fmt"
	"sync"
)

// JudgeVarsBuilder derives judge input vars from the shared base vars plus
// bull/bear typed outputs.
type JudgeVarsBuilder[TBull any, TBear any] func(baseVars map[string]any, bull TBull, bear TBear) map[string]any

// FacetRunner executes one typed prompt repeatedly with different vars.
// Domain code owns the prompt IDs; runtime owns loading + execution mechanics.
type FacetRunner[T interface{ Validate() error }] struct {
	rt   *Runtime
	spec PromptSpec
}

// NewFacetRunner constructs a FacetRunner from a prompt ID.
func NewFacetRunner[T interface{ Validate() error }](rt *Runtime, loader *PromptLoader, promptID string) (*FacetRunner[T], error) {
	if loader == nil {
		return nil, fmt.Errorf("llm.NewFacetRunner: prompt loader is required")
	}
	spec, err := loader.Load(promptID)
	if err != nil {
		return nil, fmt.Errorf("llm.NewFacetRunner: load %q: %w", promptID, err)
	}
	return NewFacetRunnerWithSpec[T](rt, spec), nil
}

// NewFacetRunnerWithSpec constructs a FacetRunner from a preloaded PromptSpec.
func NewFacetRunnerWithSpec[T interface{ Validate() error }](rt *Runtime, spec PromptSpec) *FacetRunner[T] {
	return &FacetRunner[T]{rt: rt, spec: spec}
}

// Run executes the configured prompt and returns the typed output.
func (r *FacetRunner[T]) Run(ctx context.Context, vars map[string]any, opts ExtractOptions[T]) (TypedResponse[T], error) {
	if r == nil || r.rt == nil {
		return TypedResponse[T]{}, fmt.Errorf("llm.FacetRunner: runtime is required")
	}
	return Extract[T](ctx, r.rt, r.spec, vars, opts)
}

// Spec exposes the loaded prompt spec for callers that need prompt metadata.
func (r *FacetRunner[T]) Spec() PromptSpec {
	return r.spec
}

// DebateResponses contains the typed outputs of a bull/bear/judge run.
type DebateResponses[TBull any, TBear any, TJudge any] struct {
	Bull  TypedResponse[TBull]
	Bear  TypedResponse[TBear]
	Judge TypedResponse[TJudge]
}

// TextDebateResponses contains the raw bull/bear text plus the typed judge output.
type TextDebateResponses[TJudge any] struct {
	Bull  Response
	Bear  Response
	Judge TypedResponse[TJudge]
}

// NamedTextDebate bundles a label with a prepared text-debate runner.
type NamedTextDebate[TJudge interface{ Validate() error }] struct {
	Name   string
	Runner *TextDebateRunner[TJudge]
}

// DebateRunner executes a fixed bull/bear/judge prompt trio.
// Domain code defines prompt IDs and how judge vars are assembled.
type DebateRunner[
	TBull interface{ Validate() error },
	TBear interface{ Validate() error },
	TJudge interface{ Validate() error },
] struct {
	rt        *Runtime
	bullSpec  PromptSpec
	bearSpec  PromptSpec
	judgeSpec PromptSpec
}

// NewDebateRunner constructs a DebateRunner from prompt IDs.
func NewDebateRunner[
	TBull interface{ Validate() error },
	TBear interface{ Validate() error },
	TJudge interface{ Validate() error },
](rt *Runtime, loader *PromptLoader, bullID, bearID, judgeID string) (*DebateRunner[TBull, TBear, TJudge], error) {
	if loader == nil {
		return nil, fmt.Errorf("llm.NewDebateRunner: prompt loader is required")
	}
	bullSpec, err := loader.Load(bullID)
	if err != nil {
		return nil, fmt.Errorf("llm.NewDebateRunner: load bull %q: %w", bullID, err)
	}
	bearSpec, err := loader.Load(bearID)
	if err != nil {
		return nil, fmt.Errorf("llm.NewDebateRunner: load bear %q: %w", bearID, err)
	}
	judgeSpec, err := loader.Load(judgeID)
	if err != nil {
		return nil, fmt.Errorf("llm.NewDebateRunner: load judge %q: %w", judgeID, err)
	}
	return NewDebateRunnerWithSpecs[TBull, TBear, TJudge](rt, bullSpec, bearSpec, judgeSpec), nil
}

// NewDebateRunnerWithSpecs constructs a DebateRunner from preloaded PromptSpecs.
func NewDebateRunnerWithSpecs[
	TBull interface{ Validate() error },
	TBear interface{ Validate() error },
	TJudge interface{ Validate() error },
](rt *Runtime, bullSpec, bearSpec, judgeSpec PromptSpec) *DebateRunner[TBull, TBear, TJudge] {
	return &DebateRunner[TBull, TBear, TJudge]{
		rt:        rt,
		bullSpec:  bullSpec,
		bearSpec:  bearSpec,
		judgeSpec: judgeSpec,
	}
}

// Run executes bull, bear, then judge using shared base vars.
func (r *DebateRunner[TBull, TBear, TJudge]) Run(
	ctx context.Context,
	baseVars map[string]any,
	buildJudgeVars JudgeVarsBuilder[TBull, TBear],
) (DebateResponses[TBull, TBear, TJudge], error) {
	if r == nil || r.rt == nil {
		return DebateResponses[TBull, TBear, TJudge]{}, fmt.Errorf("llm.DebateRunner: runtime is required")
	}

	bull, err := Extract[TBull](ctx, r.rt, r.bullSpec, cloneVars(baseVars), ExtractOptions[TBull]{})
	if err != nil {
		return DebateResponses[TBull, TBear, TJudge]{}, fmt.Errorf("llm.DebateRunner: bull %q: %w", r.bullSpec.ID, err)
	}

	bear, err := Extract[TBear](ctx, r.rt, r.bearSpec, cloneVars(baseVars), ExtractOptions[TBear]{})
	if err != nil {
		return DebateResponses[TBull, TBear, TJudge]{}, fmt.Errorf("llm.DebateRunner: bear %q: %w", r.bearSpec.ID, err)
	}

	judgeVars := cloneVars(baseVars)
	if buildJudgeVars != nil {
		if built := buildJudgeVars(cloneVars(baseVars), bull.Value, bear.Value); built != nil {
			judgeVars = built
		}
	}

	judge, err := Extract[TJudge](ctx, r.rt, r.judgeSpec, judgeVars, ExtractOptions[TJudge]{})
	if err != nil {
		return DebateResponses[TBull, TBear, TJudge]{}, fmt.Errorf("llm.DebateRunner: judge %q: %w", r.judgeSpec.ID, err)
	}

	return DebateResponses[TBull, TBear, TJudge]{
		Bull:  bull,
		Bear:  bear,
		Judge: judge,
	}, nil
}

func cloneVars(vars map[string]any) map[string]any {
	if vars == nil {
		return map[string]any{}
	}
	cloned := make(map[string]any, len(vars))
	for k, v := range vars {
		cloned[k] = v
	}
	return cloned
}

// TextJudgeVarsBuilder derives judge input vars from shared base vars plus the
// raw bull/bear text responses.
type TextJudgeVarsBuilder func(baseVars map[string]any, bull Response, bear Response) map[string]any

// TextDebateRunner executes bull/bear as raw chat prompts and judge as typed extraction.
type TextDebateRunner[TJudge interface{ Validate() error }] struct {
	rt        *Runtime
	bullSpec  PromptSpec
	bearSpec  PromptSpec
	judgeSpec PromptSpec
}

// NewTextDebateRunner constructs a TextDebateRunner from prompt IDs.
func NewTextDebateRunner[TJudge interface{ Validate() error }](
	rt *Runtime,
	loader *PromptLoader,
	bullID, bearID, judgeID string,
) (*TextDebateRunner[TJudge], error) {
	if loader == nil {
		return nil, fmt.Errorf("llm.NewTextDebateRunner: prompt loader is required")
	}
	bullSpec, err := loader.Load(bullID)
	if err != nil {
		return nil, fmt.Errorf("llm.NewTextDebateRunner: load bull %q: %w", bullID, err)
	}
	bearSpec, err := loader.Load(bearID)
	if err != nil {
		return nil, fmt.Errorf("llm.NewTextDebateRunner: load bear %q: %w", bearID, err)
	}
	judgeSpec, err := loader.Load(judgeID)
	if err != nil {
		return nil, fmt.Errorf("llm.NewTextDebateRunner: load judge %q: %w", judgeID, err)
	}
	return NewTextDebateRunnerWithSpecs[TJudge](rt, bullSpec, bearSpec, judgeSpec), nil
}

// NewTextDebateRunnerWithSpecs constructs a TextDebateRunner from preloaded PromptSpecs.
func NewTextDebateRunnerWithSpecs[TJudge interface{ Validate() error }](
	rt *Runtime,
	bullSpec, bearSpec, judgeSpec PromptSpec,
) *TextDebateRunner[TJudge] {
	return &TextDebateRunner[TJudge]{
		rt:        rt,
		bullSpec:  bullSpec,
		bearSpec:  bearSpec,
		judgeSpec: judgeSpec,
	}
}

// Run executes bull, bear, then judge.
func (r *TextDebateRunner[TJudge]) Run(
	ctx context.Context,
	baseVars map[string]any,
	buildJudgeVars TextJudgeVarsBuilder,
) (TextDebateResponses[TJudge], error) {
	if r == nil || r.rt == nil {
		return TextDebateResponses[TJudge]{}, fmt.Errorf("llm.TextDebateRunner: runtime is required")
	}

	bull, err := r.rt.Chat(ctx, r.bullSpec, cloneVars(baseVars))
	if err != nil {
		return TextDebateResponses[TJudge]{}, fmt.Errorf("llm.TextDebateRunner: bull %q: %w", r.bullSpec.ID, err)
	}

	bear, err := r.rt.Chat(ctx, r.bearSpec, cloneVars(baseVars))
	if err != nil {
		return TextDebateResponses[TJudge]{}, fmt.Errorf("llm.TextDebateRunner: bear %q: %w", r.bearSpec.ID, err)
	}

	judgeVars := cloneVars(baseVars)
	if buildJudgeVars != nil {
		if built := buildJudgeVars(cloneVars(baseVars), bull, bear); built != nil {
			judgeVars = built
		}
	}

	judge, err := Extract[TJudge](ctx, r.rt, r.judgeSpec, judgeVars, ExtractOptions[TJudge]{})
	if err != nil {
		return TextDebateResponses[TJudge]{}, fmt.Errorf("llm.TextDebateRunner: judge %q: %w", r.judgeSpec.ID, err)
	}

	return TextDebateResponses[TJudge]{
		Bull:  bull,
		Bear:  bear,
		Judge: judge,
	}, nil
}

// RunNamedTextDebatesParallel runs multiple named text-debate runners in parallel.
// The caller still owns the shared prompt input and judge-input builder logic.
func RunNamedTextDebatesParallel[TJudge interface{ Validate() error }](
	ctx context.Context,
	baseVars map[string]any,
	buildJudgeVars TextJudgeVarsBuilder,
	items []NamedTextDebate[TJudge],
) (map[string]TextDebateResponses[TJudge], error) {
	type namedResult struct {
		name string
		resp TextDebateResponses[TJudge]
		err  error
	}

	results := make([]namedResult, len(items))
	var wg sync.WaitGroup

	for i, item := range items {
		wg.Add(1)
		go func(idx int, debate NamedTextDebate[TJudge]) {
			defer wg.Done()
			resp, err := debate.Runner.Run(ctx, cloneVars(baseVars), buildJudgeVars)
			results[idx] = namedResult{name: debate.Name, resp: resp, err: err}
		}(i, item)
	}
	wg.Wait()

	out := make(map[string]TextDebateResponses[TJudge], len(items))
	for _, result := range results {
		if result.err != nil {
			return nil, fmt.Errorf("llm.RunNamedTextDebatesParallel: %s: %w", result.name, result.err)
		}
		out[result.name] = result.resp
	}
	return out, nil
}
