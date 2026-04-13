package llm

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// FewShot is a single example in a prompt template.
type FewShot struct {
	Input  string `yaml:"input"`
	Output string `yaml:"output"`
	Reason string `yaml:"reason"`
}

// PromptSpec represents a loaded prompt template. Fields map 1:1 to the
// YAML prompt files under prompts/.
type PromptSpec struct {
	ID           string    `yaml:"id"`
	Version      string    `yaml:"version"`
	Role         string    `yaml:"role"`
	Definitions  string    `yaml:"definitions"`
	Criteria     string    `yaml:"criteria"`
	Boundaries   string    `yaml:"boundaries"`
	FewShots     []FewShot `yaml:"few_shots"`
	OutputFormat string    `yaml:"output_format"`
}

// MemoryBlock represents a previous attempt that the model should learn from.
type MemoryBlock struct {
	Input  string
	Output string
	Note   string // e.g. "this failed because..."
}

// BuiltPrompt is the final system+user prompt pair ready for an API call.
type BuiltPrompt struct {
	System string
	User   string
}

// Build constructs the final prompt from a template, variable substitutions,
// and optional memory blocks from previous attempts.
//
// System prompt = Role + Definitions + Criteria + Boundaries + OutputFormat
// (each non-empty section separated by double newline, empty sections omitted).
//
// User prompt = FewShots + MemoryBlocks + input text from vars["query"] or vars["input"].
func (p PromptSpec) Build(vars map[string]any, memory []MemoryBlock) BuiltPrompt {
	// --- system ---
	var sysParts []string
	for _, section := range []string{p.Role, p.Definitions, p.Criteria, p.Boundaries, p.OutputFormat} {
		if s := strings.TrimSpace(section); s != "" {
			sysParts = append(sysParts, s)
		}
	}

	// --- user ---
	var userParts []string

	// few-shots
	if len(p.FewShots) > 0 {
		for i, fs := range p.FewShots {
			lines := []string{fmt.Sprintf("Example %d:", i+1)}
			lines = append(lines, "Input: "+fs.Input)
			lines = append(lines, "Output: "+fs.Output)
			if fs.Reason != "" {
				lines = append(lines, "Reason: "+fs.Reason)
			}
			userParts = append(userParts, strings.Join(lines, "\n"))
		}
	}

	// memory blocks
	if len(memory) > 0 {
		var memLines []string
		memLines = append(memLines, "Previous attempts:")
		for _, m := range memory {
			memLines = append(memLines, "- Input: "+m.Input)
			memLines = append(memLines, "  Output: "+m.Output)
			if m.Note != "" {
				memLines = append(memLines, "  Note: "+m.Note)
			}
		}
		userParts = append(userParts, strings.Join(memLines, "\n"))
	}

	// input text: prefer "query", fall back to "input"
	if vars != nil {
		var input string
		if q, ok := vars["query"]; ok {
			input = fmt.Sprint(q)
		} else if inp, ok := vars["input"]; ok {
			input = fmt.Sprint(inp)
		}
		if input != "" {
			userParts = append(userParts, input)
		}
	}

	return BuiltPrompt{
		System: strings.Join(sysParts, "\n\n"),
		User:   strings.Join(userParts, "\n\n"),
	}
}

// PromptLoader loads and caches prompt templates from an explicit prompt directory.
type PromptLoader struct {
	dir   string // prompt directory; prompt assets live directly under this dir
	mu    sync.RWMutex
	cache map[string]PromptSpec
}

// NewPromptLoaderFromDir creates a loader rooted at promptDir.
// Prompt assets are expected directly under promptDir.
func NewPromptLoaderFromDir(promptDir string) *PromptLoader {
	return &PromptLoader{
		dir:   promptDir,
		cache: make(map[string]PromptSpec),
	}
}

// Load reads and parses a prompt by its ID (e.g. "cognition/fundamental/comprehensibility_gate").
// The ID is converted to a file path by replacing "/" with the OS path separator
// and appending ".yaml". Results are cached.
func (l *PromptLoader) Load(id string) (PromptSpec, error) {
	// check cache first
	l.mu.RLock()
	if spec, ok := l.cache[id]; ok {
		l.mu.RUnlock()
		return spec, nil
	}
	l.mu.RUnlock()

	// convert id to filesystem path
	parts := strings.Split(id, "/")
	relPath := filepath.Join(parts...) + ".yaml"
	fullPath := filepath.Join(l.dir, relPath)

	data, err := os.ReadFile(fullPath)
	if err != nil {
		return PromptSpec{}, fmt.Errorf("llm.PromptLoader: load %q: %w", id, err)
	}

	var spec PromptSpec
	if err := yaml.Unmarshal(data, &spec); err != nil {
		return PromptSpec{}, fmt.Errorf("llm.PromptLoader: parse %q: %w", id, err)
	}

	// validate that the file's id field matches the requested id
	if spec.ID != id {
		return PromptSpec{}, fmt.Errorf("llm.PromptLoader: id mismatch in %q: file declares %q", id, spec.ID)
	}

	// cache and return
	l.mu.Lock()
	l.cache[id] = spec
	l.mu.Unlock()

	return spec, nil
}
