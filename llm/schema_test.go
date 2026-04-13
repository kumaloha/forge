package llm

import (
	"testing"
)

type simpleStruct struct {
	Name  string  `json:"name"`
	Score float64 `json:"score"`
	Count int     `json:"count"`
	Valid bool    `json:"valid"`
}

func TestSchemaFromSimpleStruct(t *testing.T) {
	s := SchemaFrom[simpleStruct]()

	if s.Name != "simpleStruct" {
		t.Fatalf("Name = %q, want simpleStruct", s.Name)
	}

	assertPropType(t, s.Properties, "name", "string")
	assertPropType(t, s.Properties, "score", "number")
	assertPropType(t, s.Properties, "count", "integer")
	assertPropType(t, s.Properties, "valid", "boolean")

	// All fields have no omitempty, so all should be required.
	if len(s.Required) != 4 {
		t.Fatalf("Required len = %d, want 4, got %v", len(s.Required), s.Required)
	}
}

type taggedStruct struct {
	ID       string `json:"id"`
	Label    string `json:"label,omitempty"`
	Internal string `json:"-"`
	NoTag    string
}

func TestSchemaRespectJSONTags(t *testing.T) {
	s := SchemaFrom[taggedStruct]()

	// "id" should be present and required.
	assertPropType(t, s.Properties, "id", "string")

	// "label" should be present but NOT required (omitempty).
	assertPropType(t, s.Properties, "label", "string")

	// "-" tagged field should be absent.
	if _, exists := s.Properties["Internal"]; exists {
		t.Fatal("Internal field should be skipped (json:\"-\")")
	}
	if _, exists := s.Properties["-"]; exists {
		t.Fatal("should not have a field named \"-\"")
	}

	// NoTag should use the Go field name.
	assertPropType(t, s.Properties, "NoTag", "string")

	// Required: "id" and "NoTag" (not "label" because omitempty, not "Internal" because skipped).
	assertRequired(t, s.Required, "id", true)
	assertRequired(t, s.Required, "NoTag", true)
	assertRequired(t, s.Required, "label", false)

	if len(s.Required) != 2 {
		t.Fatalf("Required len = %d, want 2, got %v", len(s.Required), s.Required)
	}
}

type nestedStruct struct {
	Info    innerInfo `json:"info"`
	Rating float64   `json:"rating"`
}

type innerInfo struct {
	Title   string `json:"title"`
	Author  string `json:"author,omitempty"`
}

func TestSchemaNestedStruct(t *testing.T) {
	s := SchemaFrom[nestedStruct]()

	infoProp, ok := s.Properties["info"].(map[string]any)
	if !ok {
		t.Fatalf("expected info to be a map, got %T", s.Properties["info"])
	}
	if infoProp["type"] != "object" {
		t.Fatalf("info.type = %v, want object", infoProp["type"])
	}

	props, ok := infoProp["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected info.properties to be a map, got %T", infoProp["properties"])
	}
	assertPropType(t, props, "title", "string")
	assertPropType(t, props, "author", "string")

	// Inner struct required should only include "title" (author has omitempty).
	req, ok := infoProp["required"].([]string)
	if !ok {
		t.Fatalf("expected required to be []string, got %T", infoProp["required"])
	}
	assertRequired(t, req, "title", true)
	assertRequired(t, req, "author", false)
}

type sliceStruct struct {
	Tags   []string  `json:"tags"`
	Scores []float64 `json:"scores"`
}

func TestSchemaSliceFields(t *testing.T) {
	s := SchemaFrom[sliceStruct]()

	tagsProp, ok := s.Properties["tags"].(map[string]any)
	if !ok {
		t.Fatalf("expected tags to be a map, got %T", s.Properties["tags"])
	}
	if tagsProp["type"] != "array" {
		t.Fatalf("tags.type = %v, want array", tagsProp["type"])
	}
	items, ok := tagsProp["items"].(map[string]any)
	if !ok {
		t.Fatalf("expected tags.items to be a map")
	}
	if items["type"] != "string" {
		t.Fatalf("tags.items.type = %v, want string", items["type"])
	}

	scoresProp := s.Properties["scores"].(map[string]any)
	scoreItems := scoresProp["items"].(map[string]any)
	if scoreItems["type"] != "number" {
		t.Fatalf("scores.items.type = %v, want number", scoreItems["type"])
	}
}

type mapStruct struct {
	Labels map[string]string `json:"labels"`
}

func TestSchemaMapField(t *testing.T) {
	s := SchemaFrom[mapStruct]()

	labelsProp, ok := s.Properties["labels"].(map[string]any)
	if !ok {
		t.Fatalf("expected labels to be a map, got %T", s.Properties["labels"])
	}
	if labelsProp["type"] != "object" {
		t.Fatalf("labels.type = %v, want object", labelsProp["type"])
	}
	addProps, ok := labelsProp["additionalProperties"].(map[string]any)
	if !ok {
		t.Fatalf("expected additionalProperties to be a map")
	}
	if addProps["type"] != "string" {
		t.Fatalf("additionalProperties.type = %v, want string", addProps["type"])
	}
}

type pointerStruct struct {
	Name *string `json:"name"`
}

func TestSchemaPointerField(t *testing.T) {
	s := SchemaFrom[pointerStruct]()
	assertPropType(t, s.Properties, "name", "string")
}

// --- helpers ---

func assertPropType(t *testing.T, props map[string]any, field, wantType string) {
	t.Helper()
	raw, ok := props[field]
	if !ok {
		t.Fatalf("property %q not found", field)
	}
	m, ok := raw.(map[string]any)
	if !ok {
		t.Fatalf("property %q is not a map: %T", field, raw)
	}
	if m["type"] != wantType {
		t.Fatalf("property %q type = %v, want %v", field, m["type"], wantType)
	}
}

func assertRequired(t *testing.T, required []string, name string, want bool) {
	t.Helper()
	found := false
	for _, r := range required {
		if r == name {
			found = true
			break
		}
	}
	if found != want {
		t.Fatalf("required(%q) = %v, want %v (required = %v)", name, found, want, required)
	}
}
