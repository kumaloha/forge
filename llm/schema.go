package llm

import (
	"reflect"
	"strings"
)

// SchemaFrom generates a JSON Schema from a Go struct type using reflection.
// It handles: string, int, float64, bool, []T, map[string]T, nested structs.
// Uses json tags for field names; respects omitempty for required/optional.
func SchemaFrom[T any]() Schema {
	var zero T
	t := reflect.TypeOf(zero)
	// Dereference pointer types.
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	props, required := structSchema(t)
	return Schema{
		Name:       t.Name(),
		Properties: props,
		Required:   required,
	}
}

// structSchema walks a struct type and returns properties + required fields.
func structSchema(t reflect.Type) (map[string]any, []string) {
	props := make(map[string]any)
	var required []string

	for i := range t.NumField() {
		f := t.Field(i)
		if !f.IsExported() {
			continue
		}

		name, omit, skip := parseJSONTag(f)
		if skip {
			continue
		}

		props[name] = typeSchema(f.Type)
		if !omit {
			required = append(required, name)
		}
	}

	return props, required
}

// typeSchema converts a Go type to a JSON schema fragment.
func typeSchema(t reflect.Type) map[string]any {
	// Unwrap pointers.
	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	switch t.Kind() {
	case reflect.String:
		return map[string]any{"type": "string"}

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return map[string]any{"type": "integer"}

	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}

	case reflect.Bool:
		return map[string]any{"type": "boolean"}

	case reflect.Slice:
		items := typeSchema(t.Elem())
		return map[string]any{"type": "array", "items": items}

	case reflect.Map:
		// JSON schema: object with additionalProperties for the value type.
		valSchema := typeSchema(t.Elem())
		return map[string]any{
			"type":                 "object",
			"additionalProperties": valSchema,
		}

	case reflect.Struct:
		props, req := structSchema(t)
		s := map[string]any{
			"type":       "object",
			"properties": props,
		}
		if len(req) > 0 {
			s["required"] = req
		}
		return s

	default:
		// Fallback: treat as string.
		return map[string]any{"type": "string"}
	}
}

// parseJSONTag extracts the field name, omitempty flag, and skip flag from
// a struct field's json tag. Returns (name, omitempty, skip).
func parseJSONTag(f reflect.StructField) (string, bool, bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}

	name := f.Name
	omit := false

	if tag != "" {
		parts := strings.Split(tag, ",")
		if parts[0] != "" {
			name = parts[0]
		}
		for _, opt := range parts[1:] {
			if opt == "omitempty" {
				omit = true
			}
		}
	}

	return name, omit, false
}
