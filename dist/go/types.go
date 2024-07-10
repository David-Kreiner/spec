// Copyright 2022 Harness, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package yaml

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"time"

	"github.com/docker/go-units"
)

// StringorInt represents a string or an integer.
type StringorInt int64

// UnmarshalJSON implements the unmarshal interface.
func (s *StringorInt) UnmarshalJSON(data []byte) error {
	var intType int64
	if err := json.Unmarshal(data, &intType); err == nil {
		*s = StringorInt(intType)
		return nil
	}

	var stringType string
	if err := json.Unmarshal(data, &stringType); err == nil {
		intType, err := strconv.ParseInt(stringType, 10, 64)

		if err != nil {
			return err
		}
		*s = StringorInt(intType)
		return nil
	}

	return errors.New("failed to unmarshal string or number")
}

// MemStringorInt represents a string or an integer
// the String supports notations like 10m for then Megabyte of memory
type MemStringorInt int64

// UnmarshalJSON implements the unmarshal interface.
func (s *MemStringorInt) UnmarshalJSON(data []byte) error {
	var intType int64
	if err := json.Unmarshal(data, &intType); err == nil {
		*s = MemStringorInt(intType)
		return nil
	}

	var stringType string
	if err := json.Unmarshal(data, &stringType); err == nil {
		intType, err := units.RAMInBytes(stringType)

		if err != nil {
			return err
		}
		*s = MemStringorInt(intType)
		return nil
	}

	return errors.New("failed to unmarshal memory string to integer")
}

// Stringorslice represents
// Using engine-api Strslice and augment it with YAML marshalling stuff. a string or an array of strings.
type Stringorslice []string

// UnmarshalJSON implements the unmarshal interface.
func (s *Stringorslice) UnmarshalJSON(data []byte) error {
	var stringType string
	if err := json.Unmarshal(data, &stringType); err == nil {
		*s = []string{stringType}
		return nil
	}

	var sliceType []interface{}
	if err := json.Unmarshal(data, &sliceType); err == nil {
		parts, err := toStrings(sliceType)
		if err != nil {
			return err
		}
		*s = parts
		return nil
	}

	return errors.New("failed to unmarshal string or string array")
}

// Durationorslice represents a duration string or an array
// of duration strings.
type Durationorslice []time.Duration

// UnmarshalJSON implements the unmarshal interface.
func (s *Durationorslice) UnmarshalJSON(data []byte) error {
	var stringType time.Duration
	if err := json.Unmarshal(data, &stringType); err == nil {
		*s = []time.Duration{stringType}
		return nil
	}

	var sliceType []interface{}
	if err := json.Unmarshal(data, &sliceType); err == nil {
		parts, err := toDurations(sliceType)
		if err != nil {
			return err
		}
		*s = parts
		return nil
	}

	return errors.New("failed to unmarshal duration string or duration array")
}

func toStrings(s []interface{}) ([]string, error) {
	if len(s) == 0 {
		return nil, nil
	}
	r := make([]string, len(s))
	for k, v := range s {
		if sv, ok := v.(string); ok {
			r[k] = sv
		} else {
			return nil, fmt.Errorf("cannot unmarshal %v of type %T into a string value", v, v)
		}
	}
	return r, nil
}

func toDurations(s []interface{}) ([]time.Duration, error) {
	if len(s) == 0 {
		return nil, nil
	}
	r := make([]time.Duration, len(s))
	for k, i := range s {
		switch v := i.(type) {
		case string:
			dur, err := time.ParseDuration(v)
			if err != nil {
				return nil, fmt.Errorf("cannot unmarshal %v into type duration", i)
			}
			r[k] = dur
		case int:
			r[k] = time.Duration(int64(v))
		case int64:
			r[k] = time.Duration(v)
		default:
			return nil, fmt.Errorf("cannot unmarshal %v into type duration", i)
		}
	}
	return r, nil
}
