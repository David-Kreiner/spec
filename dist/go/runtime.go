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
	"fmt"
)

// Runtime configures the runtime environment.
type Runtime struct {
	Cloud      *RuntimeCloud      `json:"cloud,omitempty"`
	Kubernetes *RuntimeKubernetes `json:"kubernetes,omitempty"`
	VM         *RuntimeInstance   `json:"vm,omitempty"`
	Shell      bool               `json:"shell,omitempty"`
}

// UnmarshalJSON implement the json.Unmarshaler interface.
func (v *Runtime) UnmarshalJSON(data []byte) error {
	var out1 string
	var out2 = struct {
		Cloud      *RuntimeCloud      `json:"cloud,omitempty"`
		Kubernetes *RuntimeKubernetes `json:"kubernetes,omitempty"`
		VM         *RuntimeInstance   `json:"vm,omitempty"`
		Shell      bool               `json:"shell,omitempty"`
	}{}

	if err := json.Unmarshal(data, &out1); err != nil {
		switch out1 {
		case "cloud":
			v.Cloud = new(RuntimeCloud)
		case "vm":
			v.VM = new(RuntimeInstance)
		case "kubernetes":
			v.Kubernetes = new(RuntimeKubernetes)
		case "shell":
			v.Shell = true
		default:
			return fmt.Errorf("unknown runtime type: %s", out1)
		}
		return nil
	}

	if err := json.Unmarshal(data, &out2); err != nil {
		v.Cloud = out2.Cloud
		v.Kubernetes = out2.Kubernetes
		v.VM = out2.VM
		v.Shell = out2.Shell
		return nil
	} else {
		return err
	}
}
