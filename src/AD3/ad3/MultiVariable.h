// Copyright (c) 2012 Andre Martins
// All Rights Reserved.
//
// This file is part of AD3 2.0.
//
// AD3 2.0 is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// AD3 2.0 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with AD3 2.0.  If not, see <http://www.gnu.org/licenses/>.

#ifndef MULTI_VARIABLE_H
#define MULTI_VARIABLE_H

#include "Factor.h"

namespace AD3 {

// A multi-valued variable.
// Each value (state) is represented as a binary variable.
class MultiVariable {
 public:
  // Number of states.
  int GetNumStates() { return binary_variables_.size(); }
  BinaryVariable *GetState(int i) {
    return binary_variables_[i];
  }
  const vector<BinaryVariable*> &GetStates() { return binary_variables_; }

  // Get/Set log-potentials.
  double GetLogPotential(int i) {
    return binary_variables_[i]->GetLogPotential();
  }
  void SetLogPotential(int i, double log_potential) {
    binary_variables_[i]->SetLogPotential(log_potential);
  }

  // Get/Set id.
  int GetId() { return id_; };
  void SetId(int id) { id_ = id; };

  // Initialize states with binary variables.
  void Initialize(const vector<BinaryVariable*> &binary_variables) {
    binary_variables_ = binary_variables;
  }

  // Link to a factor.
  void LinkToFactor(class Factor *factor) {
    factors_.push_back(factor);
  }

  // Get the degree (number of incident factors).
  int Degree() { return factors_.size(); }

 private:
  int id_; // Variable id.
  // Indices of the binary variables corresponding
  // to the values.
  vector<BinaryVariable*> binary_variables_;
  // Factors where this multi-variable belongs to.
  vector<Factor*> factors_;
};

} // namespace AD3

#endif // MULTI_VARIABLE_H
