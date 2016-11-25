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

#include "Factor.h"
#include "Utils.h"

namespace AD3 {

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int FactorXOR::AddEvidence(vector<bool> *active_links,
                           vector<int> *evidence,
                           vector<int> *additional_evidence) {
  bool changes = false;

  // Look for absorbing elements.
  int k;
  for (k = 0; k < Degree(); ++k) {
    if (!(*active_links)[k]) continue;
    if ((*evidence)[k] < 0) continue;
    if ((!negated_[k] && (*evidence)[k] == 1) ||
        (negated_[k] && (*evidence)[k] == 0)) {
      break;
    }
  }
  if (k < Degree()) {
    // Found absorbing element. Set evidence to all the other inputs and 
    // disable the factor.
    for (int l = 0; l < Degree(); ++l) {
      (*active_links)[l] = false;
      if (k == l) continue;
      int value = negated_[l]? 1 : 0;
      // If evidence was set otherwise for this input, return contradiction.
      if ((*evidence)[l] >= 0 && (*evidence)[l] != value) return -1; 
      (*evidence)[l] = value;
    }
    // Return code to disable factor.
    return 2;
  } 

  // Look for neutral elements.
  int num_active = 0;
  for (k = 0; k < Degree(); ++k) {
    if (!(*active_links)[k]) continue;
    ++num_active;
    if ((*evidence)[k] < 0) continue;
    if ((!negated_[k] && (*evidence)[k] == 0) ||
        (negated_[k] && (*evidence)[k] == 1)) {
      // Neutral element found. Make it inactive and proceed.
      (*active_links)[k] = false;
      --num_active;
      changes = true;
    }
  }
  // If there are no active variables, return contradiction.
  if (num_active == 0) return -1;
  // If there is only one active variable, set evidence to that variable
  // and disable the factor.
  if (num_active == 1) {
    for (k = 0; k < Degree(); ++k) {
      if ((*active_links)[k]) break;
    }
    assert(k < Degree());
    (*active_links)[k] = false;
    int value = negated_[k]? 0 : 1;
    // If evidence was set otherwise for this input, return contradiction.
    if ((*evidence)[k] >= 0 && (*evidence)[k] != value) return -1; 
    (*evidence)[k] = value;
    return 2;        
  }

  return changes? 1 : 0;
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void FactorXOR::SolveMAP(const vector<double> &variable_log_potentials,
                         const vector<double> &additional_log_potentials,
                         vector<double> *variable_posteriors,
                         vector<double> *additional_posteriors,
                         double *value) {
  variable_posteriors->resize(variable_log_potentials.size());

  // Create a local copy of the log potentials.
  vector<double> log_potentials(variable_log_potentials);

  int first = -1;
  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) log_potentials[f] = -log_potentials[f];
  }

  *value = 0.0;
  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) *value -= log_potentials[f];
  }

  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (first < 0 || log_potentials[f] > log_potentials[first]) first = f;
  }

  *value += log_potentials[first];
  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) {
      (*variable_posteriors)[f] = 1.0;
    } else {
      (*variable_posteriors)[f] = 0.0;
    }
  }
  (*variable_posteriors)[first] = negated_[first]? 0.0 : 1.0;
}

// Solve the QP (local subproblem in the AD3 algorithm).
void FactorXOR::SolveQP(const vector<double> &variable_log_potentials,
                        const vector<double> &additional_log_potentials,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors) {
  variable_posteriors->resize(variable_log_potentials.size());

  for (int f = 0; f < binary_variables_.size(); ++f) {
    (*variable_posteriors)[f] = negated_[f]? 
        1 - variable_log_potentials[f] : variable_log_potentials[f];
  }

  project_onto_simplex_cached(&(*variable_posteriors)[0],
                              binary_variables_.size(), 1.0, last_sort_);

  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) {
      (*variable_posteriors)[f] = 1 - (*variable_posteriors)[f];
    }
  }
}

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int FactorAtMostOne::AddEvidence(vector<bool> *active_links,
                                 vector<int> *evidence,
                                 vector<int> *additional_evidence) {
  bool changes = false;

  // Look for absorbing elements.
  int k;
  for (k = 0; k < Degree(); ++k) {
    if (!(*active_links)[k]) continue;
    if ((*evidence)[k] < 0) continue;
    if ((!negated_[k] && (*evidence)[k] == 1) ||
        (negated_[k] && (*evidence)[k] == 0)) {
      break;
    }
  }
  if (k < Degree()) {
    // Found absorbing element. Set evidence to all the other inputs and 
    // disable the factor.
    for (int l = 0; l < Degree(); ++l) {
      (*active_links)[l] = false;
      if (k == l) continue;
      int value = negated_[l]? 1 : 0;
      // If evidence was set otherwise for this input, return contradiction.
      if ((*evidence)[l] >= 0 && (*evidence)[l] != value) return -1; 
      (*evidence)[l] = value;
    }
    // Return code to disable factor.
    return 2;
  } 

  // Look for neutral elements.
  int num_active = 0;
  for (k = 0; k < Degree(); ++k) {
    if (!(*active_links)[k]) continue;
    ++num_active;
    if ((*evidence)[k] < 0) continue;
    if ((!negated_[k] && (*evidence)[k] == 0) ||
        (negated_[k] && (*evidence)[k] == 1)) {
      // Neutral element found. Make it inactive and proceed.
      (*active_links)[k] = false;
      --num_active;
      changes = true;
    }
  }
  // If there are no active variables, disable the factor.
  if (num_active == 0) return 2;
  // If there is only one active variable, disable that link
  // and disable the factor.
  if (num_active == 1) {
    for (k = 0; k < Degree(); ++k) {
      if ((*active_links)[k]) break;
    }
    assert(k < Degree());
    (*active_links)[k] = false;
    return 2;        
  }

  return changes? 1 : 0;
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void FactorAtMostOne::SolveMAP(const vector<double> &variable_log_potentials,
                               const vector<double> &additional_log_potentials,
                               vector<double> *variable_posteriors,
                               vector<double> *additional_posteriors,
                               double *value) {
  variable_posteriors->resize(variable_log_potentials.size());

  // Create a local copy of the log potentials.
  vector<double> log_potentials(variable_log_potentials);

  int first = -1;
  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) log_potentials[f] = -log_potentials[f];
  }

  *value = 0.0;
  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) *value -= log_potentials[f];
  }

  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (first < 0 || log_potentials[f] > log_potentials[first]) first = f;
  }

  bool all_zeros = true;
  if (log_potentials[first] > 0.0) {
    *value += log_potentials[first];
    all_zeros = false;
  }

  for (int f = 0; f < binary_variables_.size(); f++) {
    if (negated_[f]) {
      (*variable_posteriors)[f] = 1.0;
    } else {
      (*variable_posteriors)[f] = 0.0;
    }
  }

  if (!all_zeros) (*variable_posteriors)[first] = negated_[first]? 0.0 : 1.0;
}

// Solve the QP (local subproblem in the AD3 algorithm).
void FactorAtMostOne::SolveQP(const vector<double> &variable_log_potentials,
                              const vector<double> &additional_log_potentials,
                              vector<double> *variable_posteriors,
                              vector<double> *additional_posteriors) {
  variable_posteriors->resize(variable_log_potentials.size());

  // Try to solve the problem with clipping.
  double s = 0.0;
  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) {
      if (variable_log_potentials[f] > 1.0) {
        (*variable_posteriors)[f] = 1.0;
      } else {
        (*variable_posteriors)[f] = variable_log_potentials[f];
        s += 1.0 - (*variable_posteriors)[f];
      }
    } else {
      if (variable_log_potentials[f] < 0.0) {
        (*variable_posteriors)[f] = 0.0;
      } else {
        (*variable_posteriors)[f] = variable_log_potentials[f];
        s += (*variable_posteriors)[f];
      }
    }
    if (s > 1.0) break;
  }
  if (s <= 1.0) return;

  // If it doesn't work, then solve the XOR.
  for (int f = 0; f < binary_variables_.size(); ++f) {
    (*variable_posteriors)[f] = negated_[f]?
        1 - variable_log_potentials[f] : variable_log_potentials[f];
  }

  project_onto_simplex_cached(&(*variable_posteriors)[0],
                              binary_variables_.size(), 1.0, last_sort_);

  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) {
      (*variable_posteriors)[f] = 1 - (*variable_posteriors)[f];
    }
  }
}

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int FactorOR::AddEvidence(vector<bool> *active_links,
                          vector<int> *evidence,
                          vector<int> *additional_evidence) {
  bool changes = false;

  // Look for absorbing elements.
  int k;
  for (k = 0; k < Degree(); ++k) {
    if (!(*active_links)[k]) continue;
    if ((*evidence)[k] < 0) continue;
    if ((!negated_[k] && (*evidence)[k] == 1) ||
        (negated_[k] && (*evidence)[k] == 0)) {
      break;
    }
  }
  if (k < Degree()) {
    // Found absorbing element. Disable the factor and all links.
    for (int l = 0; l < Degree(); ++l) {
      if (!(*active_links)[l]) continue;
      (*active_links)[l] = false;
    }    
    // Return code to disable factor.
    return 2;
  } 

  // Look for neutral elements.
  int num_active = 0;
  for (k = 0; k < Degree(); ++k) {
    if (!(*active_links)[k]) continue;
    ++num_active;
    if ((*evidence)[k] < 0) continue;
    if ((!negated_[k] && (*evidence)[k] == 0) ||
        (negated_[k] && (*evidence)[k] == 1)) {
      // Neutral element found. Make it inactive and proceed.
      (*active_links)[k] = false;
      --num_active;
      changes = true;
    }
  }

  // If there are no active variables, return contradiction.
  if (num_active == 0) return -1;
  // If there is only one active variable, set evidence to that variable
  // and disable the factor.
  if (num_active == 1) {
    for (k = 0; k < Degree(); ++k) {
      if ((*active_links)[k]) break;
    }
    assert(k < Degree());
    (*active_links)[k] = false;
    int value = negated_[k]? 0 : 1;
    // If evidence was set otherwise for this input, return contradiction.
    if ((*evidence)[k] >= 0 && (*evidence)[k] != value) return -1; 
    (*evidence)[k] = value;
    return 2;        
  }

  return changes? 1 : 0;
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void FactorOR::SolveMAP(const vector<double> &variable_log_potentials,
                        const vector<double> &additional_log_potentials,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors,
                        double *value) {
  variable_posteriors->resize(variable_log_potentials.size());

  // Create a local copy of the log potentials.
  vector<double> log_potentials(variable_log_potentials);

  int first = -1;
  double valaux;
  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) log_potentials[f] = -log_potentials[f];
  }

  *value = 0.0;
  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) *value -= log_potentials[f];
  }

  for (int f = 0; f < binary_variables_.size(); ++f) {
    valaux = log_potentials[f];
    if (valaux < 0.0) {
      valaux = 0.0;
      (*variable_posteriors)[f] = negated_[f]? 1.0 : 0.0;
    } else {
      (*variable_posteriors)[f] = negated_[f]? 0.0 : 1.0;
    }
    *value += valaux;
  }

  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (first < 0 || log_potentials[f] > log_potentials[first]) {
      first = f;
    }
  }
  valaux = log_potentials[first];
  //valaux = min(0,valaux);
  if (valaux > 0.0) {
    valaux = 0.0;
  } else {
    (*variable_posteriors)[first] = negated_[first]? 0.0 : 1.0;
  }
  *value += valaux;
}

// Solve the QP (local subproblem in the AD3 algorithm).
void FactorOR::SolveQP(const vector<double> &variable_log_potentials,
                       const vector<double> &additional_log_potentials,
                       vector<double> *variable_posteriors,
                       vector<double> *additional_posteriors) {
  variable_posteriors->resize(variable_log_potentials.size());

  for (int f = 0; f < binary_variables_.size(); ++f) {
    (*variable_posteriors)[f] = negated_[f]? 
        1 - variable_log_potentials[f] : variable_log_potentials[f];
    if ((*variable_posteriors)[f] < 0.0) {
      (*variable_posteriors)[f] = 0.0;
    } else if ((*variable_posteriors)[f] > 1.0) {
      (*variable_posteriors)[f] = 1.0;
    }
  }

  double s = 0.0;
  for (int f = 0; f < binary_variables_.size(); ++f) {
    s += (*variable_posteriors)[f];
  }

  if (s < 1.0) {
    for (int f = 0; f < binary_variables_.size(); ++f) {
      (*variable_posteriors)[f] = negated_[f]? 
          1 - variable_log_potentials[f] : variable_log_potentials[f];
    }
    project_onto_simplex_cached(&(*variable_posteriors)[0], 
                                binary_variables_.size(), 
                                1.0, 
                                last_sort_);
  }

  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) {
      (*variable_posteriors)[f] = 1 - (*variable_posteriors)[f];
    }
  }
}

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int FactorOROUT::AddEvidence(vector<bool> *active_links,
                             vector<int> *evidence,
                             vector<int> *additional_evidence) {
  bool changes = false;

  // 1) Look for absorbing elements in the first N-1 inputs.
  int k;
  for (k = 0; k < Degree() - 1; ++k) {
    if (!(*active_links)[k]) continue;
    if ((*evidence)[k] < 0) continue;
    if ((!negated_[k] && (*evidence)[k] == 1) ||
        (negated_[k] && (*evidence)[k] == 0)) {
      break;
    }
  }
  if (k < Degree() - 1) {
    // Found absorbing element. Set evidence to the last input and 
    // disable the factor.
    for (int l = 0; l < Degree(); ++l) {
      if (!(*active_links)[l]) continue;
      (*active_links)[l] = false;
    }
    int l = Degree() - 1;
    int value = negated_[l]? 0 : 1;
    // If evidence was set otherwise for this input, return contradiction.
    if ((*evidence)[l] >= 0 && (*evidence)[l] != value) return -1; 
    (*evidence)[l] = value;
    
    // Return code to disable factor.
    return 2;
  } 

  // 2) Look for neutral elements in the first N-1 inputs.
  int num_active = 0;
  for (k = 0; k < Degree() - 1; ++k) {
    if (!(*active_links)[k]) continue;
    ++num_active;
    if ((*evidence)[k] < 0) continue;
    if ((!negated_[k] && (*evidence)[k] == 0) ||
        (negated_[k] && (*evidence)[k] == 1)) {
      // Neutral element found. Make it inactive and proceed.
      (*active_links)[k] = false;
      --num_active;
      changes = true;
    }
  }

  // If there are no active variables in the first N-1 inputs,
  // set evidence to the last variable and disable the factor.
  if (num_active == 0) {
    int l = Degree() - 1;
    (*active_links)[l] = false;
    int value = negated_[l]? 1 : 0;
    // If evidence was set otherwise for this input, return contradiction.
    if ((*evidence)[l] >= 0 && (*evidence)[l] != value) return -1; 
    (*evidence)[l] = value;
    return 2;    
  }

  // 3) Handle the last input.
  k = Degree() - 1;
  if ((*active_links)[k] && (*evidence)[k] >= 0) {
    if ((!negated_[k] && (*evidence)[k] == 0) ||
        (negated_[k] && (*evidence)[k] == 1)) {
      // Absorbing element. Set evidence to all variables and disable the 
      // factor.
      (*active_links)[k] = false;
      for (int l = 0; l < Degree() - 1; ++l) {
        if (!(*active_links)[l]) continue;
        (*active_links)[l] = false;
        int value = negated_[l]? 1 : 0;
        // If evidence was set otherwise for this input, return contradiction.
        if ((*evidence)[l] >= 0 && (*evidence)[l] != value) return -1; 
        (*evidence)[l] = value;
      }
      return 2;
    } else {
      // (!negated_[k] && evidence[k] == 1) ||
      // (negated_[k] && evidence[k] == 0))
      // For now, just disable the last link.
      // Later, turn the factor into a OR factor.
      (*active_links)[k] = false;
      changes = true;
    }
  }

  return changes? 1 : 0;
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
void FactorOROUT::SolveMAP(const vector<double> &variable_log_potentials,
                           const vector<double> &additional_log_potentials,
                           vector<double> *variable_posteriors,
                           vector<double> *additional_posteriors,
                           double *value) {
  variable_posteriors->resize(variable_log_potentials.size());

  // Create a local copy of the log potentials.
  vector<double> log_potentials(variable_log_potentials);

  int first = -1;
  double valaux;
  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) log_potentials[f] = -log_potentials[f];
  }

  for (int f = 0; f < binary_variables_.size(); ++f) {
    (*variable_posteriors)[f] = 0.0;
  }

  for (int f = 0; f < binary_variables_.size() - 1; ++f) {
    if (first < 0 || log_potentials[f] > log_potentials[first]) {
      first = f;
    }
  }
  valaux = log_potentials[first];
  //valaux = min(0,valaux);
  if (valaux > 0.0) {
    valaux = 0.0;
  } else {
    (*variable_posteriors)[first] = 1.0;
  }
  *value = valaux;

  for (int f = 0; f < binary_variables_.size() - 1; ++f) {
    valaux = log_potentials[f];
    //valaux = max(0,valaux);
    if (valaux < 0.0) {
      valaux = 0.0;
    } else {
      (*variable_posteriors)[f] = 1.0;
    }
    *value += valaux;
  }

  *value += log_potentials[binary_variables_.size() - 1];
  if (*value < 0.0) {
    *value = 0.0;
    for (int f = 0; f < binary_variables_.size(); ++f) {
      (*variable_posteriors)[f] = 0.0;
    }
  } else {
    (*variable_posteriors)[binary_variables_.size() - 1] = 1.0;
  }

  for (int f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) {
      *value -= log_potentials[f];
      (*variable_posteriors)[f] = 1 - (*variable_posteriors)[f];
      //log_potentials[f] = -log_potentials[f];
    }
  }
}

// Solve the QP (local subproblem in the AD3 algorithm).
void FactorOROUT::SolveQP(const vector<double> &variable_log_potentials,
                          const vector<double> &additional_log_potentials,
                          vector<double> *variable_posteriors,
                          vector<double> *additional_posteriors) {
  variable_posteriors->resize(variable_log_potentials.size());

  // 1) Start by projecting onto the cubed cone = conv (.*1, 0)
  // Project onto the unit cube
  int f;
  for (f = 0; f < binary_variables_.size(); ++f) {
    (*variable_posteriors)[f] = negated_[f]? 
        1 - variable_log_potentials[f] : variable_log_potentials[f];
    if ((*variable_posteriors)[f] < 0.0) {
      (*variable_posteriors)[f] = 0.0;
    } else if ((*variable_posteriors)[f] > 1.0) {
      (*variable_posteriors)[f] = 1.0;
    }
  }

  //project_onto_box(&m_x[0], binary_variables_.size(), 0.0, 1.0, val);
  for (f = 0; f < binary_variables_.size() - 1; ++f) {
    if ((*variable_posteriors)[f] > 
        (*variable_posteriors)[binary_variables_.size() - 1]) break;
  }

  if (f < binary_variables_.size() - 1) { // max(x(1:(end-1))) > x(end)
    // Project onto cone
    for (f = 0; f < binary_variables_.size(); ++f) {
      (*variable_posteriors)[f] = negated_[f]? 
          1 - variable_log_potentials[f] : variable_log_potentials[f];
    }
    project_onto_cone_cached(&(*variable_posteriors)[0],
                             binary_variables_.size(), last_sort_);

    // Project onto the unit cube again
    //project_onto_box(&m_x[0], binary_variables_.size(), 0.0, 1.0, val);
    for (f = 0; f < binary_variables_.size(); ++f) {
      if ((*variable_posteriors)[f] < 0.0) {
        (*variable_posteriors)[f] = 0.0;
      } else if ((*variable_posteriors)[f] > 1.0) {
        (*variable_posteriors)[f] = 1.0;
      }
    }
  }

  // 2) Add the inequality  sum(x(1:(end-1)) >= x(end)
  double s = 0.0;
  for (f = 0; f < binary_variables_.size() - 1; ++f) {
    s += (*variable_posteriors)[f];
  }
  if (s < (*variable_posteriors)[binary_variables_.size() - 1]) {
    // Project onto xor with negated output
    for (f = 0; f < binary_variables_.size() - 1; ++f) {
      (*variable_posteriors)[f] = negated_[f]? 
          1 - variable_log_potentials[f] : variable_log_potentials[f];
    }
    (*variable_posteriors)[f] = negated_[f]? 
        variable_log_potentials[f] : 1 - variable_log_potentials[f];
    project_onto_simplex_cached(&(*variable_posteriors)[0], 
                                binary_variables_.size(),
                                1.0,
                                last_sort_);
    (*variable_posteriors)[f] = 1.0 - (*variable_posteriors)[f];
  }

  for (f = 0; f < binary_variables_.size(); ++f) {
    if (negated_[f]) (*variable_posteriors)[f] = 1 - (*variable_posteriors)[f];
  }
}

// Add evidence information to the factor.
// Returns 0 if nothing changed.
// Returns 1 if new evidence was set or new links were disabled,
// but factor keeps active.
// Returns 2 if factor became inactive.
// Returns -1 if a contradiction was found, in which case the
// problem is infeasible.
int FactorPAIR::AddEvidence(vector<bool> *active_links,
                            vector<int> *evidence,
                            vector<int> *additional_evidence) {
  bool changes = false;
  additional_evidence->assign(1, -1);

  // If there is no evidence, do nothing and return "no changes."
  if ((*evidence)[0] < 0 && (*evidence)[1] < 0) return 0;
  if ((*evidence)[0] >= 0 && (*evidence)[1] >= 0) {
    if ((*evidence)[0] == 1 && (*evidence)[1] == 1) {
      (*additional_evidence)[0] = 1;
    } else {
      (*additional_evidence)[0] = 0;
    }
    (*active_links)[0] = (*active_links)[1] = false;
    return 2;
  }
  // Only one of the variables has evidence. Disable all links and, depending
  // on the evidence, keep or discard the factor.
  if ((*active_links)[0] || (*active_links)[1]) {
    changes = true;
    (*active_links)[0] = false;
    (*active_links)[1] = false;
  }
  if ((*evidence)[0] >= 0) {
    if ((*evidence)[0] == 0) {
      (*additional_evidence)[0] = 0;
      return 2;
    } else {
      return changes? 1 : 0;
    }
  } else { // (*evidence)[1] >= 0. 
    if ((*evidence)[1] == 0) {
      (*additional_evidence)[0] = 0;
      return 2;
    } else {
      return changes? 1 : 0;
    }
  }
}

// Compute the MAP (local subproblem in the projected subgradient algorithm).
// Remark: (*additional_posteriors)[0] will be 1 iff 
// (*variable_posteriors)[0] = (*variable_posteriors)[1] = 1.
// Remark: assume inputs are NOT negated.
void FactorPAIR::SolveMAP(const vector<double> &variable_log_potentials,
                          const vector<double> &additional_log_potentials,
                          vector<double> *variable_posteriors,
                          vector<double> *additional_posteriors,
                          double *value) {
  variable_posteriors->resize(variable_log_potentials.size());
  additional_posteriors->resize(additional_log_potentials.size());

  double p[4] = { 0.0, // 00
                  variable_log_potentials[1], // 01
                  variable_log_potentials[0], // 10
                  variable_log_potentials[0] + variable_log_potentials[1] +
                    additional_log_potentials[0] // 11
  };

  int best = 0;
  for (int i = 1; i < 4; i++) {
    if (p[i] > p[best]) best = i;
  }

  *value = p[best];
  if (best == 0) {
    (*variable_posteriors)[0] = 0.0;
    (*variable_posteriors)[1] = 0.0;
    (*additional_posteriors)[0] = 0.0;
  } else if (best == 1) {
    (*variable_posteriors)[0] = 0.0;
    (*variable_posteriors)[1] = 1.0;
    (*additional_posteriors)[0] = 0.0;
  } else if (best == 2) {
    (*variable_posteriors)[0] = 1.0;
    (*variable_posteriors)[1] = 0.0;
    (*additional_posteriors)[0] = 0.0;
  } else { // if (best == 3)
    (*variable_posteriors)[0] = 1.0;
    (*variable_posteriors)[1] = 1.0;
    (*additional_posteriors)[0] = 1.0;
  }
}

// Solve the QP (local subproblem in the AD3 algorithm).
void FactorPAIR::SolveQP(const vector<double> &variable_log_potentials,
                         const vector<double> &additional_log_potentials,
                         vector<double> *variable_posteriors,
                         vector<double> *additional_posteriors) {
  variable_posteriors->resize(variable_log_potentials.size());
  additional_posteriors->resize(additional_log_potentials.size());

  // min 1/2 (u[0] - u0[0])^2 + (u[1] - u0[1])^2 + u0[2] * u[2], 
  // where u[2] is the edge marginal.
  // Remark: Assume inputs are NOT negated.
  double x0[3] = { variable_log_potentials[0],
                   variable_log_potentials[1],
                   -additional_log_potentials[0] 
  };

  double c = x0[2];
  if (additional_log_potentials[0] < 0) {
    x0[0] -= c;
    x0[1] = 1 - x0[1];
    c = -c;
  }

  if (x0[0] > x0[1] - c) {
    (*variable_posteriors)[0] = x0[0];
    (*variable_posteriors)[1] = x0[1] - c;
  } else if (x0[1] > x0[0] - c) {
    (*variable_posteriors)[0] = x0[0] - c;
    (*variable_posteriors)[1] = x0[1];
  } else {
    (*variable_posteriors)[0] = 0.5 * (x0[0] + x0[1] - c);
    (*variable_posteriors)[1] = (*variable_posteriors)[0];
  }

  // Project onto box.
  if ((*variable_posteriors)[0] < 0.0) {
    (*variable_posteriors)[0] = 0.0;
  } else if ((*variable_posteriors)[0] > 1.0) {
    (*variable_posteriors)[0] = 1.0;
  }
  if ((*variable_posteriors)[1] < 0.0) {
    (*variable_posteriors)[1] = 0.0;
  } else if ((*variable_posteriors)[1] > 1.0) {
    (*variable_posteriors)[1] = 1.0;
  }

  // u[2] = min(u[0], u[1]);
  (*additional_posteriors)[0] = 
      ((*variable_posteriors)[0] < (*variable_posteriors)[1])? 
          (*variable_posteriors)[0] : (*variable_posteriors)[1];

  if (additional_log_potentials[0] < 0) { // c > 0
    (*variable_posteriors)[1] = 1 - (*variable_posteriors)[1];
    (*additional_posteriors)[0] = 
      (*variable_posteriors)[0] - (*additional_posteriors)[0];
  }
}

} // namespace AD3
