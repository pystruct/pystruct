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

#include <iostream>
#include <math.h>
#include <sys/time.h>
#include "FactorGraph.h"
#include "Utils.h"

namespace AD3 {

// Check if there is any multi-variable which does not 
// belong to any factor, and if so, assign a XOR factor
// to the corresponding binary variables.
void FactorGraph::FixMultiVariablesWithoutFactors() {
  for (int i = 0; i < multi_variables_.size(); ++i) {
    MultiVariable *multi_variable = multi_variables_[i];
    if (multi_variable->Degree() == 0) {
      vector<BinaryVariable*> binary_variables(multi_variable->GetNumStates());
      for (int j = 0; j < multi_variable->GetNumStates(); ++j) {
        binary_variables[j] = multi_variable->GetState(j);
      }
      if (verbosity_ > 1) {
        cout << "Creating factor XOR..." << endl;
      }
      CreateFactorXOR(binary_variables);
    }
  }
}

// Convert a factor graph with multi-valued variables to one which only
// contains binary variables and hard constraint factors.
void FactorGraph::ConvertToBinaryFactorGraph(FactorGraph *binary_factor_graph) {
  // Create binary variables and connect them to XOR factors.
  vector<vector<BinaryVariable*> >
    binary_variables_from_multi(multi_variables_.size());
  for (int i = 0; i < multi_variables_.size(); ++i) {
    MultiVariable *multi_variable = multi_variables_[i];
    binary_variables_from_multi[i].resize(multi_variable->GetNumStates());
    for (int j = 0; j < multi_variable->GetNumStates(); ++j) {
      binary_variables_from_multi[i][j] = 
        binary_factor_graph->CreateBinaryVariable();
      binary_variables_from_multi[i][j]->
        SetLogPotential(multi_variable->GetLogPotential(j));
    }
    binary_factor_graph->CreateFactorXOR(binary_variables_from_multi[i]);
  }

  // Create extra binary variables for factor configurations and connect
  // them to XOR-with-output factors.
  for (int i = 0; i < factors_.size(); ++i) {
    Factor *factor = factors_[i];
    if (factor->type() == FactorTypes::FACTOR_MULTI_DENSE) {
      FactorDense *factor_dense = static_cast<FactorDense*>(factor);
      int num_configurations = factor_dense->GetNumConfigurations();

      // Create extra variables for the factor configurations.
      vector<BinaryVariable*> extra_variables(num_configurations);
      for (int index = 0; 
          index < factor_dense->GetNumConfigurations();
          ++index) {
        extra_variables[index] = 
          binary_factor_graph->CreateBinaryVariable();
        extra_variables[index]->
          SetLogPotential(factor_dense->GetAdditionalLogPotentials()[index]);
      }      

      // Create XOR-with-output factors imposing marginalization constraints.
      vector<int> states(factor_dense->GetNumMultiVariables());
      vector<vector<BinaryVariable*> > 
        binary_variables_array(factor_dense->Degree());
      for (int index = 0; 
           index < factor_dense->GetNumConfigurations();
           ++index) {
        factor_dense->GetConfigurationStates(index, &states);
        for (int j = 0; j < factor_dense->GetNumMultiVariables(); ++j) {
          MultiVariable *multi_variable = factor_dense->GetMultiVariable(j);
          int variable_index = 
            factor_dense->GetVariableIndex(j, states[j]);
          binary_variables_array[variable_index].
            push_back(extra_variables[index]);
        }
      }
      for (int j = 0; j < factor_dense->GetNumMultiVariables(); ++j) {
        MultiVariable *multi_variable = factor_dense->GetMultiVariable(j);
        for (int state = 0;
             state < multi_variable->GetNumStates();
             ++state) {
          BinaryVariable *binary_variable = 
            binary_variables_from_multi[multi_variable->GetId()][state];
          int variable_index = 
            factor_dense->GetVariableIndex(j, state);
          binary_variables_array[variable_index].push_back(binary_variable);
          binary_factor_graph->
            CreateFactorXOROUT(binary_variables_array[variable_index]);
        }
      }
    } else {    
      cout << "Error: factor type = " << factor->type()
           << " (for now, only multi-dense factors can be binarized.)"
           << endl;
    }
  }
  //binary_factor_graph->Print(cout);
}

// Copy all additional log potentials to a vector for easier manipulation.
// For each factor, store in factor_indices the starting position of the 
// additional log potentials for that factor.
void FactorGraph::CopyAdditionalLogPotentials(
    vector<double>* additional_log_potentials,
    vector<int>* factor_indices) {
  factor_indices->resize(factors_.size());
  additional_log_potentials->clear();
  for (int j = 0; j < factors_.size(); ++j) {
    Factor *factor = factors_[j];
    const vector<double> &additional_log_potentials_factor = 
      factor->GetAdditionalLogPotentials();
    (*factor_indices)[j] = additional_log_potentials->size();
    additional_log_potentials->insert(additional_log_potentials->end(),
                                      additional_log_potentials_factor.begin(),
                                      additional_log_potentials_factor.end());
  }
}

// Transform the factor graph to incorporate evidence information.
// The vector evidence is given {0,1,-1} values (-1 means no evidence). The 
// size of the vector is the number of variables plus the number of additional
// factor information, yet only the variables can have evidence values in {0,1}.
// Evidence is then propagated (to other variables and the factors), which
// may cause some variables and factors to be deleted or transformed.
// The vector recomputed_indices maps the original indices (of variables and 
// factor information) to the new indices in the transformed factor graph. 
// Entries will be set to -1 if that index is no longer part of the factor
// graph after the transformation.
int FactorGraph::AddEvidence(vector<int> *evidence,
                             vector<int> *recomputed_indices) {
  // Set array of active links and active factors.
  vector<bool> active_links(num_links_, true);
  vector<bool> active_factors(factors_.size(), true);
  bool changed = true;
  int num_passes = 0;
  while (changed) {
    changed = false;
    ++num_passes;
    int offset = GetNumVariables();
    // Go through each factor and propagate evidence.
    for (int i = 0; i < factors_.size(); ++i) {
      Factor *factor = factors_[i];
      if (!active_factors[i]) {
        offset += factor->GetAdditionalLogPotentials().size();
        continue;
      }
      // Add evidence to the factor.
      // Returns 0 if nothing changed.
      // Returns 1 if new evidence was set or new links were disabled,
      // but factor keeps active.
      // Returns 2 if factor became inactive.
      // Returns -1 if a contradiction was found, in which case the
      // problem is infeasible.
      vector<int> local_evidence(factor->Degree(), -1);
      vector<bool> local_active_links(factor->Degree(), true);
      vector<int> additional_evidence;
      for (int j = 0; j < factor->Degree(); ++j) {
        local_evidence[j] = (*evidence)[factor->GetVariable(j)->GetId()];
        local_active_links[j] = active_links[factor->GetLinkId(j)];
      }
      int ret = factor->AddEvidence(&local_active_links, &local_evidence, 
                                    &additional_evidence);
      for (int j = 0; j < factor->Degree(); ++j) {
        (*evidence)[factor->GetVariable(j)->GetId()] = local_evidence[j];
        active_links[factor->GetLinkId(j)] = local_active_links[j];
      }
      for (int j = 0; j < additional_evidence.size(); ++j) {
        assert(offset + j < evidence->size());
        (*evidence)[offset + j] = additional_evidence[j];
      }
      offset += factor->GetAdditionalLogPotentials().size();
      if (ret < 0) {
        factor->Print(cout);
        return STATUS_INFEASIBLE;
      }
      if (ret != 0) changed = true;
      if (ret == 2) active_factors[i] = false;
    }    
  }

  if (verbosity_ > 1) {
    cout << "Factor graph reduced after " << num_passes << " passes." << endl;
  }

  // Handle special factors.
  for (int i = 0; i < factors_.size(); ++i) {
    Factor *factor = factors_[i];
    if (!active_factors[i]) continue;
    // In some cases, OROUT factors may become OR factors.
    // Handle that case here.
    if (factor->type() == FactorTypes::FACTOR_OROUT) {
      int j = factor->Degree() - 1;
      if (!active_links[factor->GetLinkId(j)]) {
        // This OROUT becomes a OR.
        assert(!factor->IsVariableNegated(j) == 
               (*evidence)[factor->GetVariable(j)->GetId()]);
        FactorOR *factor_or = new FactorOR;
        factor_or->InitializeFromOROUT(factor);
        if (owned_factors_[i]) delete factor;
        factor = factor_or;
        owned_factors_[i] = true; // Mark as owned.
      }
    } else if (factor->type() == FactorTypes::FACTOR_PAIR) {
      // If both links are inactive BUT the factor is not inactive,
      // then there is evidence that one of the
      // variables is 1, hence sum that variable's log-potential with the 
      // factor log-potential and disable the factor.
      if (!active_links[factor->GetLinkId(0)] &&
          !active_links[factor->GetLinkId(1)]) {
        int evidence_first = (*evidence)[factor->GetVariable(0)->GetId()];
        int evidence_second = (*evidence)[factor->GetVariable(1)->GetId()];
        assert((evidence_first == 1 && evidence_second == -1) || 
               (evidence_first == -1 && evidence_second == 1));
        if (evidence_first == 1) {
          factor->GetVariable(1)->SetLogPotential(
              factor->GetVariable(1)->GetLogPotential() + 
              static_cast<FactorPAIR*>(factor)->GetLogPotential());
        } else { // evidence_second == 1.
          factor->GetVariable(0)->SetLogPotential(
              factor->GetVariable(0)->GetLogPotential() + 
              static_cast<FactorPAIR*>(factor)->GetLogPotential());
        }
        active_factors[i] = false;
      }
    }
  }

  // Disconnect all variables from the graph.
  // Preserve variable IDs.
  for (int i = 0; i < variables_.size(); ++i) {
    BinaryVariable *variable = variables_[i];
    variable->Disconnect();
  }

  // Delete inactive factors and rebuild the factor graph.
  vector<Factor*> copied_factors(factors_);
  vector<bool> copied_owned_factors(owned_factors_);
  factors_.clear();
  owned_factors_.clear();
  num_links_ = 0;
  recomputed_indices->assign(evidence->size(), -1);
  int offset = GetNumVariables();
  int recomputed_offset = GetNumVariables();
  for (int i = 0; i < copied_factors.size(); ++i) {
    Factor *factor = copied_factors[i];
    if (!active_factors[i]) {
      if (factor->type() == FactorTypes::FACTOR_PAIR) {
        // If both links are inactive, then there is evidence that one of the 
        // variables is 1, hence sum that variable's log-potential with the 
        // factor log-potential and disable the factor.
        int evidence_first = (*evidence)[factor->GetVariable(0)->GetId()];
        int evidence_second = (*evidence)[factor->GetVariable(1)->GetId()];
        if (evidence_first >= 0 && evidence_second < 0) {
          if (evidence_first == 1) {
            // The value of the factor is the value of the second variable.
            // Note: later need to change this variable index.
            (*recomputed_indices)[offset] = factor->GetVariable(1)->GetId();
          } else { // evidence_first = 0.
            // The factor must have evidence that it is zero.
            if (verbosity_ > 0) {
              cout << "Factor PAIR with zero evidence." << endl;
            }
            assert((*evidence)[offset] == 0);
            (*recomputed_indices)[offset] = -1;
          }
        } else if (evidence_first < 0 && evidence_second >= 0) {
          if (evidence_second == 1) {
            // The value of the factor is the value of the first variable.
            // Note: later need to change this variable index.
            (*recomputed_indices)[offset] = factor->GetVariable(0)->GetId();
          } else { // evidence_second = 0.
            // The factor must have evidence that it is zero.
            if (verbosity_ > 0) {
              cout << "Factor PAIR with zero evidence." << endl;
            }
            assert((*evidence)[offset] == 0);
            (*recomputed_indices)[offset] = -1;
          }
        }
      } else {
        assert(factor->GetAdditionalLogPotentials().size() == 0);
      }
      offset += factor->GetAdditionalLogPotentials().size();
      if (copied_owned_factors[i]) delete factor;
      continue;
    }

    // Redeclare the factor eliminating all the variables
    // which are not necessary.
    vector<BinaryVariable*> local_variables;
    vector<bool> negated;
    for (int k = 0; k < factor->Degree(); ++k) {
      int l = factor->GetLinkId(k);
      if (!active_links[l]) continue;
      local_variables.push_back(factor->GetVariable(k));
      negated.push_back(factor->IsVariableNegated(k));
    }
    DeclareFactor(factor, local_variables, negated, copied_owned_factors[i]);
    for (int l = 0; l < factor->GetAdditionalLogPotentials().size(); ++l) {
      // Note: later, after some variables be eventually deleted, we need to
      // update these "recomputed offsets."
      (*recomputed_indices)[offset + l] = recomputed_offset + l;
      // TODO: handle the case where a variable in PAIR becomes the same 
      // as the value of the factor.
    }
    offset += factor->GetAdditionalLogPotentials().size();
    recomputed_offset += factor->GetAdditionalLogPotentials().size();
  }

  // Delete variables that are disconnected. Assign evidence to some of those 
  // variables. Update variable IDs.
  int num_variables = 0;
  int num_variables_before = variables_.size();
  //recomputed_indices->assign(variables_.size(), -1);
  for (int i = 0; i < variables_.size(); ++i) {
    BinaryVariable *variable = variables_[i];
    if (variable->Degree() == 0) {
      if ((*evidence)[i] < 0) {
        if (variable->GetLogPotential() > 0) {
          (*evidence)[i] = 1;
        } else {
          (*evidence)[i] = 0;
        }
        if (verbosity_ > 1) {
          if (variable->GetLogPotential() == 0) {
            cout << "Assigned 0 value to disconnected uninformative variable."
                 << endl;
          }
        }
      }
      delete variable;
      continue;
    }
    assert((*evidence)[i] < 0);
    variables_[num_variables] = variable;
    variable->SetId(num_variables);
    (*recomputed_indices)[i] = num_variables;
    ++num_variables;
  }
  variables_.resize(num_variables);

  // Update recomputed_indices to point to the right indices after the 
  // reindexation of the variables.
  for (int i = num_variables_before; i < recomputed_indices->size(); ++i) {
    if ((*recomputed_indices)[i] < 0) continue;
    if ((*recomputed_indices)[i] < num_variables_before) {
      // Pointing to a variable; update to the new variable index.
      int j = (*recomputed_indices)[i];
      (*recomputed_indices)[i] = (*recomputed_indices)[j];
    } else {
      (*recomputed_indices)[i] += num_variables - num_variables_before;
    }
  }

  if (num_variables == 0) return STATUS_OPTIMAL_INTEGER;
  return STATUS_UNSOLVED;
}

int FactorGraph::RunPSDD(double lower_bound,
                         vector<double> *posteriors,
                         vector<double> *additional_posteriors,
                         double *value,
                         double *upper_bound) {
  timeval start, end;
  gettimeofday(&start, NULL);

  // Stopping criterion parameters.
  double residual_threshold_final = 1e-12;
  double residual_threshold = 1e-6;
  //double gap_threshold = 1e-6;

  // Caching parameters.
  vector<bool> factor_is_active(factors_.size(), true);
  vector<bool> variable_is_active(variables_.size(), false);
  int num_iterations_reset = 50;
  double cache_tolerance = 1e-12;
  bool caching = true;

  // Optimization status.
  bool optimal = false;
  bool reached_lower_bound = false;

  // Miscellaneous.
  vector<double> log_potentials;
  vector<double> x0;
  vector<double> x;
  bool recompute_everything = true;
  vector<double> maps_sum(variables_.size(), 0.0);
  int t;
  double dual_obj_best = 1e100, primal_rel_obj_best = -1e100;
  //double primal_obj_best = -1e100;
  int num_iterations_compute_dual = 50;

  // Compute extra score to account for variables that are not connected 
  // to any factor.
  // TODO: Precompute the value of these variables and eliminate them
  // from the pool.
  double extra_score = 0.0;
  for (int i = 0; i < variables_.size(); ++i) {
    BinaryVariable *variable = variables_[i];
    int variable_degree = variable->Degree();
    double log_potential = variable->GetLogPotential();
    if (variable_degree == 0 && log_potential > 0) {
      if (verbosity_ > 0) {
        cout << "Warning: variable " << i << " is not linked to any factor."
             << endl;
      }
      extra_score += log_potential;
    }
  }

  posteriors->resize(variables_.size(), 0.0);

  // Copy all additional log potentials to a vector and save room 
  // for the posteriors of additional variables.
  vector<double> additional_log_potentials;
  vector<int> additional_factor_offsets(factors_.size());
  CopyAdditionalLogPotentials(&additional_log_potentials,
                              &additional_factor_offsets);
  additional_posteriors->resize(additional_log_potentials.size(), 0.0);

  // Map indices of variables in factors.
  vector<int> indVinF(num_links_, -1);
  for (int j = 0; j < factors_.size(); ++j) {
    int factor_degree = factors_[j]->Degree();
    for (int l = 0; l < factor_degree; ++l) {
      indVinF[factors_[j]->GetLinkId(l)] = l;
    }
  }

  lambdas_.clear();
  lambdas_.resize(num_links_, 0.0);
  maps_.clear();
  maps_.resize(num_links_, 0.0);
  maps_av_.clear();
  maps_av_.resize(variables_.size(), 0.5);

  int num_times_increment = 0;
  double dual_obj_prev = 1e100;
  double dual_obj = extra_score;
  // Stores each factor contribution to the dual objective.
  vector<double> dual_obj_factors(factors_.size(), 0.0);

  for (t = 0; t < psdd_max_iterations_; ++t) {
    // Set stepsize.
    //double eta = psdd_eta_ / static_cast<double>(num_times_increment+1);
    double eta = psdd_eta_ / sqrt(static_cast<double>(t+1));

    // Initialize all variables as inactive.
    for (int i = 0; i < variables_.size(); ++i) {
      variable_is_active[i] = false;
    }

    // Optimize over maps_. (Compute dual value.)
    int num_inactive_factors = 0;
    for (int j = 0; j < factors_.size(); ++j) {
      // Skip inactive factors, but periodically update everything.
      // TODO: actually use num_iterations_reset somewhere
      if ((0 != (t % num_iterations_reset)) && 
          !recompute_everything && !factor_is_active[j]) {
        ++num_inactive_factors;
        continue;
      }

      Factor *factor = factors_[j];
      int factor_degree = factor->Degree();

      // When properly flagged, need to recompute everything.
      if (recompute_everything) {
        vector<double> *cached_log_potentials =
          factor->GetMutableCachedVariableLogPotentials();
        cached_log_potentials->resize(factor_degree);
        for (int i = 0; i < factor_degree; ++i) {
          int m = factor->GetLinkId(i);
          BinaryVariable* variable = factor->GetVariable(i);
          int variable_degree = variable->Degree();
          double val = variable->GetLogPotential() / 
            static_cast<double>(variable_degree)
            + 2.0 * lambdas_[m];
          (*cached_log_potentials)[i] = val;
        }
        factor->ComputeCachedAdditionalLogPotentials(1.0);
      }

      // Compute the MAP and update the dual objective.
      double val;
      factor->SolveMAPCached(&val);
      double delta = 0.0;
      for (int i = 0; i < factor_degree; ++i) {
        int m = factor->GetLinkId(i);
        delta -= lambdas_[m];
      }
      dual_obj += val + delta - dual_obj_factors[j];
      dual_obj_factors[j] = val + delta;

      // Check the variables that must be active.
      factor_is_active[j] = false;
      const vector<double> &variable_posteriors = factor->GetCachedVariablePosteriors();
      for (int i = 0; i < factor_degree; ++i) {
        int m = factor->GetLinkId(i);
        BinaryVariable* variable = factor->GetVariable(i);
        int k = variable->GetId();
        maps_sum[k] += variable_posteriors[i] - maps_[m];
        if (t == 0 || recompute_everything || !caching ||
            !NEARLY_BINARY(variable_posteriors[i], 1e-12) ||
            !NEARLY_EQ_TOL(variable_posteriors[i], maps_[m], cache_tolerance) ||
            !NEARLY_EQ_TOL(variable_posteriors[i], maps_av_[k], cache_tolerance)) {
          variable_is_active[k] = true;
        }
        maps_[m] = variable_posteriors[i];
      }

      // Save the additionals posteriors.
      const vector<double> &factor_additional_posteriors = factor->GetCachedAdditionalPosteriors();
      int offset = additional_factor_offsets[j];
      for (int i = 0; i < factor_additional_posteriors.size(); ++i) {
        (*additional_posteriors)[offset] = factor_additional_posteriors[i];
        ++offset;
      }
    }

    // Optimize over maps_av and update Lagrange multipliers.
    double primal_residual = 0.0;
    for (int i = 0; i < variables_.size(); ++i) {
      BinaryVariable *variable = variables_[i];
      int variable_degree = variable->Degree();

      if (!variable_is_active[i]) {
        // TODO: precompute values of these variables beforehand.
        if (variable_degree == 0) {
          maps_av_[i] = (variable->GetLogPotential() > 0)? 1.0 : 0.0;
        }
        // Make sure dual_residual = 0 and maps_av_[i] does not change.
        continue; 
      }

      if (variable_degree == 0) {
        maps_av_[i] = (variable->GetLogPotential() > 0)? 1.0 : 0.0;
      } else {
        maps_av_[i] = maps_sum[i] / static_cast<double>(variable_degree);
      }
      for (int j = 0; j < variable_degree; ++j) {
        int m = variable->GetLinkId(j);
        Factor* factor = variable->GetFactor(j);
        int k = factor->GetId();
        double diff_penalty = maps_[m] - maps_av_[i];
        int l = indVinF[m];
        vector<double> *cached_log_potentials =
          factor->GetMutableCachedVariableLogPotentials();
        (*cached_log_potentials)[l] -= 2.0 * eta * diff_penalty;
        lambdas_[m] -= eta * diff_penalty;

        // Mark factor as active.
        factor_is_active[k] = true;
        primal_residual += diff_penalty * diff_penalty;
      }
    }
    primal_residual = sqrt(primal_residual / lambdas_.size()); 

    // If primal residual is low enough or enough iterations 
    // have passed, compute the dual.
    bool compute_primal_rel = false;
    // TODO: && dual_residual < residual_threshold?
    if (primal_residual < residual_threshold) {
      compute_primal_rel = true;
    } else if (t > 0 && 0 == (t % num_iterations_compute_dual)) {
      compute_primal_rel = true;
    }

    // Check if dual improved so that num_times_increment 
    // can be incremented.
    if (dual_obj < dual_obj_prev) {
      ++num_times_increment;
    }
    dual_obj_prev = dual_obj;

    // Compute relaxed primal objective.
    double primal_rel_obj = -1e100;
    if (compute_primal_rel) {
      primal_rel_obj = 0.0;
      for (int i = 0; i < variables_.size(); ++i) {
        primal_rel_obj += maps_av_[i] * variables_[i]->GetLogPotential();
      }
      for (int i = 0; i < additional_log_potentials.size(); ++i) {
        primal_rel_obj += (*additional_posteriors)[i] * additional_log_potentials[i];
      }
    }

    if (dual_obj_best > dual_obj) {
      dual_obj_best = dual_obj;
      for (int i = 0; i < variables_.size(); ++i) {
        (*posteriors)[i] = maps_av_[i];
      }
      if (dual_obj_best < lower_bound) {
        reached_lower_bound = true;
        break;
      }
    }
    if (primal_rel_obj_best < primal_rel_obj) {
      primal_rel_obj_best = primal_rel_obj; 
    }
    if (compute_primal_rel) {
      gettimeofday(&end, NULL);
      if (verbosity_ > 1) {
        cout << "Iteration = " << t
             << "\tDual obj = " << dual_obj
             << "\tPrimal rel obj = " << primal_rel_obj
             << "\tPrimal residual = " << primal_residual
             << "\tBest dual obj = " << dual_obj_best
             << "\tBest primal rel obj = " << primal_rel_obj_best
             << "\tCached factors = " << 
          static_cast<double>(num_inactive_factors) /
          static_cast<double>(factors_.size())
             << "\teta = " << eta
             << "\tTime = " << ((double) diff_ms(end,start))/1000.0 << " sec."
             << endl; 
      }
    }
    //double gap = dual_obj_best - primal_rel_obj_best;

    // If both primal and dual residuals fall below a threshold,
    // we are done. TODO: also use gap?
    if (primal_residual < residual_threshold) {
      for (int i = 0; i < variables_.size(); ++i) {
        (*posteriors)[i] = maps_av_[i];
      }
      if (primal_residual < residual_threshold_final) {
        optimal = true;
        break;
      }
    }

    recompute_everything = false;
  }

  bool fractional = false;
  *value = 0.0;
  for (int i = 0; i < variables_.size(); ++i) {
    if (!NEARLY_BINARY((*posteriors)[i], 1e-12)) fractional = true;
    *value += (*posteriors)[i] * variables_[i]->GetLogPotential();
  }
  for (int i = 0; i < additional_log_potentials.size(); ++i) {
    *value += (*additional_posteriors)[i] * additional_log_potentials[i];
  }

  if (verbosity_ > 1) {
    cout << "Solution value after "
         << t << " iterations (Projected Subgradient) = "
         << *value << endl;
  }
  *upper_bound = dual_obj_best;

  gettimeofday(&end, NULL);
  if (verbosity_ > 1) {
    cout << "Took " << ((double) diff_ms(end,start))/1000.0 << " sec." << endl;
  }

  if (optimal) {
    if (!fractional) {
      if (verbosity_ > 1) {
        cout << "Solution is integer." << endl;
      }
      return STATUS_OPTIMAL_INTEGER;
    } else {
      if (verbosity_ > 1) {
        cout << "Solution is fractional." << endl;
      }
      return STATUS_OPTIMAL_FRACTIONAL;
    }
  } else {
    if (reached_lower_bound) {
      if (verbosity_ > 1) {
        cout << "Reached lower bound: " << lower_bound << "." << endl;
      }
      return STATUS_INFEASIBLE;
    } else {
      if (verbosity_ > 1) {
        cout << "Solution is only approximate." << endl;
      }
      return STATUS_UNSOLVED;
    }
  }
}

int FactorGraph::RunBranchAndBound(double cumulative_value,
                                   vector<bool> &branched_variables,
                                   int depth,
                                   vector<double>* posteriors,
                                   vector<double>* additional_posteriors,
                                   double *value,
                                   double *best_lower_bound,
                                   double *best_upper_bound) {
  int max_branching_depth = 5; // 2;

  // Solve the LP relaxation.
  int status = RunAD3(*best_lower_bound + cumulative_value,
                      posteriors, 
                      additional_posteriors,
                      value,
                      best_upper_bound);

  *value -= cumulative_value;
  *best_upper_bound -= cumulative_value;
  if (status == STATUS_OPTIMAL_INTEGER) {
    if (*value > *best_lower_bound) {
      *best_lower_bound = *value;
    }
    return status;
  } else if (status == STATUS_INFEASIBLE) {
    *value = -1e100;
    *best_upper_bound = -1e100;
    return status;
  }

  if (max_branching_depth >= 0 && depth > max_branching_depth) {
    *value = -1e100;
    *best_upper_bound = -1e100;
    return STATUS_UNSOLVED;
  }

  // Look for the most fractional component.
  int variable_to_branch = -1;
  double most_fractional_value = 1.0;
  for (int i = 0; i < variables_.size(); ++i) {
    if (branched_variables[i]) continue; // Already branched.
    double diff = (*posteriors)[i] - 0.5;
    diff *= diff;
    if (variable_to_branch < 0 || diff < most_fractional_value) {
      variable_to_branch = i;
      most_fractional_value = diff;
    }
  }
  branched_variables[variable_to_branch] = true;
  cout << "Branching on variable " << variable_to_branch
       << " at depth " << depth
       << " (value = " << (*posteriors)[variable_to_branch] << ")"
       << endl;

  double infinite_potential = 1000.0;
  double original_potential = variables_[variable_to_branch]->GetLogPotential();
  status = STATUS_OPTIMAL_INTEGER;

  // Zero branch.
  vector<double> posteriors_zero;
  vector<double> additional_posteriors_zero;
  double value_zero;
  double upper_bound_zero;
  double score = variables_[variable_to_branch]->GetLogPotential();
  variables_[variable_to_branch]->SetLogPotential(score - infinite_potential);
  int status_zero = RunBranchAndBound(cumulative_value,
                                      branched_variables,
                                      depth + 1,
                                      &posteriors_zero,
                                      &additional_posteriors_zero,
                                      &value_zero,
                                      best_lower_bound,
                                      best_upper_bound);
  // Put back the original potential.
  variables_[variable_to_branch]->SetLogPotential(original_potential);
  if (status_zero != STATUS_OPTIMAL_INTEGER &&
      status_zero != STATUS_INFEASIBLE) {
    status = STATUS_UNSOLVED;
    //return STATUS_UNSOLVED;
  }

  // One branch.
  vector<double> posteriors_one;
  vector<double> additional_posteriors_one;
  double value_one;
  double upper_bound_one;
  score = variables_[variable_to_branch]->GetLogPotential();
  variables_[variable_to_branch]->SetLogPotential(score + infinite_potential);
  int status_one = RunBranchAndBound(cumulative_value + infinite_potential,
                                     branched_variables,
                                     depth + 1,
                                     &posteriors_one,
                                     &additional_posteriors_one,
                                     &value_one,
                                     best_lower_bound,
                                     best_upper_bound);
  // Put back the original potential.
  variables_[variable_to_branch]->SetLogPotential(original_potential);
  if (status_one != STATUS_OPTIMAL_INTEGER &&
      status_one != STATUS_INFEASIBLE) {
    status = STATUS_UNSOLVED;
    //return STATUS_UNSOLVED;
  }

  if (status_zero == STATUS_INFEASIBLE &&
      status_one == STATUS_INFEASIBLE) {
    *value = -1e100;
    return STATUS_INFEASIBLE;
  }

  if (value_zero >= value_one) {
    *value = value_zero;
    *posteriors = posteriors_zero;
    *additional_posteriors = additional_posteriors_zero;
  } else {
    *value = value_one;
    *posteriors = posteriors_one;
    *additional_posteriors = additional_posteriors_one;
  }

  //return STATUS_OPTIMAL_INTEGER;
  return status;
}

int FactorGraph::RunAD3(double lower_bound,
                        vector<double> *posteriors, 
                        vector<double> *additional_posteriors,
                        double *value,
                        double *upper_bound) {
  timeval start, end;
  gettimeofday(&start, NULL);

  // Stopping criterion parameters.
  double residual_threshold = ad3_residual_threshold_; // 1e-6;
  //double gap_threshold = 1e-6;

  // Stepsize adjustment parameters.
  double max_eta = 100.0;
  double min_eta = 1e-3;
  double gamma_primal = 100.0; // 10.0
  double gamma_dual = 10.0;
  double factor_step = 2.0;
  double tau = 1.0;
  int num_iterations_adapt_eta = 10; // 1

  // Caching parameters.
  vector<bool> factor_is_active(factors_.size(), true);
  vector<bool> variable_is_active(variables_.size(), false);
  int num_iterations_reset = 50;
  double cache_tolerance = 1e-12;
  bool caching = true; // true

  // Optimization status.
  bool optimal = false;
  bool reached_lower_bound = false;

  // Miscellaneous.
  vector<double> log_potentials;
  vector<double> factor_variable_posteriors;
  vector<double> factor_additional_posteriors;
  bool eta_changed = true;
  vector<double> maps_sum(variables_.size(), 0.0);
  int t;
  double dual_obj_best = 1e100, primal_rel_obj_best = -1e100;
  double primal_obj_best = -1e100;
  int num_iterations_compute_dual = 50;

  // Compute extra score to account for variables that are not connected 
  // to any factor.
  // TODO: Precompute the value of these variables and eliminate them
  // from the pool.
  double extra_score = 0.0;
  for (int i = 0; i < variables_.size(); ++i) {
    BinaryVariable *variable = variables_[i];
    int variable_degree = variable->Degree();
    double log_potential = variable->GetLogPotential();
    if (variable_degree == 0 && log_potential > 0) {
      if (verbosity_ > 0) {
        cout << "Warning: variable " << i << " is not linked to any factor."
             << endl;
      }
      extra_score += log_potential;
    }
  }

  posteriors->resize(variables_.size(), 0.0);

  // Copy all additional log potentials to a vector and save room 
  // for the posteriors of additional variables.
  vector<double> additional_log_potentials;
  vector<int> additional_factor_offsets(factors_.size());
  CopyAdditionalLogPotentials(&additional_log_potentials,
                              &additional_factor_offsets);
  additional_posteriors->resize(additional_log_potentials.size(), 0.0);

  // Map indices of variables in factors.
  vector<int> indVinF(num_links_, -1);
  for (int j = 0; j < factors_.size(); ++j) {
    int factor_degree = factors_[j]->Degree();
    for (int l = 0; l < factor_degree; ++l) {
      indVinF[factors_[j]->GetLinkId(l)] = l;
    }
  }

  lambdas_.clear();
  lambdas_.resize(num_links_, 0.0);
  maps_.clear();
  maps_.resize(num_links_, 0.0);
  maps_av_.clear();
  maps_av_.resize(variables_.size(), 0.5);

  double eta = ad3_eta_;
  for (t = 0; t < ad3_max_iterations_; ++t) {
    int num_inactive_factors = 0;

    // Initialize all variables as inactive.
    for (int i = 0; i < variables_.size(); ++i) {
      variable_is_active[i] = false;
    }

    // Optimize over maps_.
    for (int j = 0; j < factors_.size(); ++j) {
      // Skip inactive factors, but periodically update everything.
      // TODO: actually use num_iterations_reset somewhere
      if ((0 != (t % num_iterations_reset)) && 
          !eta_changed && !factor_is_active[j]) {
        ++num_inactive_factors;
        continue;
      }

      Factor *factor = factors_[j];
      int factor_degree = factor->Degree();

      // If stepsize has changed, need to recompute everything.
      if (eta_changed) {
        vector<double> *cached_log_potentials =
          factor->GetMutableCachedVariableLogPotentials();
        cached_log_potentials->resize(factor_degree);
        for (int i = 0; i < factor_degree; ++i) {
          int m = factor->GetLinkId(i);
          BinaryVariable* variable = factor->GetVariable(i);
          int k = variable->GetId();
          int variable_degree = variable->Degree();
          double val = variable->GetLogPotential() / 
            static_cast<double>(variable_degree)
            + 2.0 * lambdas_[m];
          (*cached_log_potentials)[i] = maps_av_[k] + val / (2.0 * eta);
        }
        factor->ComputeCachedAdditionalLogPotentials(2.0 * eta);
      }

      // Solve the QP.
      factor->SolveQPCached();

      // Check the variables that must be active.
      factor_is_active[j] = false;
      const vector<double> &variable_posteriors = factor->GetCachedVariablePosteriors();
      for (int i = 0; i < factor_degree; ++i) {
        int m = factor->GetLinkId(i);
        BinaryVariable* variable = factor->GetVariable(i);
        int k = variable->GetId();
        maps_sum[k] += variable_posteriors[i] - maps_[m];
        if (t == 0 || eta_changed || !caching ||
            !NEARLY_BINARY(variable_posteriors[i], 1e-12) ||
            !NEARLY_EQ_TOL(variable_posteriors[i], maps_[m], cache_tolerance) ||
            !NEARLY_EQ_TOL(variable_posteriors[i], maps_av_[k], cache_tolerance)) {
          variable_is_active[k] = true;
        }
        maps_[m] = variable_posteriors[i];
      }

      // Save the additionals posteriors.
      const vector<double> &factor_additional_posteriors = factor->GetCachedAdditionalPosteriors();
      int offset = additional_factor_offsets[j];
      for (int i = 0; i < factor_additional_posteriors.size(); ++i) {
        (*additional_posteriors)[offset] = factor_additional_posteriors[i];
        ++offset;
      }
    }

    // Optimize over maps_av and update Lagrange multipliers.
    double primal_residual = 0.0;
    double dual_residual = 0.0;
    for (int i = 0; i < variables_.size(); ++i) {
      BinaryVariable *variable = variables_[i];
      int variable_degree = variable->Degree();

      if (!variable_is_active[i]) {
        // TODO: precompute values of these variables beforehand.
        if (variable_degree == 0) {
          maps_av_[i] = (variable->GetLogPotential() > 0)? 1.0 : 0.0;
        }
        // Make sure dual_residual = 0 and maps_av_[i] does not change.
        continue; 
      }

      double map_av_prev = maps_av_[i];
      if (variable_degree == 0) {
        maps_av_[i] = (variable->GetLogPotential() > 0)? 1.0 : 0.0;
      } else {
        maps_av_[i] = maps_sum[i] / static_cast<double>(variable_degree);
      }
      double diff = maps_av_[i] - map_av_prev;
      dual_residual += variable_degree * diff * diff;
      for (int j = 0; j < variable_degree; ++j) {
        int m = variable->GetLinkId(j);
        Factor* factor = variable->GetFactor(j);
        int k = factor->GetId();
        double diff_penalty = maps_[m] - maps_av_[i];
        int l = indVinF[m];
        vector<double> *cached_log_potentials =
          factor->GetMutableCachedVariableLogPotentials();
        (*cached_log_potentials)[l] += diff - tau * diff_penalty;
        lambdas_[m] -= tau * eta * diff_penalty;

        // Mark factor as active.
        factor_is_active[k] = true;
        primal_residual += diff_penalty * diff_penalty;
      }
    }
    primal_residual = sqrt(primal_residual / lambdas_.size()); 
    dual_residual = sqrt(dual_residual / lambdas_.size());

    // If primal residual is low enough or enough iterations 
    // have passed, compute the dual.
    bool compute_dual = false;
    bool compute_primal_rel = false;
    // TODO: && dual_residual < residual_threshold?
    if (primal_residual < residual_threshold) {
      compute_dual = true;
      compute_primal_rel = true;
    } else if (t > 0 && 0 == (t % num_iterations_compute_dual)) {
      compute_dual = true;
    }

    // Compute dual value.
    // TODO: make this a function of its own?
    double dual_obj = 1e100;
    if (compute_dual) {
      dual_obj = 0.0;
      for (int j = 0; j < factors_.size(); ++j) {
        Factor *factor = factors_[j];
        int factor_degree = factor->Degree();
        log_potentials.resize(factor_degree);
        factor_variable_posteriors.resize(factor_degree);
        int num_additional = factor->GetAdditionalLogPotentials().size();
        factor_additional_posteriors.resize(num_additional);
        double delta = 0.0;
        for (int i = 0; i < factor_degree; ++i) {
          int m = factor->GetLinkId(i);
          BinaryVariable *variable = factor->GetVariable(i);
          int variable_degree = variable->Degree();
          log_potentials[i] = variable->GetLogPotential() / 
            static_cast<double>(variable_degree)
            + 2.0 * lambdas_[m];
          delta -= lambdas_[m];
        }
        double val;
        factor->SolveMAP(log_potentials,
                         factor->GetAdditionalLogPotentials(),
                         &factor_variable_posteriors,
                         &factor_additional_posteriors,
                         &val); 
        dual_obj += val + delta;
      }
      dual_obj += extra_score;
    }

    // Compute relaxed primal objective.
    double primal_rel_obj = -1e100;
    if (compute_primal_rel) {
      primal_rel_obj = 0.0;
      for (int i = 0; i < variables_.size(); ++i) {
        primal_rel_obj += maps_av_[i] * variables_[i]->GetLogPotential();
      }
      for (int i = 0; i < additional_log_potentials.size(); ++i) {
        primal_rel_obj += (*additional_posteriors)[i] * additional_log_potentials[i];
      }
    }

    // Compute primal objective.
    double primal_obj = -1e100;
    bool compute_primal = false;
    if (compute_primal) {
      //TODO
      if (primal_obj > primal_obj_best) {
        primal_obj_best = primal_obj;
      }
    }

    if (dual_obj_best > dual_obj) {
      dual_obj_best = dual_obj;
      for (int i = 0; i < variables_.size(); ++i) {
        (*posteriors)[i] = maps_av_[i];
      }
      if (dual_obj_best < lower_bound) {
        reached_lower_bound = true;
        break;
      }
    }
    if (primal_rel_obj_best < primal_rel_obj) {
      primal_rel_obj_best = primal_rel_obj; 
    }
    if (compute_dual) {
      gettimeofday(&end, NULL);
      if (verbosity_ > 1) { 
        cout << "Iteration = " << t
             << "\tDual obj = " << dual_obj
             << "\tPrimal rel obj = " << primal_rel_obj
             << "\tPrimal obj = " << primal_obj
             << "\tDual residual = " << dual_residual
             << "\tPrimal residual = " << primal_residual
             << "\tBest dual obj = " << dual_obj_best
             << "\tBest primal rel obj = " << primal_rel_obj_best
             << "\tBest primal obj = " << primal_obj_best
             << "\tCached factors = " << 
          static_cast<double>(num_inactive_factors) /
          static_cast<double>(factors_.size())
             << "\teta = " << eta
             << "\tChanged eta = " << (eta_changed? "true" : "false") 
             << "\tTime = " << ((double) diff_ms(end,start))/1000.0 << " sec."
             << endl; 
      }
    }
    //double gap = dual_obj_best - primal_rel_obj_best;

    // If both primal and dual residuals fall below a threshold,
    // we are done. TODO: also use gap?
    if (dual_residual < residual_threshold && 
        primal_residual < residual_threshold) {
      for (int i = 0; i < variables_.size(); ++i) {
        (*posteriors)[i] = maps_av_[i];
      }
      optimal = true;
      break;
    }

    // Adjust the stepsize if residuals are very asymmetric.
    eta_changed = false;
    if (ad3_adapt_eta_ && 0 == (t % num_iterations_adapt_eta)) {
      if (primal_residual > gamma_primal * dual_residual) {
        if (eta < max_eta) {
          eta *= factor_step;
          eta_changed = true;
        }
      } else if (dual_residual > gamma_dual * primal_residual) {
        if (eta > min_eta) {
          eta /= factor_step;
          eta_changed = true;
        }
      }
    }
  }

  bool fractional = false;
  *value = 0.0;
  for (int i = 0; i < variables_.size(); ++i) {
    if (!NEARLY_BINARY((*posteriors)[i], 1e-12)) fractional = true;
    *value += variables_[i]->GetLogPotential() * (*posteriors)[i];
  }
  for (int i = 0; i < additional_log_potentials.size(); ++i) {
    *value += additional_log_potentials[i] * (*additional_posteriors)[i];
  } 

  if (verbosity_ > 1) {
    cout << "Solution value after "
         << t << " iterations (AD3) = "
         << *value << endl;
  }
  *upper_bound = dual_obj_best;

  gettimeofday(&end, NULL);
  if (verbosity_ > 1) {
    cout << "Took " << ((double) diff_ms(end,start))/1000.0 << " sec." << endl;
  }

  if (optimal) {
    if (!fractional) {
      if (verbosity_ > 1) {
        cout << "Solution is integer." << endl;
      }
      return STATUS_OPTIMAL_INTEGER;
    } else {
      if (verbosity_ > 1) {
        cout << "Solution is fractional." << endl;
      }
      return STATUS_OPTIMAL_FRACTIONAL;
    }
  } else {
    if (reached_lower_bound) {
      if (verbosity_ > 1) {
        cout << "Reached lower bound: " << lower_bound << "." << endl;
      }
      return STATUS_INFEASIBLE;
    } else {
      if (verbosity_ > 1) {
        cout << "Solution is only approximate." << endl;
      }
      return STATUS_UNSOLVED;
    }
  }
}


#if 0
int main(int argc, char **argv) {
  FactorGraph graph;
  MultiVariable *v1 = graph.CreateMultiVariable(5);
  MultiVariable *v2 = graph.CreateMultiVariable(4);
  vector<MultiVariable*> variables;
  variables.push_back(v1);
  variables.push_back(v2);
  Factor *f = graph.CreateFactorMultiDense(variables);
  v1->SetLogPotential(0, 0.1);
  v1->SetLogPotential(1, 0.2);
  v1->SetLogPotential(2, 0.3);
  v1->SetLogPotential(3, 0.4);
  v1->SetLogPotential(4, 0.5);
  v2->SetLogPotential(0, -0.1);
  v2->SetLogPotential(1, -0.2);
  v2->SetLogPotential(2, -0.3);
  v2->SetLogPotential(3, -0.4);
  vector<int> values(2);
  values[0] = 2;
  values[1] = 3;
  static_cast<FactorMultiDense*>(f)->SetFactorLogPotential(values, -3.1415);

  FactorGraph binary_graph;
  BinaryVariable *u1 = binary_graph.CreateBinaryVariable();
  BinaryVariable *u2 = binary_graph.CreateBinaryVariable();
  u1->SetLogPotential(-1.5);
  u2->SetLogPotential(0.5);
  vector<BinaryVariable*> binary_variables;
  vector<bool> negated;
  binary_variables.push_back(u1);
  binary_variables.push_back(u2);
  negated.push_back(true);
  negated.push_back(false);
  Factor *fb = binary_graph.CreateFactorOR(binary_variables, negated);

  vector<double> posteriors;
  double value;
  binary_graph.ComputeLPMAPWithAD3(&posteriors, &value);
  cout << "Solution value = " << value << endl;
  return 0;
}
#endif

} // namespace AD3
