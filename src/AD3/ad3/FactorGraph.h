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
#include "Factor.h"
#include "GenericFactor.h"
#include "FactorDense.h"

namespace AD3 {

enum OptimizationStatus {
  STATUS_OPTIMAL_INTEGER = 0,
  STATUS_OPTIMAL_FRACTIONAL,
  STATUS_INFEASIBLE,
  STATUS_UNSOLVED
};

class FactorGraph {
 public:
  FactorGraph() {
    verbosity_ = 2;
    num_links_ = 0;
    ResetParametersAD3();
    ResetParametersPSDD();
  }
  ~FactorGraph() {
    for (int i = 0; i < variables_.size(); ++i) {
      delete variables_[i];
    }
    for (int i = 0; i < multi_variables_.size(); ++i) {
      delete multi_variables_[i];
    }
    for (int i = 0; i < factors_.size(); ++i) {
      if (owned_factors_[i]) delete factors_[i];
    }
  }

  // Set verbosity level.
  void SetVerbosity(int verbosity) { verbosity_ = verbosity; }

  // Create a new state (binary variable).
  BinaryVariable *CreateBinaryVariable() {
    BinaryVariable *variable = new BinaryVariable;
    variable->SetId(variables_.size());
    variables_.push_back(variable);
    return variable;
  }

  // Create a new multi-valued variable.
  MultiVariable *CreateMultiVariable(int num_states) {
    MultiVariable *multi = new MultiVariable;
    multi->SetId(multi_variables_.size());
    multi_variables_.push_back(multi);
    vector<BinaryVariable*> states(num_states);
    for (int i = 0; i < num_states; ++i) {
      BinaryVariable *state = CreateBinaryVariable();
      states[i] = state;
    }
    multi->Initialize(states);
    return multi;
  }

  // Create a new multi-valued variable with existing states.
  MultiVariable *CreateMultiVariable(
      const vector<BinaryVariable*> &states) {
    MultiVariable *multi = new MultiVariable;
    multi->SetId(multi_variables_.size());
    multi_variables_.push_back(multi);
    multi->Initialize(states);
    return multi;
  }

  // Declare a factor. 
  // By default, the factor is NOT owned by the factor graph.
  void DeclareFactor(Factor *factor, const vector<BinaryVariable*> &variables,
                     bool owned_by_graph = false) {
    vector<bool> negated;
    DeclareFactor(factor, variables, negated, owned_by_graph);
  }
  void DeclareFactor(Factor *factor, const vector<BinaryVariable*> &variables,
                     const vector<bool> &negated,
                     bool owned_by_graph = false) {
    if (factor->IsGeneric()) {
      static_cast<GenericFactor*>(factor)->SetVerbosity(verbosity_);
    }
    factor->SetId(factors_.size());
    factor->Initialize(variables, negated, &num_links_);
    factors_.push_back(factor);
    owned_factors_.push_back(owned_by_graph);
  }

  // Create a new XOR factor.
  // By default, the factor will be owned by the factor graph.
  Factor *CreateFactorXOR(const vector<BinaryVariable*> &variables,
                          bool owned_by_graph = true) {
    vector<bool> negated;
    return CreateFactorXOR(variables, negated, owned_by_graph);
  }
  Factor *CreateFactorXOR(const vector<BinaryVariable*> &variables,
                          const vector<bool> &negated,
                          bool owned_by_graph = true) {
    Factor *factor = new FactorXOR;
    DeclareFactor(factor, variables, negated, owned_by_graph);
    //assert(variables.size() > 1);
    return factor;
  }

  // Create a new XOR-with-output factor. 
  // This is a XOR whose last variable is negated.
  Factor *CreateFactorXOROUT(const vector<BinaryVariable*> &variables,
                             bool owned_by_graph = true) {
    vector<bool> negated;
    return CreateFactorXOROUT(variables, negated, owned_by_graph);
  }
  Factor *CreateFactorXOROUT(const vector<BinaryVariable*> &variables,
                             const vector<bool> &negated,
                             bool owned_by_graph = true) {
    Factor *factor = new FactorXOR;
    vector<bool> negated_copy = negated;
    if (negated_copy.size() == 0) {
      negated_copy.resize(variables.size(), false);
    }
    negated_copy[variables.size() - 1] = !negated_copy[variables.size() - 1];
    DeclareFactor(factor, variables, negated_copy, owned_by_graph);
    assert(variables.size() > 1);
    return factor;
  }

  // Create a new AtMostOne factor.
  Factor *CreateFactorAtMostOne(const vector<BinaryVariable*> &variables,
                          bool owned_by_graph = true) {
    vector<bool> negated;
    return CreateFactorAtMostOne(variables, negated, owned_by_graph);
  }
  Factor *CreateFactorAtMostOne(const vector<BinaryVariable*> &variables,
                                const vector<bool> &negated,
                                bool owned_by_graph = true) {
    Factor *factor = new FactorAtMostOne;
    DeclareFactor(factor, variables, negated, owned_by_graph);
    assert(variables.size() > 1);
    return factor;
  }

  // Create a new OR factor.
  Factor *CreateFactorOR(const vector<BinaryVariable*> &variables,
                          bool owned_by_graph = true) {
    vector<bool> negated;
    return CreateFactorOR(variables, negated, owned_by_graph);
  }
  Factor *CreateFactorOR(const vector<BinaryVariable*> &variables,
                         const vector<bool> &negated,
                         bool owned_by_graph = true) {
    Factor *factor = new FactorOR;
    DeclareFactor(factor, variables, negated, owned_by_graph);
    assert(variables.size() > 1);
    return factor;
  }

  // Create a new OR-with-output factor.
  Factor *CreateFactorOROUT(const vector<BinaryVariable*> &variables,
                            bool owned_by_graph = true) {
    vector<bool> negated;
    return CreateFactorOROUT(variables, negated, owned_by_graph);
  }
  Factor *CreateFactorOROUT(const vector<BinaryVariable*> &variables,
                            const vector<bool> &negated,
                            bool owned_by_graph = true) {
    Factor *factor = new FactorOROUT;
    DeclareFactor(factor, variables, negated, owned_by_graph);
    assert(variables.size() > 2);
    return factor;
  }

  // Create a new AND-with-output factor. 
  // This is a OROUT whose all variables negated.
  Factor *CreateFactorANDOUT(const vector<BinaryVariable*> &variables,
                             bool owned_by_graph = true) {
    vector<bool> negated;
    return CreateFactorANDOUT(variables, negated, owned_by_graph);
  }
  Factor *CreateFactorANDOUT(const vector<BinaryVariable*> &variables,
                             const vector<bool> &negated,
                             bool owned_by_graph = true) {
    Factor *factor = new FactorOROUT;
    vector<bool> negated_copy = negated;
    if (negated_copy.size() == 0) {
      negated_copy.resize(variables.size(), false);
    }
    for (int i = 0; i < negated_copy.size(); ++i) {
      negated_copy[i] = !negated_copy[i];
    }
    DeclareFactor(factor, variables, negated_copy, owned_by_graph);
    assert(variables.size() > 2);
    return factor;
  }

  // Create a new IMPLY factor. 
  // This is a OR whose first K-1 variables are negated.
  // It imposes (A_1 ^ A_2 ^ ... ^ A_{K-1} => A_K).
  Factor *CreateFactorIMPLY(const vector<BinaryVariable*> &variables,
                             bool owned_by_graph = true) {
    vector<bool> negated;
    return CreateFactorIMPLY(variables, negated, owned_by_graph);
  }
  Factor *CreateFactorIMPLY(const vector<BinaryVariable*> &variables,
                            const vector<bool> &negated,
                            bool owned_by_graph = true) {
    Factor *factor = new FactorOR;
    vector<bool> negated_copy = negated;
    if (negated_copy.size() == 0) {
      negated_copy.resize(variables.size(), false);
    }
    for (int i = 0; i < negated_copy.size() - 1; ++i) {
      negated_copy[i] = !negated_copy[i];
    }
    DeclareFactor(factor, variables, negated_copy, owned_by_graph);
    assert(variables.size() > 1);
    return factor;
  }

  // Create a new PAIR factor. 
  // All edge log-potentials are assumed to be zero, except for the
  // configuration where both inputs are 1, which receives the value
  // in edge_log_potential.
  // REMARK: for efficiency, cannot have negated variables.
  Factor *CreateFactorPAIR(const vector<BinaryVariable*> &variables,
                           double edge_log_potential,
                           bool owned_by_graph = true) {
    Factor *factor = new FactorPAIR;
    vector<bool> negated;
    DeclareFactor(factor, variables, negated, owned_by_graph);
    vector<double> additional_log_potentials(1, edge_log_potential);
    factor->SetAdditionalLogPotentials(additional_log_potentials);
    return factor;
  }

  // Create a new dense factor. 
  // All additional log-potentials are assumed to be in the following order:
  // scores[0,0,...,0], scores[0,0,...,1], etc.
  Factor *CreateFactorDense(const vector<MultiVariable*> &multi_variables,
                            const vector<double> &additional_log_potentials,
                            bool owned_by_graph = true) {
    Factor *factor = new FactorDense;
    vector<BinaryVariable*> variables;
    for (int i = 0; i < multi_variables.size(); ++i) {
      variables.insert(variables.end(),
                       multi_variables[i]->GetStates().begin(),
                       multi_variables[i]->GetStates().end());
    }
    vector<bool> negated;
    DeclareFactor(factor, variables, negated, owned_by_graph);
    static_cast<FactorDense*>(factor)->Initialize(multi_variables);
    factor->SetAdditionalLogPotentials(additional_log_potentials);
    return factor;
  }

  // Count variables/factors.
  int GetNumVariables() { return variables_.size(); }
  int GetNumFactors() { return factors_.size(); }

  // Get variables/factors.
  BinaryVariable *GetBinaryVariable(int i) { return variables_[i]; }
  Factor *GetFactor(int i) { return factors_[i]; }
  
  // Check if there is any multi-variable which does not 
  // belong to any factor, and if so, assign a XOR factor
  // to the corresponding binary variables.
  void FixMultiVariablesWithoutFactors();
  
  // Convert a factor graph with multi-valued variables to one which only
  // contains binary variables and hard constraint factors.
  void ConvertToBinaryFactorGraph(FactorGraph *binary_factor_graph);

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
  int AddEvidence(vector<int> *evidence,
                  vector<int> *recomputed_indices);

  // Print factor graph as a string.
  void Print(ostream& stream) {
    stream << GetNumVariables() << endl;
    stream << GetNumFactors() << endl;
    for (int i = 0; i < GetNumVariables(); ++i) {
      stream << setprecision(9) << variables_[i]->GetLogPotential() << endl;
    }
    for (int i = 0; i < GetNumFactors(); ++i) {
      factors_[i]->Print(stream);
    }
  }

  // Set options of AD3/PSDD algorithms.
  void SetMaxIterationsAD3(int max_iterations) { 
    ad3_max_iterations_ = max_iterations;
  }
  void SetEtaAD3(double eta) { ad3_eta_ = eta; }
  void AdaptEtaAD3(bool adapt) { ad3_adapt_eta_ = adapt; }
  void SetResidualThresholdAD3(double threshold) { 
    ad3_residual_threshold_ = threshold; 
  }
  void SetMaxIterationsPSDD(int max_iterations) { 
    psdd_max_iterations_ = max_iterations;
  }
  void SetEtaPSDD(double eta) { psdd_eta_ = eta; }

  int SolveLPMAPWithAD3(vector<double> *posteriors,
                        vector<double> *additional_posteriors, 
                        double *value) {
    double upper_bound;
    return RunAD3(-1e100, posteriors, additional_posteriors, value, &upper_bound);
  }
  
  int SolveExactMAPWithAD3(vector<double> *posteriors,
                           vector<double> *additional_posteriors, 
                           double *value) {
    double best_lower_bound = -1e100;
    double upper_bound;
    vector<bool> branched_variables(variables_.size(), false);
    int depth = 0;
    int status = RunBranchAndBound(0.0, 
				                           branched_variables,
				                           depth,
				                           posteriors,
				                           additional_posteriors,
				                           value,
				                           &best_lower_bound,
				                           &upper_bound);
	  if (verbosity_ > 1) {
      cout << "Solution value for AD3 ILP: " << *value << endl;
    }
    return status;
  }
  
  int SolveLPMAPWithPSDD(vector<double> *posteriors,
                         vector<double> *additional_posteriors, 
                         double *value) {
    // Add code here for tuning the stepsize.
    double upper_bound;
    return RunPSDD(-1e100, posteriors, additional_posteriors, value, &upper_bound);
  }

 private:
  void ResetParametersAD3() {
    ad3_eta_ = 0.1;
    ad3_adapt_eta_ = true;
    ad3_max_iterations_ = 1000;
    ad3_residual_threshold_ = 1e-6;
  }

  void ResetParametersPSDD() {
    psdd_eta_ = 1.0;
    psdd_max_iterations_ = 1000;
  }

  void CopyAdditionalLogPotentials(vector<double>* additional_log_potentials,
                                   vector<int>* factor_indices);

  int RunPSDD(double lower_bound,
              vector<double> *posteriors, 
              vector<double> *additional_posteriors, 
              double *value,
              double *upper_bound);

  int RunAD3(double lower_bound,
             vector<double> *posteriors, 
             vector<double> *additional_posteriors, 
             double *value,
             double *upper_bound);
             
  int RunBranchAndBound(double cumulative_value,
                        vector<bool> &branched_variables,
                        int depth,
                        vector<double>* posteriors,
                        vector<double>* additional_posteriors,
                        double *value,
                        double *best_lower_bound,
                        double *best_upper_bound);
             
 private:
  vector<BinaryVariable*> variables_;
  vector<MultiVariable*> multi_variables_;
  vector<Factor*> factors_;
  vector<bool> owned_factors_;
  int num_links_;

  // Verbosity level. 0 only displays error/warning messages, 
  // 1 displays info messages, >1 displays additional info.
  int verbosity_;

  // Parameters for AD3:
  int ad3_max_iterations_; // Maximum number of iterations.
  double ad3_eta_; // Initial penalty parameter of the augmented Lagrangian.
  // If true, eta_ is adjusted automatically as described in 
  // Boyd et al. (2011).
  bool ad3_adapt_eta_; 
  // Threshold for primal/dual residuals.
  double ad3_residual_threshold_;

  // Parameters for PSDD:
  int psdd_max_iterations_; // Maximum number of iterations.
  double psdd_eta_; // Initial stepsize.

  // Parameters for AD3 and PSDD:
  vector<double> lambdas_;
  vector<double> maps_;
  vector<double> maps_av_;
};

} // namespace AD3
