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

#ifndef GENERIC_FACTOR_H_
#define GENERIC_FACTOR_H_

#include "Factor.h"

namespace AD3 {

// This must be implemented by the user-defined factor.
typedef void *Configuration;

// Base class for a generic factor.
// Specialized factors should be derived from this class.
class GenericFactor : public Factor {
 public:
  GenericFactor() {
    verbosity_ = 2;
    num_max_iterations_QP_ = 10;
  };
  virtual ~GenericFactor() {};

  virtual int type() { return FactorTypes::FACTOR_GENERIC; }
  bool IsGeneric() { return true; }
  void SetVerbosity(int verbosity) { verbosity_ = verbosity; }

 protected:
  // Compute posterior marginals from a sparse distribution, 
  // expressed as a set of configurations (active_set) and 
  // a probability/weight for each configuration (stored in
  // vector distribution).
  void ComputeMarginalsFromSparseDistribution(
    const vector<Configuration> &active_set,
    const vector<double> &distribution,
    vector<double> *variable_posteriors,
    vector<double> *additional_posteriors) {
    variable_posteriors->assign(binary_variables_.size(), 0.0);
    additional_posteriors->assign(additional_log_potentials_.size(), 0.0);
    for (int i = 0; i < active_set.size(); ++i) {
      UpdateMarginalsFromConfiguration(active_set[i],
                                       distribution[i],
                                       variable_posteriors,
                                       additional_posteriors);
    }
  }

  bool InvertAfterInsertion(const vector<Configuration> &active_set,
                            const Configuration &inserted_element);

  void InvertAfterRemoval(const vector<Configuration> &active_set,
                          int removed_index);

  void ComputeActiveSetSimilarities(const vector<Configuration> &active_set,
						            vector<double> *similarities);

  void EigenDecompose(vector<double> *similarities,
                      vector<double> *eigenvalues);

  void Invert(const vector<double> &eigenvalues,
              const vector<double> &eigenvectors);

  bool IsSingular(vector<double> &eigenvalues,
                  vector<double> &eigenvectors,
                  vector<double> *null_space_basis);

 protected:
  // Compute the score of a given assignment.
  // This must be implemented in the user-defined factor.
  virtual void Evaluate(const vector<double> &variable_log_potentials,
                        const vector<double> &additional_log_potentials,
                        const Configuration configuration,
                        double *value) = 0;

  // Find the most likely assignment.
  // This must be implemented in the user-defined factor.
  virtual void Maximize(const vector<double> &variable_log_potentials,
                        const vector<double> &additional_log_potentials,
                        Configuration &configuration,
                        double *value) = 0;

  // Given a configuration with a probability (weight), 
  // increment the vectors of variable and additional posteriors.
  virtual void UpdateMarginalsFromConfiguration(
    const Configuration &configuration,
    double weight,
    vector<double> *variable_posteriors,
    vector<double> *additional_posteriors) = 0;
    
  // Count how many common values two configurations have.
  virtual int CountCommonValues(
    const Configuration &configuration1,
    const Configuration &configuration2) = 0;

  // Check if two configurations are the same.
  virtual bool SameConfiguration(
    const Configuration &configuration1,
    const Configuration &configuration2) = 0;

  // Create configuration.
  virtual Configuration CreateConfiguration() = 0;

  // Delete configuration.
  virtual void DeleteConfiguration(
    Configuration configuration) = 0;

 public:
  // Compute the MAP (local subproblem in the projected subgradient algorithm).
  // The user-defined factor may override this.
  virtual void SolveMAP(const vector<double> &variable_log_potentials,
                        const vector<double> &additional_log_potentials,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors,
                        double *value) {
    Configuration configuration = CreateConfiguration();
    Maximize(variable_log_potentials,
             additional_log_potentials,
             configuration,
             value);
    variable_posteriors->assign(binary_variables_.size(), 0.0);
    additional_posteriors->assign(additional_log_potentials_.size(), 0.0);
    UpdateMarginalsFromConfiguration(configuration,
                                     1.0,
                                     variable_posteriors,
                                     additional_posteriors);
    DeleteConfiguration(configuration);
  }

  // Solve the QP (local subproblem in the AD3 algorithm).
  // By default, used the active set method. 
  // The user-defined factor may override this.
  virtual void SolveQP(const vector<double> &variable_log_potentials,
                       const vector<double> &additional_log_potentials,
                       vector<double> *variable_posteriors,
                       vector<double> *additional_posteriors);

 protected:
  vector<Configuration> active_set_;
  vector<double> distribution_;
  vector<double> inverse_A_;
  int num_max_iterations_QP_; // Initialize to 10.
  int verbosity_; // Verbosity level.
};

} // namespace AD3

#endif // GENERIC_FACTOR_H_
