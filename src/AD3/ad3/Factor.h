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

#ifndef FACTOR_H_
#define FACTOR_H_

#include <vector>
#include <iostream>
#include <iomanip>
#include <assert.h>

using namespace std;

namespace AD3 {

struct FactorTypes {
  enum {
    FACTOR_GENERIC = 0,
    FACTOR_PAIR,
    FACTOR_XOR,
    FACTOR_OR,
    FACTOR_OROUT,
    FACTOR_ATMOSTONE,
    FACTOR_MULTI_DENSE
  };
};

// A binary variable.
class BinaryVariable {
 public:
  BinaryVariable() { id_ = -1; log_potential_ = 0.0; }
  virtual ~BinaryVariable() {}

  // Number of factors linked to the variable.
  int Degree() { return factors_.size(); }

  // Get factor/link id.
  class Factor *GetFactor(int i) { return factors_[i]; }
  int GetLinkId(int i) { return links_[i]; }

  // Get/Set log-potential.
  double GetLogPotential() { return log_potential_; }
  void SetLogPotential(double log_potential) {
    log_potential_ = log_potential;
  }

  // Get/Set id.
  int GetId() { return id_; };
  void SetId(int id) { id_ = id; };

  // Add a new factor and link.
  void LinkToFactor(class Factor *factor, int link_id) {
    factors_.push_back(factor);
    links_.push_back(link_id);
  }

  // Remove all factors and links.
  void Disconnect() {
    factors_.clear();
    links_.clear();
  }

 private:
  int id_; // Variable Id.
  double log_potential_; // Log-potential of the variable.
  vector<Factor*> factors_; // Factors linked to the variable.
  vector<int> links_; // Link identifiers.
};

// Base class for a factor.
class Factor {
 public:
  Factor() {}
  virtual ~Factor() {}

  // Return the type.
  virtual int type() = 0;
  
  // True if generic factor.
  virtual bool IsGeneric() { return false; }

  // Return the number of binary variables linked to the factor.
  int Degree() { return binary_variables_.size(); };

  // Return a binary variable.
  BinaryVariable *GetVariable(int i) { return binary_variables_[i]; }

  // True if a binary variable is negated.
  bool IsVariableNegated(int i) { return negated_[i]; }

  // Print as a string.
  virtual void Print(ostream& stream) {
    stream << " " << binary_variables_.size();
    for (int i = 0; i < binary_variables_.size(); ++i) {
      stream << " " << (negated_[i]? "-" : "")
             << binary_variables_[i]->GetId() + 1;
    }
  }

  // Initialize factor.
  virtual void Initialize(const vector<BinaryVariable*> &binary_variables,
                          const vector<bool> &negated,
                          int *link_id) {
    binary_variables_ = binary_variables;
    if (negated.size() == 0) {
      negated_.assign(binary_variables_.size(), false);
    } else {
      negated_ = negated;
    }
    links_.resize(binary_variables_.size());
    for (int i = 0; i < binary_variables_.size(); ++i) {
      links_[i] = *link_id;
      binary_variables_[i]->LinkToFactor(this, links_[i]);
      ++(*link_id);
    }
  }

  // Get/Set id.
  int GetId() { return id_; };
  void SetId(int id) { id_ = id; };

  // Get link id.
  int GetLinkId(int i) { return links_[i]; };

  // Add evidence information to the factor.
  // The vector evidence contains global evidence information for
  // the variables. Entries set to -1 mean no evidence; entries set
  // to 0/1 mean that the variables are forced to have those values.
  // The vector active_links contains information about the links 
  // which were already disabled. 
  // Returns 0 if nothing changed.
  // Returns 1 if new evidence was set or new links were disabled,
  // but factor keeps active.
  // Returns 2 if factor became inactive.
  // Returns -1 if a contradiction was found, in which case the
  // problem is infeasible.
  virtual int AddEvidence(vector<bool> *active_links,
                          vector<int> *evidence,
                          vector<int> *additional_evidence) {
    // TODO: Implement this function for all the factors and make this pure
    // virtual. Right now we just have this implemented for the logic factors.
    assert(false);
    return 0;
  }

  // Gets/Sets additional log potentials.
  const vector<double> &GetAdditionalLogPotentials() {
    return additional_log_potentials_;
  }

  void SetAdditionalLogPotentials(
      const vector<double> &additional_log_potentials) {
    additional_log_potentials_ = additional_log_potentials;
  }

  // Gets/Sets/Computes cached values.
  vector<double> *GetMutableCachedVariableLogPotentials() {
    return &variable_log_potentials_last_;
  }
  void ComputeCachedAdditionalLogPotentials(double denominator) {
    additional_log_potentials_last_.resize(additional_log_potentials_.size());
    for (int i = 0; i < additional_log_potentials_.size(); ++i) {
      additional_log_potentials_last_[i] = 
          additional_log_potentials_[i] / denominator;
    }
  }
  const vector<double> &GetCachedVariablePosteriors() {
    return variable_posteriors_last_;
  }
  const vector<double> &GetCachedAdditionalPosteriors() {
    return additional_posteriors_last_;
  }

  // Compute the MAP (local subproblem in the projected subgradient algorithm).
  virtual void SolveMAP(const vector<double> &variable_log_potentials,
                        const vector<double> &additional_log_potentials,
                        vector<double> *variable_posteriors,
                        vector<double> *additional_posteriors,
                        double *value) = 0;

  // Solve the QP (local subproblem in the AD3 algorithm).
  virtual void SolveQP(const vector<double> &variable_log_potentials,
                       const vector<double> &additional_log_potentials,
                       vector<double> *variable_posteriors,
                       vector<double> *additional_posteriors) = 0;

  // Cached version of SolveMAP.
  virtual void SolveMAPCached(double *value) {
    SolveMAP(variable_log_potentials_last_,
             additional_log_potentials_last_,
             &variable_posteriors_last_,
             &additional_posteriors_last_,
             value);
  }
 
  // Cached version of SolveQP.
  virtual void SolveQPCached() {
    SolveQP(variable_log_potentials_last_,
            additional_log_potentials_last_,
            &variable_posteriors_last_,
            &additional_posteriors_last_);
  }

 private:
  int id_; // Factor id.

 protected:
  // Properties of the factor.
  vector<BinaryVariable*> binary_variables_;
  vector<bool> negated_;
  vector<int> links_;
  vector<double> additional_log_potentials_;

  // Cached potentials/posteriors.
  vector<double> variable_log_potentials_last_;
  vector<double> additional_log_potentials_last_;
  vector<double> variable_posteriors_last_;
  vector<double> additional_posteriors_last_;

};

// XOR factor. Only configurations with exactly one 1 are legal.
class FactorXOR : public Factor {
 public:
  int type() { return FactorTypes::FACTOR_XOR; }

  // Print as a string.
  void Print(ostream& stream) {
    stream << "XOR";
    Factor::Print(stream);
    stream << endl;
  }

  // Add evidence information to the factor.
  int AddEvidence(vector<bool> *active_links,
                  vector<int> *evidence,
                  vector<int> *additional_evidence);

  // Compute the MAP (local subproblem in the projected subgradient algorithm).
  void SolveMAP(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors,
                double *value);

  // Solve the QP (local subproblem in the AD3 algorithm).
  void SolveQP(const vector<double> &variable_log_potentials,
               const vector<double> &additional_log_potentials,
               vector<double> *variable_posteriors,
               vector<double> *additional_posteriors);

 private:
  // Cached copy of the last sort.
  vector<pair<double,int> > last_sort_;
};

// AtMostOne factor. Only configurations with at most one 1 are legal.
class FactorAtMostOne : public Factor {
 public:
  int type() { return FactorTypes::FACTOR_ATMOSTONE; }

  // Print as a string.
  void Print(ostream& stream) {
    stream << "ATMOSTONE";
    Factor::Print(stream);
    stream << endl;
  }

  // Add evidence information to the factor.
  int AddEvidence(vector<bool> *active_links,
                  vector<int> *evidence,
                  vector<int> *additional_evidence);

  // Compute the MAP (local subproblem in the projected subgradient algorithm).
  void SolveMAP(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors,
                double *value);

  // Solve the QP (local subproblem in the AD3 algorithm).
  void SolveQP(const vector<double> &variable_log_potentials,
               const vector<double> &additional_log_potentials,
               vector<double> *variable_posteriors,
               vector<double> *additional_posteriors);

 private:
  // Cached copy of the last sort.
  vector<pair<double,int> > last_sort_;
};

// OR factor. Only configurations with at least one 1 are legal.
class FactorOR : public Factor {
 public:
  int type() { return FactorTypes::FACTOR_OR; }

  // Print as a string.
  void Print(ostream& stream) {
    stream << "OR";
    Factor::Print(stream);
    stream << endl;
  }

  // Add evidence information to the factor.
  int AddEvidence(vector<bool> *active_links,
                  vector<int> *evidence,
                  vector<int> *additional_evidence);

  // Initialize from a OROUT factor in which the last variable is dropped (set 
  // to 1), turning the factor into a OR.
  void InitializeFromOROUT(Factor *factor) {
    assert(factor->type() == FactorTypes::FACTOR_OROUT);
    int degree = factor->Degree() - 1;
    binary_variables_.resize(degree);
    negated_.resize(degree);
    links_.resize(degree);
    for (int i = 0; i < degree; ++i) {
      binary_variables_[i] = factor->GetVariable(i);
      negated_[i] = factor->IsVariableNegated(i);
      links_[i] = factor->GetLinkId(i);
    }
  }

  // Compute the MAP (local subproblem in the projected subgradient algorithm).
  void SolveMAP(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors,
                double *value);

  // Solve the QP (local subproblem in the AD3 algorithm).
  void SolveQP(const vector<double> &variable_log_potentials,
               const vector<double> &additional_log_potentials,
               vector<double> *variable_posteriors,
               vector<double> *additional_posteriors);

 private:
  // Cached copy of the last sort.
  vector<pair<double,int> > last_sort_;
};

// OR-with-output factor. The last variable is the output 
// variable, and it must be the OR of the others.
class FactorOROUT : public Factor {
public:
  int type() { return FactorTypes::FACTOR_OROUT; }

  // Print as a string.
  void Print(ostream& stream) {
    stream << "OROUT";
    Factor::Print(stream);
    stream << endl;
  }

  // Add evidence information to the factor.
  int AddEvidence(vector<bool> *active_links,
                  vector<int> *evidence,
                  vector<int> *additional_evidence);

  // Compute the MAP (local subproblem in the projected subgradient algorithm).
  void SolveMAP(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors,
                double *value);

  // Solve the QP (local subproblem in the AD3 algorithm).
  void SolveQP(const vector<double> &variable_log_potentials,
               const vector<double> &additional_log_potentials,
               vector<double> *variable_posteriors,
               vector<double> *additional_posteriors);

 private:
  // Cached copy of the last sort.
  vector<pair<double,int> > last_sort_;
};

// PAIR factor. It connects a pair of binary variables, 
// with a Boltzmann log-potential for the case where both 
// variables are 1.
class FactorPAIR : public Factor {
 public:
  int type() { return FactorTypes::FACTOR_PAIR; }

  // Print as a string.
  void Print(ostream& stream) {
    stream << "PAIR";
    Factor::Print(stream);
    stream << " " << setprecision(9) << GetLogPotential() << endl;
  }

  // Get edge log-potential.
  double GetLogPotential() { return additional_log_potentials_[0]; }

  // Compute the MAP (local subproblem in the projected subgradient algorithm).
  void SolveMAP(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors,
                double *value);

  // Solve the QP (local subproblem in the AD3 algorithm).
  void SolveQP(const vector<double> &variable_log_potentials,
               const vector<double> &additional_log_potentials,
               vector<double> *variable_posteriors,
               vector<double> *additional_posteriors);

  // Add evidence information to the factor.
  int AddEvidence(vector<bool> *active_links,
                  vector<int> *evidence,
                  vector<int> *additional_evidence);
};

} // namespace AD3

#endif /* FACTOR_H_ */
