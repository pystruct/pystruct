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

#ifndef FACTOR_SEQUENCE
#define FACTOR_SEQUENCE

#include "ad3/GenericFactor.h"

namespace AD3 {

class FactorSequence : public GenericFactor {
 public:
  // Compute the score of a given assignment.
  void Maximize(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                Configuration &configuration,
                double *value) {
    // Decode using the Viterbi algorithm.
    int length = num_states_.size();
    vector<vector<double> > values(length);
    vector<vector<int> > path(length);
    int offset_states = 0;

    // Initialization.
    int num_states = num_states_[0];
    values[0].resize(num_states);
    path[0].resize(num_states);
    for (int l = 0; l < num_states; ++l) {
      int index = index_edges_[0][0][l];
      values[0][l] = variable_log_potentials[offset_states + l] +
        additional_log_potentials[index]; 
      path[0][l] = -1; // This won't be used.
    }
    offset_states += num_states;

    // Recursion.
    for (int i = 0; i < length - 1; ++i) {
      int num_states = num_states_[i + 1];
      values[i + 1].resize(num_states);
      path[i + 1].resize(num_states);
      for (int k = 0; k < num_states; ++k) {
        double best_value;
        int best = -1;
        for (int l = 0; l < num_states_[i]; ++l) {
          int index = index_edges_[i+1][l][k];
          double val = values[i][l] + additional_log_potentials[index];
          if (best < 0 || val > best_value) {
            best_value = val;
            best = l;
          }
        }
        values[i+1][k] = best_value + 
          variable_log_potentials[offset_states + k];
        path[i+1][k] = best;
      }
      offset_states += num_states;
    }

    // Termination.
    double best_value;
    int best = -1;
    for (int l = 0; l < num_states_[length - 1]; ++l) {
      int index = index_edges_[length][l][0];
      double val = values[length - 1][l] + 
        additional_log_potentials[index]; 
      if (best < 0 || val > best_value) {
        best_value = val;
        best = l;
      }
    }

    // Path (state sequence) backtracking.
    vector<int> *sequence = static_cast<vector<int>*>(configuration);
    assert(sequence->size() == length);
    (*sequence)[length - 1] = best;
    for (int i = length - 1; i > 0; --i) {
      (*sequence)[i - 1] = path[i][(*sequence)[i]];
    }

    *value = best_value;
  }

  // Compute the score of a given assignment.
  void Evaluate(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                const Configuration configuration,
                double *value) {
    const vector<int>* sequence =
        static_cast<const vector<int>*>(configuration);
    *value = 0.0;
    int offset_states = 0;
    int previous_state = 0;
    for (int i = 0; i < sequence->size(); ++i) {
      int state = (*sequence)[i];
      *value += variable_log_potentials[offset_states + state];
      offset_states += num_states_[i]; 
      int index = index_edges_[i][previous_state][state];
      *value += additional_log_potentials[index];
      previous_state = state;
    }
    int index = index_edges_[sequence->size()][previous_state][0];
    *value += additional_log_potentials[index];
  }

  // Given a configuration with a probability (weight), 
  // increment the vectors of variable and additional posteriors.
  void UpdateMarginalsFromConfiguration(
    const Configuration &configuration,
    double weight,
    vector<double> *variable_posteriors,
    vector<double> *additional_posteriors) {
    const vector<int> *sequence =
        static_cast<const vector<int>*>(configuration);
    int offset_states = 0;
    int previous_state = 0;
    for (int i = 0; i < sequence->size(); ++i) {
      int state = (*sequence)[i];
      (*variable_posteriors)[offset_states + state] += weight;
      offset_states += num_states_[i]; 
      int index = index_edges_[i][previous_state][state];
      (*additional_posteriors)[index] += weight;
      previous_state = state;
    }
    int index = index_edges_[sequence->size()][previous_state][0];
    (*additional_posteriors)[index] += weight;
  }

  // Count how many common values two configurations have.
  int CountCommonValues(const Configuration &configuration1,
                        const Configuration &configuration2) {
    const vector<int> *sequence1 =
        static_cast<const vector<int>*>(configuration1);
    const vector<int> *sequence2 =
        static_cast<const vector<int>*>(configuration2);
    assert(sequence1->size() == sequence2->size());
    int count = 0;
    for (int i = 0; i < sequence1->size(); ++i) {
      if ((*sequence1)[i] == (*sequence2)[i]) ++count;
    }
    return count;
  }

  // Check if two configurations are the same.
  bool SameConfiguration(
    const Configuration &configuration1,
    const Configuration &configuration2) {
    const vector<int> *sequence1 = static_cast<const vector<int>*>(configuration1);
    const vector<int> *sequence2 = static_cast<const vector<int>*>(configuration2);
    assert(sequence1->size() == sequence2->size());
    for (int i = 0; i < sequence1->size(); ++i) {
      if ((*sequence1)[i] != (*sequence2)[i]) return false;
    }
    return true;
  }

  // Delete configuration.
  void DeleteConfiguration(
    Configuration configuration) {
    vector<int> *sequence = static_cast<vector<int>*>(configuration);
    delete sequence;
  }

  Configuration CreateConfiguration() {
    int length = num_states_.size();
    vector<int>* sequence = new vector<int>(length, -1);
    return static_cast<Configuration>(sequence); 
  }

 public:
  // num_states contains the number of states at each position
  // in the sequence. The start and stop positions are not considered here.
  // Note: the variables and the the additional log-potentials must be ordered
  // properly.
  void Initialize(const vector<int> &num_states) {
    int length = num_states.size();
    num_states_ = num_states;
    index_edges_.resize(length + 1);
    int index = 0;
    for (int i = 0; i <= length; ++i) {
      // If i == 0, the previous state is the start symbol.
      int num_previous_states = (i > 0)? num_states_[i - 1] : 1;
      // If i == length-1, the previous state is the final symbol.
      int num_current_states = (i < length)? num_states_[i] : 1;
      index_edges_[i].resize(num_previous_states);
      for (int j = 0; j < num_previous_states; ++j) {
        index_edges_[i][j].resize(num_current_states);
        for (int k = 0; k < num_current_states; ++k) {
          index_edges_[i][j][k] = index;
          ++index;
        }
      }
    }
  }

 private:
  // Number of states for each position.
  vector<int> num_states_;
  // At each position, map from edges of states to a global index which 
  // matches the index of additional_log_potentials_.
  vector<vector<vector<int> > > index_edges_; 
};

} // namespace AD3

#endif // FACTOR_SEQUENCE
