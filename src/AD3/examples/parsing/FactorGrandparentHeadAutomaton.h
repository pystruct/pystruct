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

#ifndef FACTOR_GRANDPARENT_HEAD_AUTOMATON
#define FACTOR_GRANDPARENT_HEAD_AUTOMATON

#include "ad3/GenericFactor.h"
#include "FactorHeadAutomaton.h"

namespace AD3 {

class Grandparent {
 public:
 Grandparent(int g, int h, int m) : g_(g), h_(h), m_(m) {}
  ~Grandparent() {}

  int grandparent() { return g_; }
  int head() { return h_; }
  int modifier() { return m_; }

 private:
  int g_;
  int h_;
  int m_;
};

class FactorGrandparentHeadAutomaton : public GenericFactor {
 public:
  // Compute the score of a given assignment.
  void Maximize(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                Configuration &configuration,
                double *value) {
    // Decode maximizing over the grandparents and using the Viterbi algorithm
    // as an inner loop.
    int num_grandparents = index_grandparents_.size();
    int best_grandparent = -1;
    int length = length_;
    vector<vector<double> > values(length);
    vector<vector<int> > path(length);
    vector<int> best_path(length);

    // Run Viterbi for each possible grandparent.
    for (int g = 0; g < num_grandparents; ++g) {
      // The start state is m = 0.
      values[0].resize(1);
      values[0][0] = 0.0;
      path[0].resize(1);
      path[0][0] = 0;
      for (int m = 1; m < length; ++m) {
        // m+1 possible states: either keep the previous state (no arc added)
        // or transition to a new state (arc between h and m).
        values[m].resize(m+1);
        path[m].resize(m+1);
        for (int i = 0; i < m; ++i) {
          // In this case, the previous state must also be i.
          values[m][i] = values[m-1][i];
          path[m][i] = i;
        }
        // For the m-th state, the previous state can be anything up to m-1.
        path[m][m] = -1;
        for (int j = 0; j < m; ++j) {
          int index = index_siblings_[j][m];
          double score = values[m-1][j] + additional_log_potentials[index];
          if (path[m][m] < 0 || score > values[m][m]) {
            values[m][m] = score;
            path[m][m] = j;
          } 
        }
        int index = index_grandparents_[g][m];
        values[m][m] += variable_log_potentials[num_grandparents+m-1] +
          additional_log_potentials[index];
      }

      // The end state is m = length.
      int best_last_state = -1;
      double best_score;
      for (int j = 0; j < length; ++j) {
        int index = index_siblings_[j][length];
        double score = values[length-1][j] + additional_log_potentials[index];
        if (best_last_state < 0 || score > best_score) {
          best_score = score;
          best_last_state = j;
        } 
      }

      // Add the score of the arc (g-->h).
      best_score += variable_log_potentials[g];

      // Only backtrack if the solution is the best so far.
      if (best_grandparent < 0 || best_score > *value) {
        // This is the best grandparent so far.
        best_grandparent = g;
        *value = best_score;
        best_path[length-1] = best_last_state;

        // Backtrack.
        for (int m = length-1; m > 0; --m) {
          best_path[m-1] = path[m][best_path[m]];
        }
      }
    }

    // Now write the configuration.
    vector<int> *grandparent_modifiers = 
      static_cast<vector<int>*>(configuration);
    grandparent_modifiers->push_back(best_grandparent);
    for (int m = 1; m < length; ++m) {
      if (best_path[m] == m) {
        grandparent_modifiers->push_back(m);
      }
    }
  }

  // Compute the score of a given assignment.
  void Evaluate(const vector<double> &variable_log_potentials,
                const vector<double> &additional_log_potentials,
                const Configuration configuration,
                double *value) {
    const vector<int>* grandparent_modifiers =
      static_cast<const vector<int>*>(configuration);
    // Grandparent belong to {0,1,...}
    // Modifiers belong to {1,2,...}
    *value = 0.0;
    int g = (*grandparent_modifiers)[0];
    *value += variable_log_potentials[g];
    int num_grandparents = index_grandparents_.size();
    int m = 0;
    for (int i = 1; i < grandparent_modifiers->size(); ++i) {
      int s = (*grandparent_modifiers)[i];
      *value += variable_log_potentials[num_grandparents+s-1];
      int index = index_siblings_[m][s];
      *value += additional_log_potentials[index];
      m = s;
      index = index_grandparents_[g][m];
      *value += additional_log_potentials[index];      
    }
    int s = index_siblings_.size();
    int index = index_siblings_[m][s];
    *value += additional_log_potentials[index];
  }

  // Given a configuration with a probability (weight), 
  // increment the vectors of variable and additional posteriors.
  void UpdateMarginalsFromConfiguration(
    const Configuration &configuration,
    double weight,
    vector<double> *variable_posteriors,
    vector<double> *additional_posteriors) {
    const vector<int> *grandparent_modifiers =
      static_cast<const vector<int>*>(configuration);
    int g = (*grandparent_modifiers)[0];
    (*variable_posteriors)[g] += weight;
    int num_grandparents = index_grandparents_.size();
    int m = 0;
    for (int i = 1; i < grandparent_modifiers->size(); ++i) {
      int s = (*grandparent_modifiers)[i];
      (*variable_posteriors)[num_grandparents+s-1] += weight;
      int index = index_siblings_[m][s];
      (*additional_posteriors)[index] += weight;
      m = s;
      index = index_grandparents_[g][m];
      (*additional_posteriors)[index] += weight;      
    }
    int s = index_siblings_.size();
    int index = index_siblings_[m][s];
    (*additional_posteriors)[index] += weight;
  }

  // Count how many common values two configurations have.
  int CountCommonValues(const Configuration &configuration1,
                        const Configuration &configuration2) {
    const vector<int> *values1 = static_cast<const vector<int>*>(configuration1);
    const vector<int> *values2 = static_cast<const vector<int>*>(configuration2);
    int count = 0;
    if ((*values1)[0] == (*values2)[0]) ++count; // Grandparents matched.
    int j = 1;
    for (int i = 1; i < values1->size(); ++i) {
      for (; j < values2->size(); ++j) {
        if ((*values2)[j] >= (*values1)[i]) break;
      }
      if (j < values2->size() && (*values2)[j] == (*values1)[i]) {
        ++count;
        ++j;
      }
    }
    return count;
  }

  // Check if two configurations are the same.
  bool SameConfiguration(
    const Configuration &configuration1,
    const Configuration &configuration2) {
    const vector<int> *values1 = static_cast<const vector<int>*>(configuration1);
    const vector<int> *values2 = static_cast<const vector<int>*>(configuration2);
    if (values1->size() != values2->size()) return false;
    for (int i = 0; i < values1->size(); ++i) {
      if ((*values1)[i] != (*values2)[i]) return false;
    }
    return true;    
  }

  // Delete configuration.
  void DeleteConfiguration(
    Configuration configuration) {
    vector<int> *values = static_cast<vector<int>*>(configuration);
    delete values;
  }

  Configuration CreateConfiguration() {
    // The first element is the index of the grandparent.
    // The remaining elements are the indices of the modifiers.
    vector<int>* grandparent_modifiers = new vector<int>;
    return static_cast<Configuration>(grandparent_modifiers); 
  }

 public:
  // length is relative to the head position. 
  // E.g. for a right automaton with h=3 and instance_length=10,
  // length = 7. For a left automaton, it would be length = 3.
  void Initialize(int length,
                  int num_grandparents,
                  const vector<Sibling*> &siblings,
                  const vector<Grandparent*> &grandparents) {
    length_ = length;
    index_grandparents_.assign(num_grandparents, vector<int>(length, -1));
    index_siblings_.assign(length, vector<int>(length+1, -1));
    for (int k = 0; k < siblings.size(); ++k) {
      int h = siblings[k]->head();
      int m = siblings[k]->modifier();
      int s = siblings[k]->sibling();
      if (s > h) {
        m -= h;
        s -= h;
      } else {
        m = h - m;
        s = h - s;
      }
      index_siblings_[m][s] = grandparents.size() + k;
    }
    for (int k = 0; k < grandparents.size(); ++k) {
      int g = grandparents[k]->grandparent();
      int h = grandparents[k]->head();
      int m = grandparents[k]->modifier();
      if (m > h) {
        m -= h;
      } else {
        m = h - m;
      }
      assert(g >= 0 && g < num_grandparents);
      index_grandparents_[g][m] = k;
    }
  }

 private:
  int length_;
  vector<vector<int> > index_siblings_;
  vector<vector<int> > index_grandparents_;
};

} // namespace AD3

#endif // FACTOR_GRANDPARENT_HEAD_AUTOMATON
