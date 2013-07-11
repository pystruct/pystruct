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

#ifndef FACTOR_TREE_H_
#define FACTOR_TREE_H_

#include "FactorTree.h"
#include <iostream>
#include <vector>
#include <utility>
#include <iterator>
#include <cstdlib>
#include <limits>
#include <cerrno>

using namespace std;

namespace AD3 {

// Decoder for the basic model; it finds a maximum weighted arborescence
// using Chu-Liu-Edmonds' algorithm.
int FactorTree::RunCLE(const vector<double>& scores,
                       vector<int> *heads,
                       double *value) {


  // Done once.
  vector<vector<int> > candidate_heads(length_);
  vector<vector<double> > candidate_scores(length_);
  vector<bool> disabled(length_, false);
  for (int m = 1; m < length_; ++m) {
    for (int h = 0; h < length_; ++h) {
      int r = index_arcs_[h][m];
      if (r < 0) continue;
      candidate_heads[m].push_back(h);
      candidate_scores[m].push_back(scores[r]);
    }
  }

  RunChuLiuEdmondsIteration(&disabled, &candidate_heads,
                            &candidate_scores, heads,
                            value);

  *value = 0;
  (*heads)[0] = -1;
  for (int m = 1; m < length_; ++m) {
    int h = (*heads)[m];
    assert(h >= 0 && h < length_);
    int r = index_arcs_[h][m];
    assert(r >= 0);
    *value += scores[r];
  }
}

void FactorTree::RunChuLiuEdmondsIteration(vector<bool> *disabled,
                                           vector<vector<int> > *candidate_heads,
                                           vector<vector<double> >
                                           *candidate_scores,
                                           vector<int> *heads,
                                           double *value) {
  // Original number of nodes (including the root).
  int length = disabled->size();

  // Pick the best incoming arc for each node.
  heads->resize(length);
  vector<double> best_scores(length);
  for (int m = 1; m < length; ++m) {
    if ((*disabled)[m]) continue;
    int best = -1;
    for (int k = 0; k < (*candidate_heads)[m].size(); ++k) {
      if (best < 0 ||
          (*candidate_scores)[m][k] > (*candidate_scores)[m][best]) {
        best = k;
      }
    }
    if (best < 0) {
      // No spanning tree exists. Assign the parent of this node
      // to the root, and give it a minus infinity score.
      (*heads)[m] = 0;
      best_scores[m] = -std::numeric_limits<double>::infinity();
    } else {
      (*heads)[m] = (*candidate_heads)[m][best]; //best;
      best_scores[m] = (*candidate_scores)[m][best]; //best;
    }
  }

  // Look for cycles. Return after the first cycle is found.
  vector<int> cycle;
  vector<int> visited(length, 0);
  for (int m = 1; m < length; ++m) {
    if ((*disabled)[m]) continue;
    // Examine all the ancestors of m until the root or a cycle is found.
    int h = m;
    while (h != 0) {
      // If already visited, break and check if it is part of a cycle.
      // If visited[h] < m, the node was visited earlier and seen not
      // to be part of a cycle.
      if (visited[h]) break;
      visited[h] = m;
      h = (*heads)[h];
    }

    // Found a cycle to which h belongs.
    // Obtain the full cycle.
    if (visited[h] == m) {
      m = h;
      do {
        cycle.push_back(m);
        m = (*heads)[m];
      } while (m != h);
      break;
    }
  }

  // If there are no cycles, then this is a well formed tree.
  if (cycle.empty()) {
    *value = 0.0;
    for (int m = 1; m < length; ++m) {
      *value += best_scores[m];
    }
    return;
  }

  // Build a cycle membership vector for constant-time querying and compute the
  // score of the cycle.
  // Nominate a representative node for the cycle and disable all the others.
  double cycle_score = 0.0;
  vector<bool> in_cycle(length, false);
  int representative = cycle[0];
  for (int k = 0; k < cycle.size(); ++k) {
    int m = cycle[k];
    in_cycle[m] = true;
    cycle_score += best_scores[m];
    if (m != representative) (*disabled)[m] = true;
  }

  // Contract the cycle.
  // 1) Update the score of each child to the maximum score achieved by a parent
  // node in the cycle.
  vector<int> best_heads_cycle(length);
  for (int m = 1; m < length; ++m) {
    if ((*disabled)[m] || m == representative) continue;
    double best_score;
    // If the list of candidate parents of m is shorter than the length of
    // the cycle, use that. Otherwise, loop through the cycle.
    int best = -1;
    for (int k = 0; k < (*candidate_heads)[m].size(); ++k) {
      if (!in_cycle[(*candidate_heads)[m][k]]) continue;
      if (best < 0 || (*candidate_scores)[m][k] > best_score) {
        best = k;
        best_score = (*candidate_scores)[m][best];
      }
    }
    if (best < 0) continue;
    best_heads_cycle[m] = (*candidate_heads)[m][best];

    // Reconstruct the list of candidate heads for this m.
    int l = 0;
    for (int k = 0; k < (*candidate_heads)[m].size(); ++k) {
      int h = (*candidate_heads)[m][k];
      double score = (*candidate_scores)[m][k];
      if (!in_cycle[h]) {
        (*candidate_heads)[m][l] = h;
        (*candidate_scores)[m][l] = score;
        ++l;
      }
    }
    // If h is in the cycle and is not the representative node,
    // it will be dropped from the list of candidate heads.
    (*candidate_heads)[m][l] = representative;
    (*candidate_scores)[m][l] = best_score;
    (*candidate_heads)[m].resize(l+1);
    (*candidate_scores)[m].resize(l+1);
  }

  // 2) Update the score of each candidate parent of the cycle supernode.
  vector<int> best_modifiers_cycle(length, -1);
  vector<int> candidate_heads_representative;
  vector<double> candidate_scores_representative;

  vector<double> best_scores_cycle(length);
  // Loop through the cycle.
  for (int k = 0; k < cycle.size(); ++k) {
    int m = cycle[k];
    for (int l = 0; l < (*candidate_heads)[m].size(); ++l) {
      // Get heads out of the cycle.
      int h = (*candidate_heads)[m][l];
      if (in_cycle[h]) continue;

      double score = (*candidate_scores)[m][l] - best_scores[m];
      if (best_modifiers_cycle[h] < 0 || score > best_scores_cycle[h]) {
        best_modifiers_cycle[h] = m;
        best_scores_cycle[h] = score;
      }
    }
  }
  for (int h = 0; h < length; ++h) {
    if (best_modifiers_cycle[h] < 0) continue;
    double best_score = best_scores_cycle[h] + cycle_score;
    candidate_heads_representative.push_back(h);
    candidate_scores_representative.push_back(best_score);
  }

  // Reconstruct the list of candidate heads for the representative node.
  (*candidate_heads)[representative] = candidate_heads_representative;
  (*candidate_scores)[representative] = candidate_scores_representative;

  // Save the current head of the representative node (it will be overwritten).
  int head_representative = (*heads)[representative];

  // Call itself recursively.
  RunChuLiuEdmondsIteration(disabled,
                            candidate_heads,
                            candidate_scores,
                            heads,
                            value);

  // Uncontract the cycle.
  int h = (*heads)[representative];
  (*heads)[representative] = head_representative;
  (*heads)[best_modifiers_cycle[h]] = h;

  for (int m = 1; m < length; ++m) {
    if ((*disabled)[m]) continue;
    if ((*heads)[m] == representative) {
      // Get the right parent from within the cycle.
      (*heads)[m] = best_heads_cycle[m];
    }
  }
  for (int k = 0; k < cycle.size(); ++k) {
    int m = cycle[k];
    (*disabled)[m] = false;
  }
}

} // namespace AD3

#endif // FACTOR_TREE_H_
