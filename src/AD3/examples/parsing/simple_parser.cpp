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
//

/////////////////////////////////////////////////////////////////////////////
//
// This file shows an example of a dependency parser which combines a
// maximum spanning tree model with head automata models.
// This model is analogous to the "sibling model" in:
//
// Terry Koo, Alexander M. Rush, Michael Collins, Tommi Jaakkola,
// and David Sontag. Dual Decomposition for Parsing with Non-Projective
// Head Automata. Proceedings of EMNLP 2010.
//
// To handle this model with AD3, we define a "tree factor" which implements
// the Chu-Liu-Edmonds algorithm, a "head automaton factor" which implements
// a sequence model representing one head word and the set of possible
// modifiers (on the same side), with scores for consecutive siblings.
// The decoder is the Viterbi algorithm.
// This is done by deriving classes FactorTree and FactorHeadAutomaton.
//
/////////////////////////////////////////////////////////////////////////////

#include "ad3/FactorGraph.h"
#include "FactorHeadAutomaton.h"
#include "FactorTree.h"
#include <cstdlib>

using namespace AD3;

void GetBestParse(int sentence_length,
                  const vector<Arc*> &arcs, const vector<Sibling*> &siblings,
                  const vector<double> &arc_scores,
                  const vector<double> &sibling_scores,
                  Factor *tree_factor, const vector<double> &posteriors);

int main(int argc, char **argv) {
  int sentence_length = 40;

  cout << "Creating model for sentence with "
       << sentence_length << " words..."
       << endl;

  // Create arcs.
  vector<Arc*> arcs;
  vector<vector<int> > index_arcs(sentence_length,
    vector<int>(sentence_length, -1));
  for (int m = 1; m < sentence_length; ++m) {
    for (int h = 0; h < sentence_length; ++h) {
      if (h == m) continue;
      Arc *arc = new Arc(h, m);
      index_arcs[h][m] = arcs.size();
      arcs.push_back(arc);
    }
  }

  // Create siblings.
  vector<Sibling*> siblings;
  // Right siblings.
  for (int h = 0; h < sentence_length; ++h) {
    for (int m = h; m < sentence_length; ++m) {
      for (int s = m+1; s <= sentence_length; ++s) {
        Sibling *sibling = new Sibling(h, m, s);
        siblings.push_back(sibling);
      }
    }
  }
  // Left siblings.
  for (int h = 1; h < sentence_length; ++h) {
    for (int m = h; m >= 1; --m) {
      for (int s = m-1; s >= 0; --s) {
        Sibling *sibling = new Sibling(h, m, s);
        siblings.push_back(sibling);
      }
    }
  }

  // Assign a score to each arc.
  srand((unsigned)time(NULL));
  vector<double> arc_scores(arcs.size());
  for (int i = 0; i < arcs.size(); ++i) {
    double score = static_cast<double>(rand()) /
      static_cast<double>(RAND_MAX) - 0.5;
    arc_scores[i] = (double) score;
  }

  // Assign a score to each sibling.
  vector<double> sibling_scores(siblings.size());
  for (int i = 0; i < siblings.size(); ++i) {
    double score = 0.1*(static_cast<double>(rand()) /
      static_cast<double>(RAND_MAX) - 0.5);
    sibling_scores[i] = score;
  }

  cout << "Building factor graph..."
       << endl;

  // Create factor graph and define factors (subproblems).
  FactorGraph factor_graph;
  vector<Factor*> factors;
  // Create variables (one per arc).
  vector<BinaryVariable*> variables(arcs.size());
  for (int i = 0; i < arcs.size(); ++i) {
    BinaryVariable* variable = factor_graph.CreateBinaryVariable();
    variable->SetLogPotential(arc_scores[i]);
    variables[i] = variable;
  }

  cout << "Creating tree factor..."
       << endl;

  // Create a tree factor connected to all arcs.
  Factor *tree_factor = new FactorTree;
  factors.push_back(tree_factor);
  factor_graph.DeclareFactor(tree_factor,
                             variables);
  static_cast<FactorTree*>(tree_factor)->Initialize(sentence_length, arcs);
  int k = 0;
  // Create right-side head automaton factors (one per head word).
  for (int h = 0; h < sentence_length; ++h) {
    vector<BinaryVariable*> local_variables;
    vector<double> additional_scores;
    vector<Sibling*> local_siblings;
    for (int m = h; m < sentence_length; ++m) {
      if (m != h) {
        int index = index_arcs[h][m];
        if (index >= 0) local_variables.push_back(variables[index]);
      }
      for (int s = m+1; s <= sentence_length; ++s) {
        // Additional score for sibling (h,m,s).
        additional_scores.push_back(sibling_scores[k]);
        local_siblings.push_back(siblings[k]);
        ++k;
      }
    }
    // Don't create an empty factor.
    //if (local_variables.size() == 0) continue;

    cout << "Creating right head automaton factor..."
         << endl;

    Factor *head_automaton_factor = new FactorHeadAutomaton;
    factors.push_back(head_automaton_factor);
    static_cast<FactorHeadAutomaton*>(head_automaton_factor)->Initialize(
        sentence_length - h, local_siblings);
    factor_graph.DeclareFactor(head_automaton_factor,
                               local_variables);
    head_automaton_factor->SetAdditionalLogPotentials(additional_scores);
  }

  // Create left-side head automaton factors (one per head word except root).
  for (int h = 1; h < sentence_length; ++h) {
    vector<BinaryVariable*> local_variables;
    vector<double> additional_scores;
    vector<Sibling*> local_siblings;
    for (int m = h; m >= 1; --m) {
      if (m != h) {
        int index = index_arcs[h][m];
        if (index >= 0) local_variables.push_back(variables[index]);
      }
      for (int s = m-1; s >= 0; --s) {
        // Additional score for sibling (h,m,s).
        additional_scores.push_back(sibling_scores[k]);
        local_siblings.push_back(siblings[k]);
        ++k;
      }
    }
    // Don't create an empty factor.
    //if (local_variables.size() == 0) continue;

    cout << "Creating left head automaton factor..."
         << endl;

    Factor *head_automaton_factor = new FactorHeadAutomaton;
    factors.push_back(head_automaton_factor);
    static_cast<FactorHeadAutomaton*>(head_automaton_factor)->Initialize(h, local_siblings);
    factor_graph.DeclareFactor(head_automaton_factor,
                               local_variables);
    head_automaton_factor->SetAdditionalLogPotentials(additional_scores);
  }
  assert(k == siblings.size());

  vector<double> posteriors;
  vector<double> additional_posteriors;
  double value;

  // Run the projected subgradient algorithm.
  cout << "Running projected subgradient..."
       << endl;
  factor_graph.SetEtaPSDD(1.0);
  factor_graph.SetMaxIterationsPSDD(1000);
  factor_graph.SolveLPMAPWithPSDD(&posteriors, &additional_posteriors, &value);
  GetBestParse(sentence_length, arcs, siblings, arc_scores, sibling_scores,
               tree_factor, posteriors);

  // Run AD3.
  cout << "Running AD3..."
       << endl;
  factor_graph.SetEtaAD3(0.1);
  factor_graph.AdaptEtaAD3(true);
  factor_graph.SetMaxIterationsAD3(1000);
  factor_graph.SolveLPMAPWithAD3(&posteriors, &additional_posteriors, &value);
  GetBestParse(sentence_length, arcs, siblings, arc_scores, sibling_scores,
               tree_factor, posteriors);

  // Destroy factors.
  for (int i = 0; i < factors.size(); ++i) {
    delete factors[i];
  }

  // Destroy arcs and siblings.
  for (int i = 0; i < arcs.size(); ++i) {
    delete arcs[i];
  }
  for (int i = 0; i < siblings.size(); ++i) {
    delete siblings[i];
  }

  return 0;
}

// Recover a valid parse tree from a possibly fractional solution.
// This is done as described in
//
// André F. T. Martins, Noah A. Smith, and Eric P. Xing.
// "Concise Integer Linear Programming Formulations for Dependency Parsing."
//  Annual Meeting of the Association for Computational Linguistics, 2009.
//
// Basically we use the fractional memberships as scores and invoke
// the Chu-Liu-Edmonds algorithm.

void GetBestParse(int sentence_length,
                  const vector<Arc*> &arcs, const vector<Sibling*> &siblings,
                  const vector<double> &arc_scores,
                  const vector<double> &sibling_scores,
                  Factor *tree_factor, const vector<double> &posteriors) {
  vector<double> scores(arcs.size());
  for (int i = 0; i < arcs.size(); ++i) {
    scores[i] = posteriors[i];
  }
  vector<int> heads(sentence_length);
  double best_value;
  static_cast<FactorTree*>(tree_factor)->RunCLE(scores, &heads, &best_value);
  cout << best_value << endl;
  for (int i = 0; i < heads.size(); ++i) {
    cout << heads[i] << " ";
  }
  cout << endl;
  best_value = 0;
  for (int i = 0; i < arcs.size(); ++i) {
    int h = arcs[i]->head();
    int m = arcs[i]->modifier();
    if (heads[m] == h) {
      best_value += arc_scores[i];
      //cout << h << " " << m << ": " << arc_scores[i] << endl;
    }
  }
  for (int i = 0; i < siblings.size(); ++i) {
    int h = siblings[i]->head();
    int m = siblings[i]->modifier();
    int s = siblings[i]->sibling();
    if ((h == m || heads[m] == h) &&
        (s == 0 || s == sentence_length || heads[s] == h)) {
      bool consecutive = true;
      if (m < s) {
        for (int t = m+1; t < s; ++t) {
          if (heads[t] == h) {
            consecutive = false;
            break;
          }
        }
      } else {
        for (int t = m-1; t > s; --t) {
          if (heads[t] == h) {
            consecutive = false;
            break;
          }
        }
      }
      if (consecutive) {
        //cout << h << " " << m << " " << s << ": " << sibling_scores[i] << endl;
        best_value += sibling_scores[i];
      }
    }
  }
  cout << "Best primal value: " << best_value << endl;
}
