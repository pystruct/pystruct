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
// This file shows an example of AD3 being used in a simple model for 
// co-reference resolution. There are pairwise similarities between words,
// and there is a transitivity constraint - if words A and B are co-referent, 
// and so are B and C, then A and C must also be coreferent. Finally, some
// there is a constraint that any cluster of co-referents must have a word 
// which is not a prononun. 
// This is done by using FactorIMPLY and FactorOR.
//
/////////////////////////////////////////////////////////////////////////////

#include "ad3/FactorGraph.h"
#include <cstdlib>

int main(int argc, char **argv) {
  int num_words = 10;
  double pronoun_probability = 0.8; // Probability of a word being a pronoun.

  cout << "Creating a coreference model with " << num_words << " words..."
       << endl;
       
  vector<bool> is_pronoun(num_words, false);
  srand((unsigned)time(NULL));
  // At least two nouns.
  is_pronoun[0] = false;
  is_pronoun[1] = false;
  for (int i = 2; i < num_words; ++i) {
    if (static_cast<double>(rand()) / static_cast<double>(RAND_MAX) < 
      pronoun_probability)    
    is_pronoun[i] = true;
  }
  
  cout << "Building factor graph..."
       << endl;       

  // Create factor graph.
  AD3::FactorGraph factor_graph;

  // Each candidate co-referent pair is a binary variable in the factor graph.
  vector<vector<AD3::BinaryVariable*> > binary_variables(num_words);
  for (int i = 0; i < num_words; ++i) {
    binary_variables[i].resize(i);
    for (int j = 0; j < i; ++j) {
      binary_variables[i][j] = factor_graph.CreateBinaryVariable();
      // Set pairwise scores for two words being co-referent.
      double score =
        static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 0.5;
      binary_variables[i][j]->SetLogPotential(score);
    }
  }

  // Impose transitivity constraints.
  for (int i = 0; i < num_words; ++i) {
    for (int j = 0; j < i; ++j) {
      for (int k = 0; k < j; ++k) {
        vector<AD3::BinaryVariable*> local_variables(3);
        // ij ^ jk => ik
        local_variables[0] = binary_variables[i][j];
        local_variables[1] = binary_variables[j][k];
        local_variables[2] = binary_variables[i][k];
        factor_graph.CreateFactorIMPLY(local_variables);
                
        // jk ^ ik => ij
        local_variables[0] = binary_variables[j][k];
        local_variables[1] = binary_variables[i][k];
        local_variables[2] = binary_variables[i][j];
        factor_graph.CreateFactorIMPLY(local_variables);
  
        // ik ^ ij => jk
        local_variables[0] = binary_variables[i][k];
        local_variables[1] = binary_variables[i][j];
        local_variables[2] = binary_variables[j][k];
        factor_graph.CreateFactorIMPLY(local_variables);
      }
    }
  }
  
  // Impose that every cluster must have a non-pronoun.
  // This is done by defining a OR factor for each pronoun, 
  // linked to all the pairs formed by that pronoun and each noun word.
  for (int i = 0; i < num_words; ++i) {
    if (!is_pronoun[i]) continue;
    vector<AD3::BinaryVariable*> local_variables;
    for (int j = 0; j < i; ++j) {
      if (is_pronoun[j]) continue;
      local_variables.push_back(binary_variables[i][j]);
    }
    for (int k = i+1; k < num_words; ++k) {
      if (is_pronoun[k]) continue;
      local_variables.push_back(binary_variables[k][i]);
    }
    factor_graph.CreateFactorOR(local_variables);
  }

  vector<double> posteriors;
  vector<double> additional_posteriors;
  double value;

  // Run the projected subgradient algorithm.
  cout << "Running projected subgradient..."
       << endl;
  factor_graph.SetEtaPSDD(1.0);
  factor_graph.SetMaxIterationsPSDD(1000);
  factor_graph.SolveLPMAPWithPSDD(&posteriors, &additional_posteriors, &value);

  // Run AD3.
  cout << "Running AD3..."
       << endl;
  factor_graph.SetEtaAD3(0.1);
  factor_graph.AdaptEtaAD3(true);
  factor_graph.SetMaxIterationsAD3(1000);
  factor_graph.SolveLPMAPWithAD3(&posteriors, &additional_posteriors, &value);

  return 0;
}

