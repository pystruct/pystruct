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
// This file shows an example of AD3 being used for inference in a grid-shaped
// Potts model. We illustrate with two settings: one in which only pairwise
// factors (FactorDense with two multi-variables) are used, and another in 
// which horizontal and vertical lines of the grid are handled with a 
// a sequential factor. For this, we define a "sequence factor" which implements
// the Viterbi algorithm for decoding.
// This is done by deriving classes FactorSequence.
//
/////////////////////////////////////////////////////////////////////////////

#include "ad3/FactorGraph.h"
#include "FactorSequence.h"
#include <cstdlib>

void GetBestConfiguration(int grid_size, int num_states,
                          const vector<vector<AD3::MultiVariable*> >
                            &multi_variables,
                          double alpha,
                          const vector<double> &posteriors);
                          
int main(int argc, char **argv) {
  int grid_size = 20; 
  int num_states = 5;
  
  cout << "Creating model for a " << grid_size << "x" << grid_size
       << " grid with " << num_states << " states..."
       << endl;
       
  cout << "Building factor graph..."
       << endl;       

  // Create factor graph.
  AD3::FactorGraph factor_graph;

  // Create a multi-valued variable for each position in the grid.
  srand((unsigned)time(NULL));
  vector<vector<AD3::MultiVariable*> > multi_variables(grid_size,
    vector<AD3::MultiVariable*>(grid_size));
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      multi_variables[i][j] = factor_graph.CreateMultiVariable(num_states);
      for (int k = 0; k < num_states; ++k) {
        // Assign a random log-potential to each state.
        double score = static_cast<double>(rand()) /
          static_cast<double>(RAND_MAX) - 0.5;
        multi_variables[i][j]->SetLogPotential(k, score);
      }
    }
  }

  // Design the edge log-potentials.
  // Right now they are diagonal and favoring smooth configurations, but
  // that needs not be the case.
  vector<double> additional_log_potentials(num_states * num_states);
  double alpha = 0.5; // The "smoothness" degree.
  int t = 0;
  for (int k = 0; k < num_states; ++k) {
    for (int l = 0; l < num_states; ++l) {
      additional_log_potentials[t] = (k == l)? alpha : 0.0;
      ++t;
    }
  }
  
  // Create a factor for each edge in the grid.
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      vector<AD3::MultiVariable*> multi_variables_local(2);
            
      // Horizontal edge.
      if (j > 0) {
        multi_variables_local[0] = multi_variables[i][j-1];
        multi_variables_local[1] = multi_variables[i][j];
        factor_graph.CreateFactorDense(multi_variables_local,
                                       additional_log_potentials);
      }

      // Vertical edge.
      if (i > 0) {
        multi_variables_local[0] = multi_variables[i-1][j];
        multi_variables_local[1] = multi_variables[i][j];
        factor_graph.CreateFactorDense(multi_variables_local,
                                       additional_log_potentials);
      }
    }
  }

  vector<double> posteriors;
  vector<double> additional_posteriors;
  double value;

#if 0
  // Run the projected subgradient algorithm.
  cout << "Running projected subgradient..."
       << endl;
  factor_graph.SetEtaPSDD(1.0);
  factor_graph.SetMaxIterationsPSDD(1000);
  factor_graph.SolveLPMAPWithPSDD(&posteriors, &additional_posteriors, &value);
  GetBestConfiguration(grid_size, num_states, multi_variables, alpha,
                       posteriors);
#endif

  // Run AD3.
  cout << "Running AD3..."
       << endl;
  factor_graph.SetEtaAD3(0.1);
  factor_graph.AdaptEtaAD3(true);
  factor_graph.SetMaxIterationsAD3(1000);
  factor_graph.SolveLPMAPWithAD3(&posteriors, &additional_posteriors, &value);
  GetBestConfiguration(grid_size, num_states, multi_variables, alpha,
                       posteriors);


  cout << "Building sequential factor graph..."
       << endl;       

  // Create a factor graph using sequence-factors which is equivalent to the 
  // previous one.
  AD3::FactorGraph sequential_factor_graph;
  
  // Create a binary variable for each state at each position in the grid.
  vector<vector<vector<AD3::BinaryVariable*> > > binary_variables(grid_size,
    vector<vector<AD3::BinaryVariable*> >(grid_size,
      vector<AD3::BinaryVariable*>(num_states)));
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      for (int k = 0; k < num_states; ++k) {
        // Assign a random log-potential to each state.
        binary_variables[i][j][k] =
          sequential_factor_graph.CreateBinaryVariable();
        binary_variables[i][j][k]->SetLogPotential(
          multi_variables[i][j]->GetLogPotential(k));
      }
    }
  }
  
  // Design the edge log-potentials.
  // Right now they are diagonal and favoring smooth configurations, but
  // that needs not be the case.
  additional_log_potentials.clear();
  for (int i = 0; i <= grid_size; ++i) {
    int num_previous_states = (i == 0)? 1 : num_states;
    int num_current_states = (i == grid_size)? 1 : num_states;    
    for (int k = 0; k < num_previous_states; ++k) {
      for (int l = 0; l < num_current_states; ++l) {
        if (i != 0 && i != grid_size) {
          additional_log_potentials.push_back((k == l)? alpha : 0.0);
        } else {
          additional_log_potentials.push_back(0.0);
        }
      }
    }
  }
  
  // Create a sequential factor for each row in the grid.
  for (int i = 0; i < grid_size; ++i) {
    vector<AD3::BinaryVariable*> variables;
    vector<int> num_states(grid_size);
    for (int j = 0; j < grid_size; ++j) {
      variables.insert(variables.end(),
                       binary_variables[i][j].begin(),
                       binary_variables[i][j].end());     
      num_states[j] = binary_variables[i][j].size();
    }
    AD3::Factor *factor = new AD3::FactorSequence;
    // Let the factor graph own the factor so that we don't need to delete it.
    sequential_factor_graph.DeclareFactor(factor, variables, true);
    static_cast<AD3::FactorSequence*>(factor)->Initialize(num_states);
    factor->SetAdditionalLogPotentials(additional_log_potentials);
  }

  // Create a sequential factor for each column in the grid.
  for (int j = 0; j < grid_size; ++j) {
    vector<AD3::BinaryVariable*> variables;
    vector<int> num_states(grid_size);
    for (int i = 0; i < grid_size; ++i) {
      variables.insert(variables.end(),
                       binary_variables[i][j].begin(),
                       binary_variables[i][j].end());     
      num_states[i] = binary_variables[i][j].size();
    }
    AD3::Factor *factor = new AD3::FactorSequence;
    // Let the factor graph own the factor so that we don't need to delete it.
    sequential_factor_graph.DeclareFactor(factor, variables, true);
    static_cast<AD3::FactorSequence*>(factor)->Initialize(num_states);
    factor->SetAdditionalLogPotentials(additional_log_potentials);
  }

#if 0
  // Run the projected subgradient algorithm.
  cout << "Running projected subgradient..."
       << endl;
  sequential_factor_graph.SetEtaPSDD(1.0);
  sequential_factor_graph.SetMaxIterationsPSDD(1000);
  sequential_factor_graph.SolveLPMAPWithPSDD(&posteriors,
                                             &additional_posteriors, &value);
  GetBestConfiguration(grid_size, num_states, multi_variables, alpha,
                       posteriors);
#endif

  // Run AD3.
  cout << "Running AD3..."
       << endl;
  sequential_factor_graph.SetEtaAD3(0.1);
  sequential_factor_graph.AdaptEtaAD3(true);
  sequential_factor_graph.SetMaxIterationsAD3(1000);
  sequential_factor_graph.SolveLPMAPWithAD3(&posteriors,
                                            &additional_posteriors, &value);
  GetBestConfiguration(grid_size, num_states, multi_variables, alpha,
                       posteriors);

  return 0;
}

// Recover a valid integer assignment from a possibly fractional one.
// Use a simple heuristic that prefers the state with highest unary posterior.

void GetBestConfiguration(int grid_size, int num_states,
                          const vector<vector<AD3::MultiVariable*> >
                            &multi_variables,
                          double alpha,
                          const vector<double> &posteriors) {
  int offset = 0;
  double value = 0.0;
  vector<vector<int> > best_states(grid_size, vector<int>(grid_size));
  for (int i = 0; i < grid_size; ++i) {
    for (int j = 0; j < grid_size; ++j) {
      int best = -1;
      for (int k = 0; k < num_states; ++k) {
        if (best < 0 || posteriors[offset + k] > posteriors[offset + best]) {
          best = k;
        }
      }
      offset += num_states;
      best_states[i][j] = best;
      value += multi_variables[i][j]->GetLogPotential(best); 
      if (j > 0 && best_states[i][j-1] == best) value += alpha;
      if (i > 0 && best_states[i-1][j] == best) value += alpha;
      cout << best;
    }
    cout << endl;
  }
  cout << endl;
  cout << "Best primal value: " << value << endl;
}
