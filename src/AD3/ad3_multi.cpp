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

#include <math.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <assert.h>
#include "ad3/FactorGraph.h"
#include "ad3/Utils.h"
#include "FactorDense.h"
#include "FactorSequence.h"
#include "FactorTree.h"
#include "FactorHeadAutomaton.h"
#include "FactorGrandparentHeadAutomaton.h"

using namespace std;
using namespace AD3;

#define BUFFERSIZE 1024

int RunAll(const string &format,
           const string &filename_graph,
           const string &algorithm,
           int niters,
           double eta,
           bool adapt_eta,
           double residual_threshold,
           bool convert_to_binary,
           bool exact,
           const string &filename_posteriors);

int LoadGraph(ifstream &file_graph, 
              FactorGraph *factor_graph);

int LoadGraphUAI(ifstream &file_graph, 
                 FactorGraph *factor_graph);

int main(int argc, char** argv) {
  string message = "Usage: ad3_multi --format=[ad3(*)|uai] " \
    "--file_graphs=[IN] --file_posteriors=[OUT] " \
    "--algorithm=[ad3(*)|psdd|mplp] " \
    "(--max_iterations=[NUM] --eta=[NUM] --adapt_eta=[true(*)|false] " \
    "--residual_threshold=[NUM] --convert_to_binary=[true|false(*)] " \
    "--exact=[true|false(*)])";
  if (argc == 1) {
    cout << message << endl;
    return 0;
  }

  string format = "ad3";
  string algorithm = "ad3";
  string filename_graph = "";
  int niters = 1000;
  double eta = 0.1;
  double residual_threshold = 1e-6;
  string filename_posteriors = "";
  bool adapt_eta = true;
  bool convert_to_binary = false;
  bool exact = false;
  
  for (int i = 1; i < argc; ++i) {
    vector<string> pair;
    StringSplit(argv[i], "=", &pair);
    if (pair.size() != 2 || pair[0].substr(0,2) != "--") {
      cout << message << endl;
      return -1;
    }
    string param_name = pair[0].substr(2);
    string param_value = pair[1];
    if (param_name == "format") {
      format = param_value;
    } else if (param_name == "algorithm") {
      algorithm = param_value;
    } else if (param_name == "file_graphs") {
      filename_graph = param_value;
    } else if (param_name == "file_posteriors") {
      filename_posteriors = param_value;
    } else if (param_name == "max_iterations") {
      niters = atoi(param_value.c_str());
    } else if (param_name == "eta") {
      eta = atof(param_value.c_str());
    } else if (param_name == "adapt_eta") {
      if (param_value == "false") {
        adapt_eta = false;
      } else if (param_value == "true") {
        adapt_eta = true;
      } else {
        cout << "Unknown value for flag " << param_name 
             << ": " << param_value << endl;
        cout << message << endl;
        return -1;
      }
    } else if (param_name == "residual_threshold") {
      residual_threshold = atof(param_value.c_str());
    } else if (param_name == "convert_to_binary") {
      if (param_value == "false") {
        convert_to_binary = false;
      } else if (param_value == "true") {
        convert_to_binary = true;
      } else {
        cout << "Unknown value for flag " << param_name 
             << ": " << param_value << endl;
        cout << message << endl;
        return -1;
      }
    } else if (param_name == "exact") {
      if (param_value == "false") {
        exact = false;
      } else if (param_value == "true") {
        exact = true;
      } else {
        cout << "Unknown value for flag " << param_name << ": " << param_value << endl;
        cout << message << endl;
        return -1;
      }
    } else {
      cout << "Unknown flag: " << param_name << endl;
      cout << message << endl;
      return -1;
    }
  }

  if (exact && algorithm != "ad3") {
    cout << "Error: flag --exact=true can only be set with --algorithm=ad3.";
    return -1;
  }

  RunAll(format,
         filename_graph,
         algorithm,
         niters,
         eta,
         adapt_eta,
         residual_threshold,
         convert_to_binary,
         exact,
         filename_posteriors);

  return 0;
}

int RunAll(const string &format,
           const string &filename_graph,
           const string &algorithm,
           int niters,
           double eta,
           bool adapt_eta,
           double residual_threshold,
           bool convert_to_binary,
           bool exact,
           const string &filename_posteriors) {
  int time_ddadmm_relax = 0;
  int time_ddadmm = 0;
  int time_cplex_relax = 0;
  int time_cplex_integer = 0;
  ifstream file_graph(filename_graph.c_str(), ios_base::in);
  ofstream file_posteriors(filename_posteriors.c_str(), ios_base::out);
  if (file_graph.is_open()) {
    while (!file_graph.eof()) {
      FactorGraph factor_graph;
      timeval start, end;
      if (format == "ad3") {
        if (0 > LoadGraph(file_graph, &factor_graph)) continue;
      } else if (format == "uai") {
#if 0
        cout << "UAI format not implemented yet." << endl;
        assert(false);
#else
        if (convert_to_binary) {
          FactorGraph factor_graph_original;
          if (0 > LoadGraphUAI(file_graph, &factor_graph_original)) continue;
          factor_graph_original.ConvertToBinaryFactorGraph(&factor_graph);
        } else {
          if (0 > LoadGraphUAI(file_graph, &factor_graph)) continue;
          factor_graph.FixMultiVariablesWithoutFactors();
        }
#endif
      }
      cout << "Running " << niters << " iterations of "
           << algorithm << " (eta = "
           << eta << ")..." << endl;

      gettimeofday(&start, NULL);
      vector<double> posteriors;
      vector<double> additional_posteriors;
      double value;
      if (algorithm == "ad3") {
        factor_graph.SetEtaAD3(eta);
        factor_graph.AdaptEtaAD3(adapt_eta);
        factor_graph.SetMaxIterationsAD3(niters);
        factor_graph.SetResidualThresholdAD3(residual_threshold);
        if (exact) {
          factor_graph.SolveExactMAPWithAD3(&posteriors, &additional_posteriors,
                                         &value);
        } else {
          factor_graph.SolveLPMAPWithAD3(&posteriors, &additional_posteriors,
                                         &value);
        }
      } else if (algorithm == "psdd") {
        assert(!exact);
        factor_graph.SetEtaPSDD(eta);
        factor_graph.SetMaxIterationsPSDD(niters);
        factor_graph.SolveLPMAPWithPSDD(&posteriors, &additional_posteriors, &value);
      } else if (algorithm == "mplp") {
        cout << "MPLP is not implemented yet.";
        assert(false);
      } else {
        cout << "Unknown algorithm: " << algorithm << endl;
      }
      gettimeofday(&end, NULL);
      time_ddadmm += diff_ms(end,start);

#if 0
      gettimeofday(&start, NULL);
      vector<double> posteriors_relax;
      factor_graph.ComputeLPMAPWithAD3(&posteriors_relax, &value);
      gettimeofday(&end, NULL);
      time_ddadmm_relax += diff_ms(end,start);
#endif

#ifdef LPSOLVER_CPLEX
      gettimeofday(&start, NULL);
      vector<double> posteriors_cplex_relax;
      factor_graph.ComputeLPMAPWithCPLEX(&posteriors_cplex_relax,
                                         &additional_posteriors_cplex_relax,
                                         &value);
      gettimeofday(&end, NULL);
      time_cplex_relax += diff_ms(end,start);

      gettimeofday(&start, NULL);
      vector<double> posteriors_cplex_integer;
      factor_graph.ComputeLPMAPWithCPLEX(&posteriors_cplex_integer,
                                         &additional_posteriors_cplex_integer,
                                         &value);
      gettimeofday(&end, NULL);
      time_cplex_integer += diff_ms(end,start);
#endif

      if (file_posteriors.is_open()) {
        for (int i = 0; i < posteriors.size(); ++i) {
          file_posteriors << posteriors[i];
#ifdef LPSOLVER_CPLEX
          file_posteriors << "\t" << posteriors_cplex_relax[i]
                          << "\t" << posteriors_cplex_integer[i];
#endif
          file_posteriors << endl;
        }
        file_posteriors << endl;
        for (int i = 0; i < additional_posteriors.size(); ++i) {
          file_posteriors << additional_posteriors[i];
#ifdef LPSOLVER_CPLEX
          file_posteriors << "\t" << additional_posteriors_cplex_relax[i]
                          << "\t" << additional_posteriors_cplex_integer[i];
#endif
          file_posteriors << endl;
        }
        file_posteriors << endl;
      } else {
        cout << "Error: Could not open " << filename_posteriors << " for writing." << endl;
        return -1;
      }
    }
  } else {
    cout << "Error: Could not open " << filename_graph << " for reading." << endl;
    return -1;
  }
  file_graph.clear();
  file_graph.close();
  file_posteriors.flush();
  file_posteriors.clear();
  file_posteriors.close();

#if LPSOLVER_CPLEX
  cout << "Elapsed times: " << endl;
  cout << "AD3 relax: " << static_cast<double>(time_ddadmm_relax)/1000.0 
       << " sec." << endl; 
  cout << "AD3 integer: " << static_cast<double>(time_ddadmm)/1000.0 
       << " sec." << endl; 
  cout << "CPLEX relax: " << static_cast<double>(time_cplex_relax)/1000.0 
       << " sec." << endl; 
  cout << "CPLEX integer: " << static_cast<double>(time_cplex_integer)/1000.0 
       << " sec." << endl; 
#else
  cout << "Elapsed time: " << static_cast<double>(time_ddadmm)/1000.0 
       << " sec." << endl; 
#endif
  return 0;
}

int LoadGraph(ifstream &file_graph, 
              FactorGraph *factor_graph) {
  string line;

  // Read number of variables.
  getline(file_graph, line);
  //cout << line << endl;
  if (file_graph.eof()) return -1;
  TrimComments("#", &line);
  int num_variables = atoi(line.c_str());

  // Read number of factors.
  getline(file_graph, line);
  //cout << line << endl;
  TrimComments("#", &line);
  int num_factors = atoi(line.c_str());

  // Read variable log-potentials.
  vector<BinaryVariable*> variables(num_variables);
  for (int i = 0; i < num_variables; ++i) {
    getline(file_graph, line);
    TrimComments("#", &line);
    double log_potential = atof(line.c_str());
    BinaryVariable* variable = factor_graph->CreateBinaryVariable();
    variable->SetLogPotential(log_potential);
    variables[i] = variable;
  }

  // Read factors.
  int num_messages = 0;
  int num_factor_log_potentials = 0;
  for (int i = 0; i < num_factors; ++i) {
    getline(file_graph, line);
    TrimComments("#", &line);
    vector<string> fields;
    StringSplit(line, "\t ", &fields);

    // Read linked variables.
    int offset = 1;
    int num_links = atoi(fields[1].c_str());
    vector<BinaryVariable*> binary_variables(num_links);
    vector<bool> negated(num_links, false);
    ++offset;

    if (fields[0] == "PAIR" && num_links != 2) {
      cout << "Error: PAIR factor must be attached to 2 variables." << endl;
      return -1;
    }
    for (int j = 0; j < num_links; ++j) {
      int k = atoi(fields[offset+j].c_str());
      if (k < 0) {
        negated[j] = true;
        k = -k;
      }
      --k;
      binary_variables[j] = variables[k];
    }

    // Read factor type.
    Factor *factor;
    if (fields[0] == "XOR") {
      factor = factor_graph->CreateFactorXOR(binary_variables, negated);
    } else if (fields[0] == "XOROUT") {
      factor = factor_graph->CreateFactorXOROUT(binary_variables, negated);
    } else if (fields[0] == "ATMOSTONE") {
      factor = factor_graph->CreateFactorAtMostOne(binary_variables,
                                                   negated);
    } else if (fields[0] == "OR") {
      factor = factor_graph->CreateFactorOR(binary_variables, negated);
    } else if (fields[0] == "OROUT") {
      factor = factor_graph->CreateFactorOROUT(binary_variables, negated);
    } else if (fields[0] == "ANDOUT") {
      factor = factor_graph->CreateFactorANDOUT(binary_variables, negated);
    } else if (fields[0] == "PAIR") {
      // If it is a soft factor, read the factor log-potential.
      double log_potential = atof(fields[offset+num_links].c_str());
      //int r = num_variables + num_factor_log_potentials;
      ++num_factor_log_potentials;
      //static_cast<FactorPAIR*>(factor)->SetGlobalIndex(r);
      //static_cast<FactorPAIR*>(factor)->SetFactorLogPotential(log_potential);
      factor = factor_graph->CreateFactorPAIR(binary_variables, log_potential);
    } else if (fields[0] == "DENSE") {
      // Read the number of multi-variables.
      int num_multi_variables = atoi(fields[offset+num_links].c_str());
      // Read the number of states for each multi-variable.
      vector<MultiVariable*> multi_variables(num_multi_variables);
      int num_configurations = 1;
      int total_states = 0;
      for (int k = 0; k < num_multi_variables; ++k) {
        int num_states = atoi(fields[offset+num_links+1+k].c_str());
        num_configurations *= num_states;
        vector<BinaryVariable*> states(binary_variables.begin() + total_states,
                                       binary_variables.begin() + total_states +
                                         num_states);
        total_states += num_states;
        multi_variables[k] = factor_graph->CreateMultiVariable(states);
      }

      // Read the additional log-potentials.
      vector<double> additional_scores;
      for (int index = 0; index < num_configurations; ++index) {
        // Read the factor log-potential for this configuration.
        double log_potential = atof(fields[offset+num_links+1+num_multi_variables+index].c_str());
        additional_scores.push_back(log_potential);
      }

      // Create the factor and declare it.
      factor = new FactorDense;
      factor_graph->DeclareFactor(factor, binary_variables, true);
      static_cast<FactorDense*>(factor)->Initialize(multi_variables);
      factor->SetAdditionalLogPotentials(additional_scores);
      num_factor_log_potentials += additional_scores.size();
      cout << "Read dense factor." << endl;
    } else if (fields[0] == "SEQUENCE") {
      // Read the sequence length.
      int length = atoi(fields[offset+num_links].c_str());
      // Read the number of states for each position in the sequence.
      vector<int> num_states(length);
      int total_states = 0;
      for (int k = 0; k < length; ++k) {
        num_states[k] = atoi(fields[offset+num_links+1+k].c_str());
        total_states += num_states[k];
      }

      // Read the additional log-potentials.
      vector<double> additional_scores;
      int index = 0;
      for (int i = 0; i <= length; ++i) {
        // If i == 0, the previous state is the start symbol.
        int num_previous_states = (i > 0)? num_states[i - 1] : 1;
        // If i == length-1, the previous state is the final symbol.
        int num_current_states = (i < length)? num_states[i] : 1;
        for (int j = 0; j < num_previous_states; ++j) {
          for (int k = 0; k < num_current_states; ++k) {
            double log_potential = atof(fields[offset+num_links+1+length+index].c_str());
            additional_scores.push_back(log_potential);
            ++index;
          }
        }
      }
      if (fields.size() != offset+num_links+1+length+index) {
        cout << fields.size() << " "
             << offset+num_links+1+length+index;
        assert(false);
      }

      // Create the factor and declare it.
      factor = new FactorSequence;
      factor_graph->DeclareFactor(factor, binary_variables, true);
      static_cast<FactorSequence*>(factor)->Initialize(num_states);
      factor->SetAdditionalLogPotentials(additional_scores);
      num_factor_log_potentials += additional_scores.size();
      cout << "Read sequence factor." << endl;
    } else if (fields[0] == "ARBORESCENCE") {
      // Read the sentence length.
      int sentence_length = atoi(fields[offset+num_links].c_str());
      // Read the arcs.
      vector<Arc*> arcs(binary_variables.size());
      for (int r = 0; r < binary_variables.size(); ++r) {
        //cout << fields.size() << " " << offset+num_links+2*r+1 << endl;
        int h = atoi(fields[offset+num_links+1+2*r].c_str());
        int m = atoi(fields[offset+num_links+1+2*r+1].c_str());
        Arc *arc = new Arc(h, m);
        arcs[r] = arc;
      }
      factor = new FactorTree;
      factor_graph->DeclareFactor(factor, binary_variables, true);
      static_cast<FactorTree*>(factor)->Initialize(sentence_length, arcs);
      for (int r = 0; r < arcs.size(); ++r) {
        delete arcs[r];
      }
      cout << "Read tree factor." << endl;
    } else if (fields[0] == "HEAD_AUTOMATON") {
      // Read the length of the automaton.
      int length = binary_variables.size() + 1;
      vector<vector<int> > index_siblings(length, vector<int>(length+1, -1));
      int total = 0;
      vector<Sibling*> siblings;
      vector<double> additional_scores;
      for (int m = 0; m < length; ++m) {
        for (int s = m+1; s <= length; ++s) {
          // Create a fake sibling.
          Sibling *sibling = new Sibling(0, m, s);
          siblings.push_back(sibling);
          // Read the sibling log-potential.
          double log_potential = atof(fields[offset+num_links+total].c_str());
          additional_scores.push_back(log_potential);
          ++total;
        }
      }
      factor = new FactorHeadAutomaton;
      factor_graph->DeclareFactor(factor, binary_variables, true);
      static_cast<FactorHeadAutomaton*>(factor)->Initialize(length, siblings);
      for (int r = 0; r < siblings.size(); ++r) {
        delete siblings[r];
      }
      factor->SetAdditionalLogPotentials(additional_scores);
      num_factor_log_potentials += additional_scores.size();
      cout << "Read head automaton factor." << endl;
    } else if (fields[0] == "GRANDPARENT_HEAD_AUTOMATON") {
      // Read the number of grandparents.
      int num_grandparents = atoi(fields[offset+num_links].c_str());
      // Read the length of the automaton.
      int length = binary_variables.size() + 1 - num_grandparents;
      vector<vector<int> > index_siblings(length, vector<int>(length+1, -1));
      int total = 0;
      vector<Grandparent*> grandparents;
      vector<double> additional_scores;
      for (int g = 0; g < num_grandparents; ++g) {
        for (int m = 1; m < length; ++m) {
          // Create a fake grandparent.
          Grandparent *grandparent = new Grandparent(g, 0, m);
          grandparents.push_back(grandparent);
          // Read the sibling log-potential.
          double log_potential = atof(fields[offset+num_links+1+total].c_str());
          additional_scores.push_back(log_potential);
          ++total;
        }
      }
      vector<Sibling*> siblings;
      for (int m = 0; m < length; ++m) {
        for (int s = m+1; s <= length; ++s) {
          // Create a fake sibling.
          Sibling *sibling = new Sibling(0, m, s);
          siblings.push_back(sibling);
          // Read the sibling log-potential.
          double log_potential = atof(fields[offset+num_links+1+total].c_str());
          additional_scores.push_back(log_potential);
          ++total;
        }
      }
      factor = new FactorGrandparentHeadAutomaton;
      factor_graph->DeclareFactor(factor, binary_variables, true);
      static_cast<FactorGrandparentHeadAutomaton*>(factor)->
        Initialize(length, num_grandparents, siblings, grandparents);
      for (int r = 0; r < grandparents.size(); ++r) {
        delete grandparents[r];
      }
      for (int r = 0; r < siblings.size(); ++r) {
        delete siblings[r];
      }
      factor->SetAdditionalLogPotentials(additional_scores);
      num_factor_log_potentials += additional_scores.size();
      cout << "Read grandparent head automaton factor." << endl;
    } else {
      cout << "Unknown factor type: " << fields[0] << endl;
      return -1;
    }
  }

  // Read blank line.
  getline(file_graph, line);

  cout << "Read " << num_variables << " variables and "
       << num_factors << " factors." << endl;

  //factor_graph->Initialize(variables, factors, num_messages);

  return 0;
}

// This loads a graph in the format of PIC 2011.
int LoadGraphUAI(ifstream &file_graph, 
                 FactorGraph *factor_graph) {
  string line = "";

  // Read header.
  while (line == "") {
    getline(file_graph, line);
    if (file_graph.eof()) return -1;
    TrimComments("#", &line);
    Trim("\t ", &line);
  }
  if (line != "MARKOV") {
    cout << "Wrong header: " << line << endl;
    return -1;
  }

  int num_factor_log_potentials = 0;

  // Read number of multi-variables.
  getline(file_graph, line);
  TrimComments("#", &line);
  int num_multi_variables = atoi(line.c_str());
  vector<MultiVariable*> multi_variables(num_multi_variables);

  // Read cardinality of each multi-variable.
  getline(file_graph, line);
  TrimComments("#", &line);
  vector<string> fields;
  StringSplit(line, "\t ", &fields);
  assert(fields.size() == num_multi_variables);
  for (int i = 0; i < num_multi_variables; ++i) {
    int num_states = atoi(fields[i].c_str());
    MultiVariable* multi_variable = 
      factor_graph->CreateMultiVariable(num_states);
    multi_variables[i] = multi_variable;
  }

  // Read number of factors (includes unary factors).
  getline(file_graph, line);
  TrimComments("#", &line);
  int num_factors = atoi(line.c_str());

  // Read factors (just the structure).
  vector<Factor*> factors(num_factors);
  vector<MultiVariable*> unary_factors(num_factors);
  for (int i = 0; i < num_factors; ++i) {
    getline(file_graph, line);
    TrimComments("#", &line);
    fields.clear();
    StringSplit(line, "\t ", &fields);

    // Read linked multi-variables.
    int num_links = atoi(fields[0].c_str());
    int offset = 1;
    assert(num_links == fields.size() - offset);
    if (num_links == 1) { 
      // Unary factor; in our formalism this is just a multi-variable.
      int k = atoi(fields[offset].c_str());
      unary_factors[i] = multi_variables[k];
    } else {
      vector<MultiVariable*> multi_variables_local(num_links);
      for (int j = 0; j < num_links; ++j) {
        int k = atoi(fields[offset + j].c_str());
        multi_variables_local[j] = multi_variables[k];
      }
      // For now, set an empty vector of additional log potentials.
      vector<double> additional_log_potentials;
      Factor *factor = 
        factor_graph->CreateFactorDense(multi_variables_local,
                                        additional_log_potentials);
      factors[i] = factor;
    }
  }

  // Read factors (the log-potentials).
  // IMPORTANT: the scores in the UAI files are potentials (not log-potentials!)
  for (int i = 0; i < num_factors; ++i) {
    Factor *factor = factors[i];
    line = "";
    while (line == "") {
      getline(file_graph, line);
      TrimComments("#", &line);
      Trim(" \t", &line);
    }
    int num_configurations = atoi(line.c_str());
    if (factor == NULL) { 
      // Unary factor; in our formalism this is just a multi-variable.
      assert(unary_factors[i] != NULL);
      MultiVariable *multi_variable = unary_factors[i];
      int index = 0;
      assert(num_configurations == multi_variable->GetNumStates());
      while (index < num_configurations) {
        getline(file_graph, line);
        TrimComments("#", &line);
        Trim(" \t", &line);
        fields.clear();
        StringSplit(line, "\t ", &fields);
        for (int j = 0; j < fields.size(); ++j) {
          double log_potential = LOG_STABLE(atof(fields[j].c_str()));
          multi_variable->SetLogPotential(index, log_potential);
          assert(index < num_configurations);
          ++index;
        }
      }
    } else {
      int num_links = static_cast<FactorDense*>(factor)->
        GetNumMultiVariables();
      int index = 0;
      assert(num_configurations == 
             static_cast<FactorDense*>(factor)->GetNumConfigurations());

      //int r = factor_graph->GetNumVariables() + num_factor_log_potentials;
      num_factor_log_potentials += num_configurations;
      //static_cast<FactorMultiDense*>(factor)->SetFirstGlobalIndex(r);
      vector<double> additional_log_potentials(num_configurations);
      while (index < num_configurations) {
        getline(file_graph, line);
        TrimComments("#", &line);
        Trim(" \t", &line);
        fields.clear();
        StringSplit(line, "\t ", &fields);
        for (int j = 0; j < fields.size(); ++j) {
          double log_potential = LOG_STABLE(atof(fields[j].c_str()));
          additional_log_potentials[index] = log_potential;
          assert(index < num_configurations);
          ++index;
        }
      }
      factor->SetAdditionalLogPotentials(additional_log_potentials);
    }
  }

  cout << "Read " << num_multi_variables << " multi-variables and "
       << num_factors << " factors." << endl;

  return 0;
}
