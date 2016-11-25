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

#include "GenericFactor.h"
#include "Utils.h"
#define EIGEN
#ifdef EIGEN
#include <../Eigen/Eigenvalues>
#else
#include "lapacke/lapacke.h"
#endif
#include <math.h>
#include <limits>

namespace AD3 {

#ifdef PRINT_INVERSION_STATS
static int num_inversions_from_scratch = 0;
static int num_inversions_incremental = 0;
static int num_eigenvalue_computations = 0;
#endif

bool GenericFactor::InvertAfterInsertion(
    const vector<Configuration> &active_set,
    const Configuration &inserted_element) {
#ifdef PRINT_INVERSION_STATS
  ++num_inversions_incremental;
  if (0 == num_inversions_incremental % 10000) {
    cout << "Number of incremental inversions: "
         << num_inversions_incremental << endl;
    cout << "Number of standard inversions: "
         << num_inversions_from_scratch << endl;
    cout << "Number of eigenvalue computations: "
         << num_eigenvalue_computations << endl;
  }
#endif

  vector<double> inverse_A = inverse_A_;
  int size_A = active_set.size() + 1;
  vector<double> r(size_A);

  r[0] = 1.0;
  for (int i = 0; i < active_set.size(); ++i) {
    // Count how many variable values the new assignment 
    // have in common with the i-th assignment.
    int num_common_values = CountCommonValues(active_set[i], inserted_element);
    r[i+1] = static_cast<double>(num_common_values);
  }

  double r0 = static_cast<double>(CountCommonValues(
      inserted_element, inserted_element));
  double s = r0;
  for (int i = 0; i < size_A; ++i) {
    if (r[i] == 0.0) continue;
    s -= r[i] * r[i] * inverse_A[i * size_A + i];
    for (int j = i+1; j < size_A; ++j) {
      if (r[j] == 0.0) continue;
      s -= 2 * r[i] * r[j] * inverse_A[i * size_A + j];
    }
  }

  if (NEARLY_ZERO_TOL(s, 1e-9)) {
    if (verbosity_ > 2) {
      cout << "Warning: updated matrix will become singular after insertion."
           << endl;
    }
    return false;
  }

  double invs = 1.0 / s;
  vector<double> d(size_A, 0.0);
  for (int i = 0; i < size_A; ++i) {
    if (r[i] == 0.0) continue;
    for (int j = 0; j < size_A; ++j) {
      d[j] += inverse_A[i * size_A + j] * r[i];
    }
  }

  int size_A_after = size_A + 1;
  inverse_A_.resize(size_A_after * size_A_after);
  for (int i = 0; i < size_A; ++i) {
    for (int j = 0; j < size_A; ++j) {
      inverse_A_[i * size_A_after + j] = inverse_A[i * size_A + j] +
          invs * d[i] * d[j];
    }
    inverse_A_[i * size_A_after + size_A] = -invs * d[i];
    inverse_A_[size_A * size_A_after + i] = -invs * d[i];
  }
  inverse_A_[size_A * size_A_after + size_A] = invs;

  return true;
}

void GenericFactor::InvertAfterRemoval(const vector<Configuration> &active_set,
                                       int removed_index) {
#ifdef PRINT_INVERSION_STATS
  ++num_inversions_incremental;
  if (0 == num_inversions_incremental % 10000) {
    cout << "Number of incremental inversions: "
         << num_inversions_incremental << endl;
    cout << "Number of standard inversions: "
         << num_inversions_from_scratch << endl;
    cout << "Number of eigenvalue computations: "
         << num_eigenvalue_computations << endl;
  }
#endif

  vector<double> inverse_A = inverse_A_;
  int size_A = active_set.size() + 1;
  vector<double> r(size_A);

  ++removed_index; // Index in A has an offset of 1.
  double invs = inverse_A[removed_index * size_A + removed_index];
  assert(!NEARLY_ZERO_TOL(invs, 1e-12));
  double s = 1.0 / invs;
  vector<double> d(size_A - 1, 0.0);
  int k = 0;
  for (int i = 0; i < size_A; ++i) {
    if (i == removed_index) continue;
    d[k] = -s * inverse_A[removed_index * size_A + i];
    ++k;
  }

  int size_A_after = size_A - 1;
  inverse_A_.resize(size_A_after * size_A_after);
  k = 0;
  for (int i = 0; i < size_A; ++i) {
    if (i == removed_index) continue;
    int l = 0;
    for (int j = 0; j < size_A; ++j) {
      if (j == removed_index) continue;
      inverse_A_[k * size_A_after + l] = inverse_A[i * size_A + j] -
          invs * d[k] * d[l];
      ++l;
    }
    ++k;
  }
}

// Compute Mnz'*Mnz
void GenericFactor::ComputeActiveSetSimilarities(
    const vector<Configuration> &active_set,
    vector<double> *similarities) {
  int size = active_set.size();

  // Compute similarity matrix.
  similarities->resize(size * size);
  (*similarities)[0] = 0.0;
  for (int i = 0; i < active_set.size(); ++i) {
    (*similarities)[i*size + i] = static_cast<double>(
        CountCommonValues(active_set[i], active_set[i]));
    for (int j = i+1; j < active_set.size(); ++j) {
      // Count how many variable values the i-th and j-th 
      // assignments have in common.
      int num_common_values = CountCommonValues(active_set[i], active_set[j]);
      (*similarities)[i*size + j] = num_common_values;
      (*similarities)[j*size + i] = num_common_values;
    }
  }
}

// Compute eigendecomposition of M'*M.
// Remark: overwrite similarities with the eigenvectors.
void GenericFactor::EigenDecompose(vector<double> *similarities,
                            vector<double> *eigenvalues) {
#ifdef PRINT_INVERSION_STATS
  ++num_eigenvalue_computations;
  if (0 == num_eigenvalue_computations % 10000) {
    cout << "Number of incremental inversions: "
         << num_inversions_incremental << endl;
    cout << "Number of standard inversions: "
         << num_inversions_from_scratch << endl;
    cout << "Number of eigenvalue computations: "
         << num_eigenvalue_computations << endl;
  }
#endif

  int size = sqrt(similarities->size());
#ifdef EIGEN
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es;
  Eigen::MatrixXd sim(size, size);
  int t = 0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      sim(i, j) = (*similarities)[t];
      ++t;
    }
  }
  es.compute(sim);
  const Eigen::VectorXd &eigvals = es.eigenvalues(); 
  eigenvalues->resize(size);
  for (int i = 0; i < size; ++i) {
    (*eigenvalues)[i] = eigvals[i];
  }
  const Eigen::MatrixXd &eigvectors = es.eigenvectors().transpose();
  t = 0;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      (*similarities)[t] = eigvectors(i, j);
      ++t;
    }
  }
#else
  lapack_int info;
  eigenvalues->resize(size);
  info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', 
                       size,
                       &(*similarities)[0],
                       size,
                       &(*eigenvalues)[0]);
#endif
}

// Compute inverse of A from scratch.
// Uses eigendecomposition of M'*M.
void GenericFactor::Invert(const vector<double> &eigenvalues,
                           const vector<double> &eigenvectors) {
#ifdef PRINT_INVERSION_STATS
  ++num_inversions_from_scratch;
  if (0 == num_inversions_from_scratch % 10000) {
    cout << "Number of incremental inversions: "
         << num_inversions_incremental << endl;
    cout << "Number of standard inversions: "
         << num_inversions_from_scratch << endl;
    cout << "Number of eigenvalue computations: "
         << num_eigenvalue_computations << endl;
  }
#endif
  int size = eigenvalues.size();
  int size_A = size + 1;
  inverse_A_.assign(size_A * size_A, 0.0);
  for (int i = 0; i < size; ++i) {
    double s = 1.0 / eigenvalues[i];
    const double *v = &eigenvectors[0] + i * size;
    for (int j = 0; j < size; ++j) {
      for (int k = j; k < size; ++k) {
        inverse_A_[(1+j)*size_A + (1+k)] += s * v[j] * v[k];
      }
    }
  }
  double s = 0.0;
  vector<double> d(size, 0.0);
  for (int j = 1; j <= size; ++j) {
    s -= inverse_A_[j*size_A + j];
    d[j-1] += inverse_A_[j*size_A + j];
    for (int k = j+1; k <= size; ++k) {
      inverse_A_[k*size_A + j] = inverse_A_[j*size_A + k];
      s -= 2*inverse_A_[j*size_A + k];
      d[j-1] += inverse_A_[j*size_A + k];
      d[k-1] += inverse_A_[j*size_A + k];
    }
  }
    
  double invs = 1.0 / s;
  inverse_A_[0] = invs;
  for (int j = 1; j <= size; ++j) {
    inverse_A_[j*size_A] = -invs * d[j-1];
    inverse_A_[j] = inverse_A_[j*size_A];
    inverse_A_[j*size_A + j] += invs * d[j-1] * d[j-1];
    for (int k = j+1; k <= size; ++k) {
      inverse_A_[j*size_A + k] += invs * d[j-1] * d[k-1];
      inverse_A_[k*size_A + j] = inverse_A_[j*size_A + k];
    }
  }
}

// Checks if M'*M is singular.
// If so, asserts that the null space is 1-dimensional 
// and returns a basis of the null space.
bool GenericFactor::IsSingular(vector<double> &eigenvalues,
                        vector<double> &eigenvectors,
                        vector<double> *null_space_basis) {
  int size = eigenvalues.size();
  int zero_eigenvalue = -1;
  for (int i = 0; i < size; ++i) {
    if (eigenvalues[i] < 1e-12) {
      if (zero_eigenvalue >= 0) {
        cout << eigenvalues[i] << " " << eigenvalues[zero_eigenvalue] << endl;
        assert(false);
      }
      zero_eigenvalue = i;
    }
  }
  if (zero_eigenvalue < 0) return false;
  if (null_space_basis) {
    null_space_basis->assign(eigenvectors.begin() +
                             zero_eigenvalue * size,
                             eigenvectors.begin() +
                             (1 + zero_eigenvalue) * size);
  }
  return true;
}

void GenericFactor::SolveQP(const vector<double> &variable_log_potentials,
                            const vector<double> &additional_log_potentials,
                            vector<double> *variable_posteriors,
                            vector<double> *additional_posteriors) {
  // Initialize the active set.
  if (active_set_.size() == 0) {
    variable_posteriors->resize(variable_log_potentials.size());
    additional_posteriors->resize(additional_log_potentials.size());
    distribution_.clear();
    // Initialize by solving the LP, discarding the quadratic
    // term.
    Configuration configuration = CreateConfiguration();
    double value;
    Maximize(variable_log_potentials,
             additional_log_potentials,
             configuration,
             &value);
    active_set_.push_back(configuration);
    distribution_.push_back(1.0);

    // Initialize inv(A) as [-M,1;1,0].
    inverse_A_.resize(4);
    inverse_A_[0] = static_cast<double>(
        -CountCommonValues(configuration, configuration));
    inverse_A_[1] = 1;
    inverse_A_[2] = 1;
    inverse_A_[3] = 0;
  }

  bool changed_active_set = true;
  vector<double> z;
  int num_max_iterations = num_max_iterations_QP_;
  double tau = 0;
  for (int iter = 0; iter < num_max_iterations; ++iter) {
    bool same_as_before = true;
    bool unbounded = false;
    if (changed_active_set) {
      // Recompute vector b.
      vector<double> b(active_set_.size() + 1, 0.0);
      b[0] = 1.0;
      for (int i = 0; i < active_set_.size(); ++i) {
        const Configuration &configuration = active_set_[i];
        double score;
        Evaluate(variable_log_potentials,
                 additional_log_potentials,
                 configuration,
                 &score);
        b[i+1] = score;
      }

      // Solve the system Az = b.
      z.resize(active_set_.size());
      int size_A = active_set_.size() + 1;
      for (int i = 0; i < active_set_.size(); ++i) {
        z[i] = 0.0;
        for (int j = 0; j < size_A; ++j) {
          z[i] += inverse_A_[(i+1) * size_A + j] * b[j];
        }
      }
      tau = 0.0;
      for (int j = 0; j < size_A; ++j) {
        tau += inverse_A_[j] * b[j];
      }

      same_as_before = false;
    }

    if (same_as_before) {
      // Compute the variable marginals from the full distribution
      // stored in z.
      ComputeMarginalsFromSparseDistribution(active_set_,
                                             z,
                                             variable_posteriors,
                                             additional_posteriors);

      // Get the most violated constraint
      // (by calling the black box that computes the MAP).
      vector<double> scores = variable_log_potentials;
      for (int i = 0; i < scores.size(); ++i) {
        scores[i] -= (*variable_posteriors)[i];
      }
      Configuration configuration = CreateConfiguration();
      double value;
      Maximize(scores,
               additional_log_potentials,
               configuration,
               &value);

      double very_small_threshold = 1e-9;
      if (value <= tau + very_small_threshold) { // value <= tau.
        // We have found the solution;
        // the distribution, active set, and inv(A) are cached for the next round.
        DeleteConfiguration(configuration);
        return;
      } else {
        for (int k = 0; k < active_set_.size(); ++k) {
          // This is expensive and should just be a sanity check.
          // However, in practice, numerical issues force an already existing
          // configuration to try to be added. Therefore, we always check
          // if a configuration already exists before inserting it.
          // If it does, that means the active set method converged to a
          // solution (but numerical issues had prevented us to see it.)
          if (SameConfiguration(active_set_[k], configuration)) {
            if (verbosity_ > 2) {
              cout << "Warning: value - tau = "
                   << value - tau << " " << value << " " << tau
                   << endl;
            }
            // We have found the solution;
            // the distribution, active set, and inv(A)
            // are cached for the next round.
            DeleteConfiguration(configuration);

            // Just in case, clean the cache.
            // This may prevent eventual numerical problems in the future.
            for (int j = 0; j < active_set_.size(); ++j) {
              if (j == k) continue; // This configuration was deleted already.
              DeleteConfiguration(active_set_[j]);
            }
            active_set_.clear();
            inverse_A_.clear();
            distribution_.clear();

            // Return.
            return;
          }
        }
        z.push_back(0.0);
        distribution_ = z;

        // Update inv(A).
        bool singular = !InvertAfterInsertion(active_set_, configuration);
        if (singular) {
          // If adding a new configuration causes the matrix to be singular,
          // don't just add it. Instead, look for a configuration in the null
          // space and remove it before inserting the new one.
          // Right now, if more than one such configuration exists, we just
          // remove the first one we find. There's a chance this could cause
          // some cyclic behaviour. If that is the case, we should randomize
          // this choice.
          // Note: This step is expensive and requires an eigendecomposition.
          // TODO: I think there is a graph interpretation for this problem.
          // Maybe some specialized graph algorithm is cheaper than doing
          // the eigendecomposition.
          vector<double> similarities(active_set_.size() * active_set_.size());
          ComputeActiveSetSimilarities(active_set_, &similarities);
          vector<double> padded_similarities((active_set_.size()+2) * 
                                             (active_set_.size()+2), 1.0);
          for (int i = 0; i < active_set_.size(); ++i) {
            for (int j = 0; j < active_set_.size(); ++j) {
              padded_similarities[(i+1)*(active_set_.size()+2) + (j+1)] =
                  similarities[i*active_set_.size() + j];
            }
          }
          padded_similarities[0] = 0.0;
          for (int i = 0; i < active_set_.size(); ++i) {
            double value = static_cast<double>(
                CountCommonValues(configuration, active_set_[i]));
            padded_similarities[(i+1)*(active_set_.size()+2) +
                                (active_set_.size()+1)] = value;
            padded_similarities[(active_set_.size()+1)*(active_set_.size()+2) +
                                (i+1)] = value;
          }
          double value = static_cast<double>(
              CountCommonValues(configuration, configuration));
          padded_similarities[(active_set_.size()+1)*(active_set_.size()+2) +
                              (active_set_.size()+1)] = value;

          vector<double> eigenvalues(active_set_.size()+2);
          EigenDecompose(&padded_similarities, &eigenvalues);
          int zero_eigenvalue = -1;
          for (int i = 0; i < active_set_.size()+2; ++i) {
            if (NEARLY_EQ_TOL(eigenvalues[i], 0.0, 1e-9)) {
              if (zero_eigenvalue >= 0) {
                // If this happens, something failed. Maybe a numerical problem
                // may cause this. In that case, just give up, clean the cache
                // and return. Hopefully the next iteration will fix it.
                cout << "Multiple zero eigenvalues: "
                     << eigenvalues[zero_eigenvalue] << " and "
                     << eigenvalues[i] << endl;
                cout << "Warning: Giving up." << endl;
                // Clean the cache.
                for (int j = 0; j < active_set_.size(); ++j) {
                  DeleteConfiguration(active_set_[j]);
                }
                active_set_.clear();
                inverse_A_.clear();
                distribution_.clear();
                return;
              }
              zero_eigenvalue = i;
            }
          }
          assert(zero_eigenvalue >= 0);
          vector<int> configurations_to_remove;
          for (int j = 1; j < active_set_.size()+1; ++j) {
            double value = padded_similarities[zero_eigenvalue*(active_set_.size()+2) + j];
            if (!NEARLY_EQ_TOL(value, 0.0, 1e-9)) {
              configurations_to_remove.push_back(j-1);
            }
          }
          if (verbosity_ > 2) {
            cout << "Pick a configuration to remove (" << configurations_to_remove.size()
                 << " out of " << active_set_.size() << ")." << endl;
          }

          assert(configurations_to_remove.size() >= 1);
          int j = configurations_to_remove[0];

          // Update inv(A).
          InvertAfterRemoval(active_set_, j);

          // Remove blocking constraint from the active set.
          DeleteConfiguration(active_set_[j]); // Delete configutation.
          active_set_.erase(active_set_.begin() + j);

          singular = !InvertAfterInsertion(active_set_, configuration);
          assert(!singular);
        }

        // Insert configuration to active set.
        if (verbosity_ > 2) {
          cout << "Inserted one element to the active set (iteration "
               << iter << ")." << endl;
        }
        active_set_.push_back(configuration);
        changed_active_set = true;
      }      
    } else {
      // Solution has changed from the previous iteration.
      // Look for blocking constraints.
      int blocking = -1;
      bool exist_blocking = false;
      double alpha = 1.0;
      for (int i = 0; i < active_set_.size(); ++i) {
        assert(distribution_[i] >= -1e-12);
        if (z[i] >= distribution_[i]) continue;
        if (z[i] < 0) exist_blocking = true;
        double tmp = distribution_[i] / (distribution_[i] - z[i]);
        if (blocking < 0 || tmp < alpha) {
          alpha = tmp;
          blocking = i;
        }
      }

      if (!exist_blocking) {
        // No blocking constraints.
        assert(!unbounded);
        distribution_ = z;
        alpha = 1.0;
        changed_active_set = false;
      } else {
        if (alpha > 1.0 && !unbounded) alpha = 1.0;
        // Interpolate between factor_posteriors_[i] and z.
        if (alpha == 1.0) {
          distribution_ = z;
        } else {
          for (int i = 0; i < active_set_.size(); ++i) {
            z[i] = (1 - alpha) * distribution_[i] + alpha * z[i];
            distribution_[i] = z[i];
          }
        }

        // Update inv(A).
        InvertAfterRemoval(active_set_, blocking);

        // Remove blocking constraint from the active set.
        if (verbosity_ > 2) {
          cout << "Removed one element to the active set (iteration "
               << iter << ")." << endl;
        }

        DeleteConfiguration(active_set_[blocking]); // Delete configutation.
        active_set_.erase(active_set_.begin() + blocking);

        z.erase(z.begin() + blocking);
        distribution_.erase(distribution_.begin() + blocking);
        changed_active_set = true;
        for (int i = 0; i < distribution_.size(); ++i) {
          assert(distribution_[i] > -1e-16);
        }
      }
    }
  }

  // Maximum number of iterations reached.
  // Return the best existing solution by computing the variable marginals 
  // from the full distribution stored in z.
  //assert(false);
  ComputeMarginalsFromSparseDistribution(active_set_,
                                         z,
                                         variable_posteriors,
                                         additional_posteriors);
}

}  // namespace AD3
