from libcpp.vector cimport vector
from libcpp cimport bool

# get the classes from the c++ headers

cdef extern from "ad3/Factor.h" namespace "AD3":
    cdef cppclass BinaryVariable:
        BinaryVariable()
        double GetLogPotential()
        void SetLogPotential(double log_potential)

    cdef cppclass Factor:
        Factor()

cdef extern from "ad3/MultiVariable.h" namespace "AD3":
    cdef cppclass MultiVariable:
        int GetNumStates()
        double GetLogPotential(int i)
        void SetLogPotential(int i, double log_potential)


cdef extern from "ad3/FactorGraph.h" namespace "AD3":
    cdef cppclass FactorGraph:
        FactorGraph()
        void SetVerbosity(int verbosity)
        void SetEtaAD3(double eta)
        void AdaptEtaAD3(bool adapt)
        void SetMaxIterationsAD3(int max_iterations)
        void FixMultiVariablesWithoutFactors()
        int SolveLPMAPWithAD3(vector[double]* posteriors,
                              vector[double]* additional_posteriors,
                              double* value)
        int SolveExactMAPWithAD3(vector[double] *posteriors,
                                 vector[double] *additional_posteriors, 
                                 double *value)

        BinaryVariable * CreateBinaryVariable()
        MultiVariable * CreateMultiVariable(int num_states)
        Factor * CreateFactorDense(vector[MultiVariable*] multi_variables,
                                   vector[double] additional_log_potentials,
                                   bool owned_by_graph)


# wrap them into python extension types
cdef class PBinaryVariable:
    cdef BinaryVariable *thisptr
    def __cinit__(self, allocate=True):
        if allocate:
            self.thisptr = new BinaryVariable()

    def __dealloc__(self):
        del self.thisptr

    def get_log_potential(self):
        return self.thisptr.GetLogPotential()

    def set_log_potential(self, double log_potential):
        self.thisptr.SetLogPotential(log_potential)



cdef class PMultiVariable:
    cdef MultiVariable *thisptr
    cdef bool allocate
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new MultiVariable()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def get_log_potential(self, int i):
        return self.thisptr.GetLogPotential(i)

    def set_log_potential(self, int i, double log_potential):
        self.thisptr.SetLogPotential(i, log_potential)


cdef class PFactorGraph:
    cdef FactorGraph *thisptr
    def __cinit__(self):
        self.thisptr = new FactorGraph()

    def __dealloc__(self):
        del self.thisptr

    def set_verbosity(self, int verbosity):
        self.thisptr.SetVerbosity(verbosity)

    def create_multi_variable(self, int num_states):
        cdef MultiVariable * mult =  self.thisptr.CreateMultiVariable(num_states)
        pmult = PMultiVariable(allocate=False)
        pmult.thisptr = mult
        return pmult

    def fix_multi_variables_without_factors(self):
        self.thisptr.FixMultiVariablesWithoutFactors()

    def set_eta_ad3(self, double eta):
        self.thisptr.SetEtaAD3(eta)

    def adapt_eta_ad3(self, bool adapt):
        self.thisptr.AdaptEtaAD3(adapt)
    
    def set_max_iterations_ad3(self, int max_iterations):
        self.thisptr.SetMaxIterationsAD3(max_iterations)

    def solve_lp_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveLPMAPWithAD3(&posteriors,
                                                       &additional_posteriors,
                                                       &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors, solver_status

    def solve_exact_map_ad3(self):
        cdef vector[double] posteriors
        cdef vector[double] additional_posteriors
        cdef double value
        cdef int solver_status
        solver_status = self.thisptr.SolveExactMAPWithAD3(&posteriors,
                                                       &additional_posteriors,
                                                       &value)
        p_posteriors, p_additional_posteriors = [], []
        cdef size_t i
        for i in range(posteriors.size()):
            p_posteriors.append(posteriors[i])
        for i in range(additional_posteriors.size()):
            p_additional_posteriors.append(additional_posteriors[i])

        return value, p_posteriors, p_additional_posteriors, solver_status

    def create_factor_dense(self,  p_multi_variables, p_additional_log_potentials, bool owned_by_graph=True):
        cdef vector[MultiVariable*] multi_variables
        cdef PMultiVariable blub
        for var in p_multi_variables:
            blub = var
            multi_variables.push_back(<MultiVariable*>blub.thisptr)

        cdef vector[double] additional_log_potentials
        for potential in p_additional_log_potentials:
            additional_log_potentials.push_back(potential)
        self.thisptr.CreateFactorDense(multi_variables, additional_log_potentials, owned_by_graph)

