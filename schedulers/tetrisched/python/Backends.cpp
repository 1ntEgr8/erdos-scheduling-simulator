#include <pybind11/pybind11.h>

#include "tetrisched/CPLEXSolver.hpp"

namespace py = pybind11;

#ifdef _TETRISCHED_WITH_CPLEX_
void defineCPLEXBackend(py::module_& tetrisched_m) {
  py::class_<tetrisched::CPLEXSolver>(tetrisched_m, "CPLEXSolver")
      .def(py::init<>())
      .def("getModel", &tetrisched::CPLEXSolver::getModel)
      .def("translateModel", &tetrisched::CPLEXSolver::translateModel)
      .def("exportModel", &tetrisched::CPLEXSolver::exportModel)
      .def("solveModel", &tetrisched::CPLEXSolver::solveModel);
}
#endif  //_TETRISCHED_WITH_CPLEX_
