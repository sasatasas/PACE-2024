#include <cstdio>
#include <iostream>
#include <map>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "ortools/linear_solver/linear_solver.h"

#define _pair std::pair<uint32_t, uint32_t>

std::vector<uint32_t> MILP1(const size_t n, const size_t m,
                            const std::vector<std::vector<uint32_t> >& NB) {
  std::unique_ptr<operations_research::MPSolver> solver(
      operations_research::MPSolver::CreateSolver("SCIP"));

  const double INF = solver->infinity();

  std::vector<std::vector<bool> > q1(n, std::vector<bool>(n));

  std::vector<std::vector<operations_research::MPVariable*> > q2(
      m, std::vector<operations_research::MPVariable*>(m));

  std::map<
      std::pair<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t> >,
      operations_research::MPVariable*>
      C;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      q1[i][j] = 1;
    }
  }

  for (size_t i = 0; i < m - 1; ++i) {
    for (size_t j = i + 1; j < m; ++j) {
      // q2
      q2[i][j] = solver->MakeBoolVar("q_" + std::to_string(i) + "_" +
                                     std::to_string(j));

      // C
      for (size_t k = 0; k < n; ++k) {
        for (size_t l = 0; l < n; ++l) {

          if (k == l) continue;

          C[{{k, l}, {i, j}}] = solver->MakeBoolVar(
              "C_" + std::to_string(k) + "_" + std::to_string(l) + "_" +
              std::to_string(i) + "_" + std::to_string(j));
        }
      }
    }
  }

  operations_research::MPObjective* obj = solver->MutableObjective();
  uint32_t offset = 0;
  for (size_t i = 0; i < m - 1; ++i) {
    for (size_t j = i + 1; j < m; ++j) {
      for (auto& k : NB[i]) {
        for (auto& l : NB[j]) {
          if (k == l) continue;

          operations_research::MPConstraint* constr1 =
              solver->MakeRowConstraint(
                  -INF, q1[k][l],
                  "c1_" + std::to_string(k) + "_" + std::to_string(l) + "_" +
                      std::to_string(i) + "_" + std::to_string(j));

          constr1->SetCoefficient(C[{{k, l}, {i, j}}], 1.);

          // mb не в том форе
          operations_research::MPConstraint* constr2 =
              solver->MakeRowConstraint(
                  -INF, 0.,
                  "c2_" + std::to_string(k) + "_" + std::to_string(l) + "_" +
                      std::to_string(i) + "_" + std::to_string(j));

          constr2->SetCoefficient(C[{{k, l}, {i, j}}], 1.);
          constr2->SetCoefficient(q2[i][j], -1.);

          operations_research::MPConstraint* constr3 =
              solver->MakeRowConstraint(
                  -INF, 1 - q1[k][l],
                  "c_opt_" + std::to_string(k) + "_" + std::to_string(l) + "_" +
                      std::to_string(i) + "_" + std::to_string(j));

          constr3->SetCoefficient(C[{{k, l}, {i, j}}], -1.);
          constr3->SetCoefficient(q2[i][j], 1.);

          offset += q1[k][l];
          obj->SetOffset(obj->offset() + q1[k][l]);  // mb бесполезно
          obj->SetCoefficient(q2[i][j], 1.);
          obj->SetCoefficient(C[{{k, l}, {i, j}}], -2.);
        }
      }
      for (size_t k = j + 1; k < m; ++k) {
        operations_research::MPConstraint* constr4 = solver->MakeRowConstraint(
            0., 1.,
            "c4_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                std::to_string(k));

        constr4->SetCoefficient(q2[i][j], 1.);
        constr4->SetCoefficient(q2[j][k], 1.);
        constr4->SetCoefficient(q2[i][k], -1.);
      }
    }
  }

  obj->SetMinimization();

  const operations_research::MPSolver::ResultStatus result_status =
      solver->Solve();


  std::vector<uint32_t> ans(m), range(m);
  for (size_t i = 0; i < m; ++i) {
    range[i] = i;
    for (size_t j = i + 1; j < m; ++j) {
      if (i == j) continue;
      ans[j] += q2[i][j]->solution_value();
      ans[i] += (uint32_t)q2[i][j]->solution_value() ^ 1;
    }
  }

  for (size_t i = 0; i < m; ++i) {
    range[ans[i]] = i;
  }

  return range;
}

std::vector<uint32_t> MILP2(const size_t n, const size_t m,
                            const std::vector<std::vector<uint32_t> >& NB) {
  std::unique_ptr<operations_research::MPSolver> solver(
      operations_research::MPSolver::CreateSolver("GLOP"));

  solver->Clear();

  const double INF = solver->infinity();

  std::vector<std::vector<bool> > q1(n, std::vector<bool>(n));

  std::vector<std::vector<operations_research::MPVariable*> > q2(
      m, std::vector<operations_research::MPVariable*>(m));

  std::map<
      std::pair<std::pair<uint32_t, uint32_t>, std::pair<uint32_t, uint32_t> >,
      operations_research::MPVariable*>
      C;

  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      q1[i][j] = 1;
    }
  }


  for (size_t i = 0; i < m - 1; ++i) {
    for (size_t j = i + 1; j < m; ++j) {
      // q2
      q2[i][j] = solver->MakeBoolVar("q_" + std::to_string(i) + "_" +
                                     std::to_string(j));

      // C
      for (size_t k = 0; k < n; ++k) {
        for (size_t l = 0; l < n; ++l) {


          if (k == l) continue;

          C[{{k, l}, {i, j}}] = solver->MakeBoolVar(
              "C_" + std::to_string(k) + "_" + std::to_string(l) + "_" +
              std::to_string(i) + "_" + std::to_string(j));
        }
      }
    }
  }

  operations_research::MPObjective* obj = solver->MutableObjective();
  uint32_t offset = 0;
  for (size_t i = 0; i < m - 1; ++i) {
    for (size_t j = i + 1; j < m; ++j) {
      for (auto& k : NB[i]) {
        for (auto& l : NB[j]) {
          if (k == l) continue;

          operations_research::MPConstraint* constr1 =
              solver->MakeRowConstraint(
                  -INF, q1[k][l],
                  "c1_" + std::to_string(k) + "_" + std::to_string(l) + "_" +
                      std::to_string(i) + "_" + std::to_string(j));

          constr1->SetCoefficient(C[{{k, l}, {i, j}}], 1.);

          // mb не в том форе
          operations_research::MPConstraint* constr2 =
              solver->MakeRowConstraint(
                  -INF, 0.,
                  "c2_" + std::to_string(k) + "_" + std::to_string(l) + "_" +
                      std::to_string(i) + "_" + std::to_string(j));

          constr2->SetCoefficient(C[{{k, l}, {i, j}}], 1.);
          constr2->SetCoefficient(q2[i][j], -1.);

          operations_research::MPConstraint* constr3 =
              solver->MakeRowConstraint(
                  -INF, 1 - q1[k][l],
                  "c_opt_" + std::to_string(k) + "_" + std::to_string(l) + "_" +
                      std::to_string(i) + "_" + std::to_string(j));

          constr3->SetCoefficient(C[{{k, l}, {i, j}}], -1.);
          constr3->SetCoefficient(q2[i][j], 1.);

          offset += q1[k][l];
          obj->SetOffset(obj->offset() + q1[k][l]);  // mb бесполезно
          obj->SetCoefficient(q2[i][j], 1.);
          obj->SetCoefficient(C[{{k, l}, {i, j}}], -1.);
        }
      }
      for (size_t k = j + 1; k < m; ++k) {
        operations_research::MPConstraint* constr4 = solver->MakeRowConstraint(
            0., 1.,
            "c4_" + std::to_string(i) + "_" + std::to_string(j) + "_" +
                std::to_string(k));

        constr4->SetCoefficient(q2[i][j], 1.);
        constr4->SetCoefficient(q2[j][k], 1.);
        constr4->SetCoefficient(q2[i][k], -1.);
      }
    }
  }

  obj->SetMinimization();

  const operations_research::MPSolver::ResultStatus result_status =
      solver->Solve();

  std::vector<uint32_t> ans(m), range(m);
  for (size_t i = 0; i < m; ++i) {
    range[i] = i;
    for (size_t j = i + 1; j < m; ++j) {
      if (i == j) continue;
      ans[j] += q2[i][j]->solution_value();
      ans[i] += (uint32_t)q2[i][j]->solution_value() ^ 1;
    }
  }
  uint32_t cross = 0;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      for (auto& k : NB[range[i]]) {
        for (auto& l : NB[range[j]]) {
          if (k < l && i > j) ++cross;
          if (k > l && i < j) ++cross;
        }
      }
    }
  }
  cross /= 2;
  for (size_t i = 0; i < m; ++i) {
    range[ans[i]] = i;
  }

  return range;
}

int main() {
  std::string b;

  size_t n, m, nE;

  std::cin >> b >> b;
  std::cin >> n >> m >> nE;

  std::vector<std::vector<uint32_t> > NB(m);

  for (size_t i = 0; i < nE; ++i) {
    uint32_t a, b;
    std::cin >> a >> b;
    --a, --b;
    if (a > b) std::swap(a, b);
    b -= n;
    NB[b].push_back(a);
  }

  const std::vector<uint32_t> ans1 = MILP1(n, m, NB);
  const std::vector<uint32_t> ans2 = MILP2(n, m, NB);

  uint32_t cross1 = 0, cross2 = 0;

  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      for (auto& k : NB[ans1[i]]) {
        for (auto& l : NB[ans1[j]]) {
          if (k < l && i > j) ++cross1;
          if (k > l && i < j) ++cross1;
        }
      }

      for (auto& k : NB[ans2[i]]) {
        for (auto& l : NB[ans2[j]]) {
          if (k < l && i > j) ++cross2;
          if (k > l && i < j) ++cross2;
        }
      }
    }
  }

  const std::vector<uint32_t>* ans = (cross1 > cross2) ? &ans2 : &ans1;
  for (size_t i = 0; i < m; ++i) {
    std::cout << (*ans)[i] + n + 1 << std::endl;
  }
  return 0;
}
