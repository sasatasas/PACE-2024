#include <cstdio>
#include <iostream>
#include <map>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "ortools/linear_solver/linear_solver.h"

#define _pair std::pair<uint32_t, uint32_t>

uint32_t getNumOfCrosses(const std::vector<std::vector<uint32_t> >& NB,
                         const std::vector<uint32_t>& ans, const uint32_t m) {
  uint32_t cross = 0;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < m; ++j) {
      for (auto& k : NB[ans[i]]) {
        for (auto& l : NB[ans[j]]) {
          if (k < l && i > j) ++cross;
        }
      }
    }
  }

  return cross;
}

std::vector<uint32_t> MILP(const size_t n, const size_t m,
                           const std::vector<std::vector<uint32_t> >& NB) {
  std::unique_ptr<operations_research::MPSolver> solver(
      operations_research::MPSolver::CreateSolver("SCIP"));

  solver->SetSolverSpecificParametersAsString("limits/maxsol=100"); // for multi solving 
  solver->SetSolverSpecificParametersAsString("constraints/countsols/collect=TRUE");
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
          // std::cerr << "C_" + std::to_string(k) + "_" + std::to_string(l) +
          //                  "_" + std::to_string(i) + "_" + std::to_string(j)
          //           << "\n";

          if (k == l) continue;

          C[{{k, l}, {i, j}}] = solver->MakeBoolVar(
              "C_" + std::to_string(k) + "_" + std::to_string(l) + "_" +
              std::to_string(i) + "_" + std::to_string(j));
        }
      }
    }
  }

  // mb q[k][l] просто заменить на проверку k < l ? 1 : 0
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
          obj->SetOffset(obj->offset() + q1[k][l]); 
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
      // std::cerr << q2[i][j]->solution_value() << " ";
    }
    // std::cerr << std::endl;
  }
  for (size_t i = 0; i < m; ++i) {
    range[ans[i]] = i;
  }



  uint32_t cross = getNumOfCrosses(NB, range, m);
  // std::cout << "num: " << 0 << ", crosses=" << cross << " " << obj->Value()  << std::endl;




  uint32_t count = 0.;
  while (solver->NextSolution()) {
    std::vector<uint32_t> ansRes(m), rangeRes(m);
    for (size_t i = 0; i < m; ++i) {
      rangeRes[i] = i;
      for (size_t j = i + 1; j < m; ++j) {
        if (i == j) continue;
        ansRes[j] += q2[i][j]->solution_value();
        ansRes[i] += (uint32_t)q2[i][j]->solution_value() ^ 1;
      }
    }
    for (size_t i = 0; i < m; ++i) {
      rangeRes[ansRes[i]] = i;
    }

    uint32_t crossRes = getNumOfCrosses(NB, rangeRes, m);

    // std::cout << "Solution " << count << ", crosses=" << crossRes << " " << obj->Value()  << std::endl;
    ++count;
    if (crossRes > cross) {
      continue;
    }
    ans = ansRes, range = rangeRes, cross = crossRes;
  }




  uint32_t addConsrt = 1;
  while (true) {
    int32_t objRes = obj->Value();

    operations_research::MPConstraint* constr = solver->MakeRowConstraint(
        -INF, objRes-1-offset,"optional_" + std::to_string(addConsrt));

    for (size_t i = 0; i < m - 1; ++i) {
      for (size_t j = i + 1; j < m; ++j) {
        for (auto& k : NB[i]) {
          for (auto& l : NB[j]) {
            if (k == l) continue;
            constr->SetCoefficient(q2[i][j], 1.);
            constr->SetCoefficient(C[{{k, l}, {i, j}}], -2.);
          }
        }
      }
    }

    const operations_research::MPSolver::ResultStatus result_status =
        solver->Solve();

    if (result_status != operations_research::MPSolver::OPTIMAL) {
      break;
    }

    std::vector<uint32_t> ansRes(m), rangeRes(m);
    for (size_t i = 0; i < m; ++i) {
      rangeRes[i] = i;
      for (size_t j = i + 1; j < m; ++j) {
        if (i == j) continue;
        ansRes[j] += q2[i][j]->solution_value();
        ansRes[i] += (uint32_t)q2[i][j]->solution_value() ^ 1;
      }
    }
    for (size_t i = 0; i < m; ++i) {
      rangeRes[ansRes[i]] = i;
    }

    uint32_t crossRes = getNumOfCrosses(NB, rangeRes, m);

    // std::cout << "num: " << addConsrt << ", crosses=" << crossRes << " " << objRes << std::endl;

    uint32_t count = 0.;
    while (solver->NextSolution()) {
      std::vector<uint32_t> ansRes2(m), rangeRes2(m);
      for (size_t i = 0; i < m; ++i) {
        rangeRes2[i] = i;
        for (size_t j = i + 1; j < m; ++j) {
          if (i == j) continue;
          ansRes2[j] += q2[i][j]->solution_value();
          ansRes2[i] += (uint32_t)q2[i][j]->solution_value() ^ 1;
        }
      }
      for (size_t i = 0; i < m; ++i) {
        rangeRes2[ansRes2[i]] = i;
      }

      uint32_t crossRes2 = getNumOfCrosses(NB, rangeRes2, m);

      // std::cout << "Solution " << count << ", crosses=" << crossRes << " " << objRes  << std::endl;
      ++count;
      if (crossRes2 > crossRes) {
        continue;
      }
      ansRes = ansRes2, rangeRes = rangeRes2, crossRes = crossRes2;
    }

    if (crossRes >= cross) {
      break;
    }

    ans = ansRes, range = rangeRes, cross = crossRes;
    ++addConsrt;
  }

  return range;
}

int main() {
  std::string b;
  // std::cin >> b;
  // // b = "website_20.gr";
  // freopen(("tiny_set/" + b).c_str(), "r", stdin);

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

  const std::vector<uint32_t> ans = MILP(n, m, NB);

  uint32_t cross1 = getNumOfCrosses(NB, ans, m);

  // std::cout << cross1 << std::endl;
  for (size_t i = 0; i < m; ++i) {
    std::cout << ans[i] + n + 1 << std::endl;
  }

  return 0;
}
