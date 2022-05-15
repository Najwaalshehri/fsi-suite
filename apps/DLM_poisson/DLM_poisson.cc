/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */


// @sect3{Include files}

// The first few files have already been covered in previous examples and will
// thus not be further commented on.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/discrete_time.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parsed_function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <fstream>
#include <deal.II/fe/fe_q.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/meshworker/copy_data.h>
#include <deal.II/meshworker/scratch_data.h>

// #include "parsed_tools/components.h"
// #include "lac.h"
// #include "lac_initializer.h"
#include <deal.II/non_matching/coupling.h>

using namespace dealii;
// using ParsedTools::Components::join;

template <int dim>
// class Step6 : public ParameterAcceptor
class Step6 
{
public:
      /**
       * Constructor. Initialize all parameters, including the base class, and
       * make sure the class is ready to run.
       */
  Step6();

      /**
       * Destroy the Poisson object
       */
  // virtual ~Step6() = default;
  void run();
      /**
       * Build a custom error estimator1.
       */
  virtual void
  estimator1();
  virtual void
  estimator2();

  //   /**
//      * Default CopyData object, used in the WorkStream class.
//      */
  using CopyData = MeshWorker::CopyData<1, 1, 1>;

//     /**
//      * Default ScratchData object, used in the workstream class.
//      */
  using ScratchData = MeshWorker::ScratchData<dim>;
private:
      /**
       * Explicitly assemble the Poisson problem on a single cell.
       */
  void
  assemble_system_one_cell_omega(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData & scratch,
    CopyData &copy);
  void
  assemble_system_one_cell_omega2(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData & scratch,
    CopyData &copy);

  void
  copy_one_cell_omega(const CopyData &copy);
  void
  copy_one_cell_omega2(const CopyData &copy);

  void make_grid_omega();
  void make_grid_omega2();
  void setup_system_omega();
  void setup_system_omega2();
  void assemble_system_omega();
  void assemble_system_omega2();   

  void solve_u1();
  void solve_u2();
  // void mark(const Vector<float> &error_per_cell_omega);
  void refine_omega();
  void refine_omega2();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim> triangulation_omega;
  Triangulation<dim> triangulation_omega2;
  FE_Q<dim>       fe;
  FE_Q<dim>       fe_iv1;
  FE_Q<dim>       fe2;
  FE_Q<dim>       fe_iv2;
  DoFHandler<dim> omega_dh;
  DoFHandler<dim> omega2_dh;
  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints2;

  SparseMatrix<double> A_omega;
  SparsityPattern      sparsity_pattern_omega;
  SparseMatrix<double> A_omega2;
  SparsityPattern      sparsity_pattern_omega2;
  

  Vector<double> u_omega;
  Vector<double> u_omega2;
  Vector<double> rhs_omega;
  Vector<double> rhs_omega2;
  Vector<double> error_per_cell_omega;
  Vector<double> error_per_cell_omega2;
  
  
  double coefficient_omega =100.0;
  // const unsigned int degree_omega;
  double coefficient_omega2 =1.0;
  // const unsigned int degree_omega2;

  
  //omega2:
  //       /**
  //      * Make sure we initialize the right type of linear solver.
  //      */

  // // ParsedTools::Function<spacedim> coefficient_omega;


  
};


// template <int dim>
// double coefficient_omega(const Point<dim> &p)
// // double coefficient_omega()
// {
//   if (p.square() < 0.5 * 0.5)
//     return 1;
//   else
//     return 1;
// }

template <int dim>
Step6<dim>::Step6()
  // : degree_omega(degree_omega)
  // , degree_omega2(degree_omega2)
  : fe(1)
  , fe_iv1(1)
  , fe2(1)
  , fe_iv2(1)
  , omega_dh(triangulation_omega)
  , omega2_dh(triangulation_omega2)
{
//   this->add_parameter("Number of cycles", n_cycles);
}


template <int dim>
void Step6<dim>::make_grid_omega()
{
  GridGenerator::hyper_cube(triangulation_omega, 0, 6);
  std::cout << "Number of active cells omega: " << triangulation_omega.n_active_cells()
            << std::endl;
}

template <int dim>
void Step6<dim>::make_grid_omega2()
{
  GridGenerator::hyper_cube(triangulation_omega2, exp(1), 1+M_PI);
  std::cout << "Number of active cells omega2: " << triangulation_omega2.n_active_cells()
            << std::endl;
}

template <int dim>
void Step6<dim>::setup_system_omega()
{
  // const FEValuesExtractors::Scalar fe(0);
  // const FEValuesExtractors::Scalar fe2(1);
  omega_dh.distribute_dofs(fe);

  constraints.clear();
  DoFTools::make_hanging_node_constraints(omega_dh, constraints);

  VectorTools::interpolate_boundary_values(omega_dh,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();
  DynamicSparsityPattern dsp(omega_dh.n_dofs(),omega_dh.n_dofs());
  DoFTools::make_sparsity_pattern(omega_dh,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern_omega.copy_from(dsp);
  A_omega.reinit(sparsity_pattern_omega);
  u_omega.reinit(omega_dh.n_dofs());
  rhs_omega.reinit(omega_dh.n_dofs());
  error_per_cell_omega.reinit(triangulation_omega.n_active_cells());
  
  deallog << "Omega dofs: " << omega_dh.n_dofs() << std::endl;
}

template <int dim>
void Step6<dim>::setup_system_omega2()
{
  omega2_dh.distribute_dofs(fe2);

  constraints2.clear();
  DoFTools::make_hanging_node_constraints(omega2_dh, constraints2);

  VectorTools::interpolate_boundary_values(omega2_dh,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints2);
  constraints2.close();
  DynamicSparsityPattern dsp(omega2_dh.n_dofs(),omega2_dh.n_dofs());
  DoFTools::make_sparsity_pattern(omega2_dh,
                                  dsp,
                                  constraints2,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern_omega2.copy_from(dsp);
  A_omega2.reinit(sparsity_pattern_omega2);
  u_omega2.reinit(omega2_dh.n_dofs());
  rhs_omega2.reinit(omega2_dh.n_dofs());
  error_per_cell_omega2.reinit(triangulation_omega2.n_active_cells());

  deallog << "Omega2 dofs: " << omega2_dh.n_dofs() << std::endl;
}

template <int dim>
void Step6<dim>::assemble_system_one_cell_omega(
  const typename DoFHandler<dim>::active_cell_iterator &  cell,
  ScratchData &                                           scratch,
  CopyData &                                              copy)
{
  auto &cell_matrix = copy.matrices[0];
  auto &cell_rhs    = copy.vectors[0];

  cell->get_dof_indices(copy.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  cell_matrix           = 0;
  cell_rhs              = 0;

  for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
      // const double current_coefficient_omega =
      //   coefficient_omega(fe_values.quadrature_point(q_index));
        // coefficient_omega(fe_values.quadrature_point(q_index));
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
            (coefficient_omega *                            // a(x_q)
              fe_values.shape_grad(i, q_index) *       // grad phi_i(x_q)
              fe_values.shape_grad(j, q_index) *       // grad phi_j(x_q)
              fe_values.JxW(q_index));                 // dx
      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += (fe_values.shape_value(i, q_index) *   // phi_i(x_q)
                        1.0 *                                 // f(x)
                        fe_values.JxW(q_index));              // dx
    }
}

template <int dim>
void Step6<dim>::assemble_system_one_cell_omega2(
  const typename DoFHandler<dim>::active_cell_iterator &  cell,
  ScratchData &                                           scratch,
  CopyData &                                              copy)
{
  auto &cell_matrix = copy.matrices[0];
  auto &cell_rhs    = copy.vectors[0];

  cell->get_dof_indices(copy.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  cell_matrix          = 0;
  cell_rhs             = 0;

  for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
      // const double current_coefficient_omega =
      //   coefficient_omega(fe_values.quadrature_point(q_index));
        // coefficient_omega(fe_values.quadrature_point(q_index));
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          cell_matrix(i, j) +=
            (coefficient_omega2 *                            // a(x_q)
              fe_values.shape_grad(i, q_index) *       // grad phi_i(x_q)
              fe_values.shape_grad(j, q_index) *       // grad phi_j(x_q)
              fe_values.JxW(q_index));                 // dx
      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += (fe_values.shape_value(i, q_index) *   // phi_i(x_q)
                        1.0 *                                 // f(x)
                        fe_values.JxW(q_index));              // dx
    }
}

template <int dim>
void Step6<dim>::estimator1()
  {
      //TimerOutput::Scope timer_section(this->timer, "estimator1");
      error_per_cell_omega = 0;
      const QGauss<dim> quadrature_formula(fe.degree + 1);
      const QGauss<dim-1> face_quadrature_formula(fe.degree + 1);
      // Quadrature<dim> quadrature_formula =
      //   ParsedTools::Components::get_cell_quadrature(
      //     this->triangulation_omega, this->fe().tensor_degree() + 1);

      // Quadrature<dim - 1> face_quadrature_formula =
      //   ParsedTools::Components::get_face_quadrature(
      //     this->triangulation_omega, this->fe_iv().tensor_degree() + 1);

  
      ScratchData scratch(this->fe,
                          quadrature_formula,
                          update_quadrature_points | update_hessians |
                            update_JxW_values,
                          face_quadrature_formula,
                          update_normal_vectors | update_gradients |
                            update_quadrature_points | update_JxW_values);

      // A copy data for error estimator1s for each cell. We store the indices of
      // the cells, and the values of the error estimator1 to be added to the
      // cell indicators.
      struct MyCopyData
      {
        std::vector<unsigned int> cell_indices;
        std::vector<float>        indicators;
      };

      MyCopyData copy;

      // I will use this FEValuesExtractor to leverage the capabilities of the
      // ScratchData
      FEValuesExtractors::Scalar scalar(0);

      // This is called in each cell
      auto cell_worker = [&](const auto &cell, auto &scratch, auto &copy) {
        const FEValues<dim> &fe_value = scratch.reinit(cell);
        const auto  H    = cell->diameter();

        // Reset the copy data
        copy.cell_indices.resize(0);
        copy.indicators.resize(0);

        // Save the index of this cell
        copy.cell_indices.emplace_back(cell->active_cell_index());

        // At every call of this function, a new vector of dof values is
        // generated and stored internally, so that you can later call
        // scratch.get_values(...)
        scratch.extract_local_dof_values("u_omega",
                                         u_omega);

        // Get the values of the u_omega at the quadrature points
        const auto &lap_u = scratch.get_laplacians("u_omega", scalar);

        // Points and weights of the quadrature formula
        //const auto &q_points = scratch.get_quadrature_points();
        const auto &JxW      = scratch.get_JxW_values();

        // Reset vectors
        float cell_indicator = 0;

        // Now store the values of the residual square in the copy data
        for (const auto q_index : fe_value.quadrature_point_indices())
          {
            // const double current_coefficient_omega =
            //    coefficient_omega(fe_value.quadrature_point(q_index));
            // double coeff = coefficient_omega.value(
            //        fe_value.quadrature_point(q_index);
            const auto res =
              coefficient_omega * lap_u[q_index] + 1;
              //this->forcing_term.value(q_points[q_index]);

            cell_indicator += (H * H * res * res * JxW[q_index]); // dx
          }
        copy.indicators.emplace_back(cell_indicator);
      };

      // This is called in each face, refined or not.
      auto face_worker = [&](const auto &cell,
                             const auto &f,
                             const auto &sf,
                             const auto &ncell,
                             const auto &nf,
                             const auto &nsf,
                             auto &      scratch,
                             auto &      copy) {
        // Here we intialize the inteface values
        //const auto &fe_ivalue = scratch.reinit(cell, f, sf, ncell, nf, nsf);
        const FEInterfaceValues<dim> &fe_ivalue = scratch.reinit(cell, f, sf, ncell, nf, nsf);

        const auto h = cell->face(f)->diameter();

        // Add this cell to the copy data
        copy.cell_indices.emplace_back(cell->active_cell_index());

        // Same as before. Extract local dof values of the u_omega
        scratch.extract_local_dof_values("u_omega",
                                         u_omega);

        // ...so that we can call scratch.get_(...)
        const auto jump_grad =
          scratch.get_jumps_in_gradients("u_omega", scalar);

        const auto &JxW     = scratch.get_JxW_values();
        const auto &normals = scratch.get_normal_vectors();
        // const auto &q_points2 = scratch.get_quadrature_points();

        // Now store the values of the gradient jump in the copy data
        float face_indicator = 0;
        for (const auto q_index : fe_ivalue.quadrature_point_indices())
          {
            // const double current_coefficient_omega =
            //    coefficient_omega(q_points2[q_index]);
            const auto J = coefficient_omega * jump_grad[q_index] * normals[q_index];

            face_indicator += (h * J * J * JxW[q_index]); // dx
          }
        copy.indicators.emplace_back(face_indicator);
      };


      auto copier = [&](const auto &copy) {
        AssertDimension(copy.cell_indices.size(), copy.indicators.size());
        for (unsigned int i = 0; i < copy.cell_indices.size(); ++i)
          {
            error_per_cell_omega[copy.cell_indices[i]] += copy.indicators[i];
          }
      };

      using CellFilter = FilteredIterator<
        typename DoFHandler<dim>::active_cell_iterator>;

      MeshWorker::mesh_loop(this->omega_dh.begin_active(),
                            this->omega_dh.end(),
                            cell_worker,
                            copier,
                            scratch,
                            copy,
                            MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_own_interior_faces_both,
                            {},
                            face_worker);

      deallog << "L2 norm of indicator_omega: " << std::sqrt(error_per_cell_omega.l1_norm()) << 
      ", n_dofs_omega: " << omega_dh.n_dofs() << std::endl;
  }

template <int dim>
void Step6<dim>::estimator2()
  {
      //TimerOutput::Scope timer_section(this->timer, "estimator1");
      error_per_cell_omega2 = 0;
      const QGauss<dim> quadrature_formula(fe2.degree + 1);
      const QGauss<dim-1> face_quadrature_formula(fe2.degree + 1);
      // Quadrature<dim> quadrature_formula =
      //   ParsedTools::Components::get_cell_quadrature(
      //     this->triangulation_omega, this->fe().tensor_degree() + 1);

      // Quadrature<dim - 1> face_quadrature_formula =
      //   ParsedTools::Components::get_face_quadrature(
      //     this->triangulation_omega, this->fe_iv().tensor_degree() + 1);

  
      ScratchData scratch(this->fe2,
                          quadrature_formula,
                          update_quadrature_points | update_hessians |
                            update_JxW_values,
                          face_quadrature_formula,
                          update_normal_vectors | update_gradients |
                            update_quadrature_points | update_JxW_values);

      // A copy data for error estimator1s for each cell. We store the indices of
      // the cells, and the values of the error estimator1 to be added to the
      // cell indicators.
      struct MyCopyData
      {
        std::vector<unsigned int> cell_indices;
        std::vector<float>        indicators;
      };

      MyCopyData copy;

      // I will use this FEValuesExtractor to leverage the capabilities of the
      // ScratchData
      FEValuesExtractors::Scalar scalar2(0);

      // This is called in each cell
      auto cell_worker = [&](const auto &cell, auto &scratch, auto &copy) {
        const FEValues<dim> &fe_value = scratch.reinit(cell);
        const auto  H    = cell->diameter();

        // Reset the copy data
        copy.cell_indices.resize(0);
        copy.indicators.resize(0);

        // Save the index of this cell
        copy.cell_indices.emplace_back(cell->active_cell_index());

        // At every call of this function, a new vector of dof values is
        // generated and stored internally, so that you can later call
        // scratch.get_values(...)
        scratch.extract_local_dof_values("u_omega2",
                                         u_omega2);

        // Get the values of the u_omega at the quadrature points
        const auto &lap_u = scratch.get_laplacians("u_omega2", scalar2);

        // Points and weights of the quadrature formula
        //const auto &q_points = scratch.get_quadrature_points();
        const auto &JxW      = scratch.get_JxW_values();

        // Reset vectors
        float cell_indicator = 0;

        // Now store the values of the residual square in the copy data
        for (const auto q_index : fe_value.quadrature_point_indices())
          {
            // const double current_coefficient_omega =
            //    coefficient_omega(fe_value.quadrature_point(q_index));
            // double coeff = coefficient_omega.value(
            //        fe_value.quadrature_point(q_index);
            const auto res =
              coefficient_omega2 * lap_u[q_index] + 1;
              //this->forcing_term.value(q_points[q_index]);

            cell_indicator += (H * H * res * res * JxW[q_index]); // dx
          }
        copy.indicators.emplace_back(cell_indicator);
      };

      // This is called in each face, refined or not.
      auto face_worker = [&](const auto &cell,
                             const auto &f,
                             const auto &sf,
                             const auto &ncell,
                             const auto &nf,
                             const auto &nsf,
                             auto &      scratch,
                             auto &      copy) {
        // Here we intialize the inteface values
        //const auto &fe_ivalue = scratch.reinit(cell, f, sf, ncell, nf, nsf);
        const FEInterfaceValues<dim> &fe_ivalue = scratch.reinit(cell, f, sf, ncell, nf, nsf);

        const auto h = cell->face(f)->diameter();

        // Add this cell to the copy data
        copy.cell_indices.emplace_back(cell->active_cell_index());

        // Same as before. Extract local dof values of the u_omega
        scratch.extract_local_dof_values("u_omega2",
                                         u_omega2);

        // ...so that we can call scratch.get_(...)
        const auto jump_grad =
          scratch.get_jumps_in_gradients("u_omega2", scalar2);

        const auto &JxW     = scratch.get_JxW_values();
        const auto &normals = scratch.get_normal_vectors();
        // const auto &q_points2 = scratch.get_quadrature_points();

        // Now store the values of the gradient jump in the copy data
        float face_indicator = 0;
        for (const auto q_index : fe_ivalue.quadrature_point_indices())
          {
            // const double current_coefficient_omega =
            //    coefficient_omega(q_points2[q_index]);
            const auto J = coefficient_omega2 * jump_grad[q_index] * normals[q_index];

            face_indicator += (h * J * J * JxW[q_index]); // dx
          }
        copy.indicators.emplace_back(face_indicator);
      };


      auto copier = [&](const auto &copy) {
        AssertDimension(copy.cell_indices.size(), copy.indicators.size());
        for (unsigned int i = 0; i < copy.cell_indices.size(); ++i)
          {
            error_per_cell_omega2[copy.cell_indices[i]] += copy.indicators[i];
          }
      };

      using CellFilter = FilteredIterator<
        typename DoFHandler<dim>::active_cell_iterator>;

      MeshWorker::mesh_loop(this->omega2_dh.begin_active(),
                            this->omega2_dh.end(),
                            cell_worker,
                            copier,
                            scratch,
                            copy,
                            MeshWorker::assemble_own_cells |
                              MeshWorker::assemble_own_interior_faces_both,
                            {},
                            face_worker);

      deallog << "L2 norm of indicator_omega2: " << std::sqrt(error_per_cell_omega2.l1_norm()) << 
      ", n_dofs_omega2: " << omega2_dh.n_dofs() << std::endl;
  }

template <int dim>
void
Step6<dim>::copy_one_cell_omega(const CopyData &copy)
{
  constraints.distribute_local_to_global(copy.matrices[0],
                                          copy.vectors[0],
                                          copy.local_dof_indices[0],
                                          A_omega,
                                          rhs_omega);
}

template <int dim>
void
Step6<dim>::copy_one_cell_omega2(const CopyData &copy)
{
  constraints2.distribute_local_to_global(copy.matrices[0],
                                          copy.vectors[0],
                                          copy.local_dof_indices[0],
                                          A_omega2,
                                          rhs_omega2);
}

template <int dim>
void Step6<dim>::assemble_system_omega()
{
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  const QGauss<dim-1> face_quadrature_formula(fe.degree + 1);
  ScratchData scratch(this->fe,
                          quadrature_formula,
                          update_values | update_quadrature_points | update_hessians |
                            update_JxW_values,
                          face_quadrature_formula,
                          update_values| update_normal_vectors | update_gradients |
                            update_quadrature_points | update_JxW_values);

  CopyData copy(fe.n_dofs_per_cell());

  auto worker = [&](const auto &cell, auto &scratch, auto &copy) {
    assemble_system_one_cell_omega(cell, scratch, copy);
  };
  auto copier = [&](const auto &copy) { copy_one_cell_omega(copy); };
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  using CellFilter = FilteredIterator<
    typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run(this->omega_dh.begin_active(),
                  this->omega_dh.end(),
                  worker,
                  copier,
                  scratch,
                  copy);

  A_omega.compress(VectorOperation::add);
  rhs_omega.compress(VectorOperation::add);
}

template <int dim>
void Step6<dim>::assemble_system_omega2()
{
  const QGauss<dim> quadrature_formula(fe2.degree + 1);
  const QGauss<dim-1> face_quadrature_formula(fe2.degree + 1);
  ScratchData scratch(this->fe2,
                          quadrature_formula,
                          update_values | update_quadrature_points | update_hessians |
                            update_JxW_values,
                          face_quadrature_formula,
                          update_values| update_normal_vectors | update_gradients |
                            update_quadrature_points | update_JxW_values);

  CopyData copy(fe2.n_dofs_per_cell());

  auto worker = [&](const auto &cell, auto &scratch, auto &copy) {
    assemble_system_one_cell_omega2(cell, scratch, copy);
  };
  auto copier = [&](const auto &copy) { copy_one_cell_omega2(copy); };
  FEValues<dim> fe_values(fe2,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  using CellFilter = FilteredIterator<
    typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run(this->omega2_dh.begin_active(),
                  this->omega2_dh.end(),
                  worker,
                  copier,
                  scratch,
                  copy);

  A_omega2.compress(VectorOperation::add);
  rhs_omega2.compress(VectorOperation::add);
}


template <int dim>
void Step6<dim>::solve_u1()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(A_omega, 1.2);

  solver.solve(A_omega, u_omega, rhs_omega, preconditioner);

  constraints.distribute(u_omega);
}

template <int dim>
void Step6<dim>::solve_u2()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);

  PreconditionSSOR<SparseMatrix<double>> preconditioner2;
  preconditioner2.initialize(A_omega2, 1.2);

  solver.solve(A_omega2, u_omega2, rhs_omega2, preconditioner2);

  constraints2.distribute(u_omega2);
}

// template <int dim>
// void
// Step6<dim>::solve()
// {
//   //TimerOutput::Scope timer_section(this->timer, "solve");
//   const auto A = linear_operator<VectorType>(this->matrix.block(0, 0));
//   this->preconditioner.initialize(this->matrix.block(0, 0));
//   const auto Ainv         = this->inverse_operator(A, this->preconditioner);
//   this->u_omega.block(0) = Ainv * this->rhs.block(0);
//   this->constraints.distribute(this->u_omega);
//   this->locally_relevant_u_omega = this->u_omega;
// }

template <int dim>
void Step6<dim>::refine_omega()
{
  // estimate();

  
  GridRefinement::refine_and_coarsen_fixed_number(triangulation_omega,
                                                  error_per_cell_omega,
                                                  0.3,
                                                  0.03);

  
  triangulation_omega.execute_coarsening_and_refinement();
}

template <int dim>
void Step6<dim>::refine_omega2()
{
  // estimate();

  
  GridRefinement::refine_and_coarsen_fixed_number(triangulation_omega2,
                                                  error_per_cell_omega2,
                                                  0.3,
                                                  0.03);

  
  triangulation_omega.execute_coarsening_and_refinement();
}


template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{
  // {
  //   GridOut               grid_out;
  //   std::ofstream         output("grid-" + std::to_string(cycle) + ".gnuplot");
  //   GridOutFlags::Gnuplot gnuplot_flags(false, 5);
  //   grid_out.set_flags(gnuplot_flags);
  //   MappingQ<dim> mapping(3);
  //   grid_out.write_gnuplot(triangulation_omega, output, &mapping);
  // }

  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(omega_dh);
    data_out.add_data_vector(u_omega, "u_omega");
    data_out.add_data_vector(error_per_cell_omega, "indicator");
    data_out.build_patches();

    std::ofstream output1("u_omega-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output1);

    DataOut<dim> data2_out;
    data2_out.attach_dof_handler(omega2_dh);
    data2_out.add_data_vector(u_omega2, "u_omega2");
    data2_out.add_data_vector(error_per_cell_omega2, "indicator2");
    data2_out.build_patches();

    std::ofstream output2("u_omega2-" + std::to_string(cycle) + ".vtu");
    data_out.write_vtu(output2);

  }
}

template <int dim>
void Step6<dim>::run()
{
  deallog.depth_console(10);
  deallog.push("RUN");

  std::ofstream outfile("indicator.txt");
  std::ofstream outfile2("indicator2.txt");

  for (unsigned int cycle = 0; cycle < 6; ++cycle)
    {
      deallog << "Cycle " << cycle << std::endl;
      if (cycle == 0)
      {
        // GridGenerator::hyper_ball(triangulation_omega);
        make_grid_omega();
        triangulation_omega.refine_global(3);
        make_grid_omega2();
        triangulation_omega2.refine_global(4);
      }
      else
      {
        refine_omega();
        refine_omega2();
      }

      setup_system_omega();
      setup_system_omega2();

      assemble_system_omega();
      assemble_system_omega2();

      solve_u1();
      solve_u2();

      estimator1();
      estimator2();
      outfile << omega_dh.n_dofs() << " "
             << std::sqrt(error_per_cell_omega.l1_norm()) << std::endl;
      output_results(cycle);
      outfile2 << omega2_dh.n_dofs() << " "
             << std::sqrt(error_per_cell_omega2.l1_norm()) << std::endl;
      output_results(cycle);
      }
    deallog.pop();
    outfile.close();
    outfile2.close();
}

int main()
{
  try
    {
      // const unsigned int     degree_omega = 1;
      // const unsigned int     degree_omega2 = 2;
      Step6<2> laplace_problem_2d;
      //parameterAcceptor::initialize("step-6.prm", "used_step-6.prm");
      laplace_problem_2d.run();
    }

  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
