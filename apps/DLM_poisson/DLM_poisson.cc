/* ---------------------------------------------------------------------
 *add:
 1- parameters file
 2- sure preco
 3- save results and plots with different names
 4- inf-sup-test
 5- is the problem in estimator 2 related to order of things?
 
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
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q_bubbles.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>

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
  using CopyData = MeshWorker::CopyData<2, 1, 1>;

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
  void setup_coupling();
  void assemble_system_omega();
  void assemble_system_omega2();  
  void assemble_coupling_system(); 

  void solve_u1();
  void solve_u2();
  void solve();
  // void mark(const Vector<float> &error_per_cell_omega);
  void refine_omega();
  void refine_omega2();
  void output_results(const unsigned int cycle) const;

  Triangulation<dim>         triangulation_omega;
  GridTools::Cache<dim, dim> space_grid_tools_cache;
  Triangulation<dim>         triangulation_omega2;

  FE_Q<dim>           fe;
  FESystem<dim>       fe2;
  DoFHandler<dim>     omega_dh;
  DoFHandler<dim>     omega2_dh;

  AffineConstraints<double> constraints;
  AffineConstraints<double> constraints2;

  SparseMatrix<double> A_omega;
  SparseMatrix<double> M_omega;
  SparsityPattern      sparsity_pattern_omega;
  SparseMatrix<double> B_omega2;
  SparseMatrix<double> M_omega2;
  SparsityPattern      sparsity_pattern_omega2;
  // BlockSparseMatrix<double> B_omega2;
  // BlockSparsityPattern      sparsity_pattern_omega2;
  SparsityPattern      coupling_sparsity;
  SparseMatrix<double> coupling_matrix;





  Vector<double> u_omega;
  Vector<double> u_omega2;
  Vector<double> u_omega_prime;
  Vector<double> u_omega2_prime;
  // Vector<double> solution_prime;
  // BlockVector<double> u_omega2;
  Vector<double> rhs_omega;
  Vector<double> rhs_omega2;
  // BlockVector<double> rhs_omega2;
  Vector<double> error_per_cell_omega;
  Vector<double> error_per_cell_omega2;
  
  
  double coefficient_omega = 1.0;
  double rhs1              = 1.0;
  // const unsigned int degree_omega;
  double coefficient_omega2 = 10.0;
  double rhs2               = 1.0;
  // const unsigned int degree_omega2;
  const FEValuesExtractors::Scalar primal;
  const FEValuesExtractors::Scalar multiplier; 
  // Vector<double> lambda_prime;
  // Vector<double> u_prime;
  // Vector<double> lambda;
  // Vector<double> u_2;
};


// template <int dim>
// double rhs_1(const Point<dim> &p)

// {
//   return Functions::CosineFunction<dim>;
// }

template <int dim>
Step6<dim>::Step6()
  // : degree_omega(degree_omega)
  // , degree_omega2(degree_omega2)
  : space_grid_tools_cache(triangulation_omega)
  , fe(1)
  // , fe2(FE_Q<dim>(2), 1, FE_DGQ<dim>(0), 1)
  , fe2(FE_Q_Bubbles<dim>(1), 1, FE_DGQ<dim>(0), 1)
  // , fe2(FE_Q<dim>(1), 1, FE_Q<dim>(1), 1)
  , omega_dh(triangulation_omega)
  , omega2_dh(triangulation_omega2)
  , primal(0)
  , multiplier(1)
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
  M_omega.reinit(sparsity_pattern_omega);
  u_omega.reinit(omega_dh.n_dofs());
  rhs_omega.reinit(omega_dh.n_dofs());
  error_per_cell_omega.reinit(triangulation_omega.n_active_cells());
  
  deallog << "Omega dofs: " << omega_dh.n_dofs() << std::endl;
}

template <int dim>
void Step6<dim>::setup_system_omega2()
{
  omega2_dh.distribute_dofs(fe2);
  DoFRenumbering::component_wise(omega2_dh);

  // const std::vector<types::global_dof_index> dofs_per_component =
  //     DoFTools::count_dofs_per_fe_component(omega2_dh);
  // const unsigned int n_u2         = dofs_per_component[0],
  //                    n_lambda     = dofs_per_component[1];
  
  // std::cout << "Number of active cells: " << triangulation_omega2.n_active_cells()
  //             << std::endl
  //             << "Total number of cells: " << triangulation_omega2.n_cells()
  //             << std::endl
  //             << "Number of degrees of freedom: " << omega2_dh.n_dofs()
  //             << " (" << n_u2  << '+' << n_lambda << ')' << std::endl;

  constraints2.clear();
  DoFTools::make_hanging_node_constraints(omega2_dh, constraints2);

  // VectorTools::interpolate_boundary_values(omega2_dh,
  //                                          0,
  //                                          Functions::ZeroFunction<dim>(2),
  //                                          constraints2);
  constraints2.close();
  DynamicSparsityPattern dsp(omega2_dh.n_dofs(),omega2_dh.n_dofs());
  DoFTools::make_sparsity_pattern(omega2_dh,
                                  dsp,
                                  constraints2,
                                  /*keep_constrained_dofs = */ false);
  sparsity_pattern_omega2.copy_from(dsp);
  B_omega2.reinit(sparsity_pattern_omega2);
  M_omega2.reinit(sparsity_pattern_omega2);
  u_omega2.reinit(omega2_dh.n_dofs());
  rhs_omega2.reinit(omega2_dh.n_dofs());
  error_per_cell_omega2.reinit(triangulation_omega2.n_active_cells());


  deallog << "Omega2 dofs: " << omega2_dh.n_dofs() << std::endl;
}

// template <int dim>
// void Step6<dim>::setup_system_omega2()
// {
//   omega2_dh.distribute_dofs(fe2);
//   DoFRenumbering::component_wise(omega2_dh);
//   const std::vector<types::global_dof_index> dofs_per_component =
//       DoFTools::count_dofs_per_fe_component(omega2_dh);
//   const unsigned int n_u2         = dofs_per_component[0],
//                      n_lambda     = dofs_per_component[1];

//   constraints2.clear();
//   DoFTools::make_hanging_node_constraints(omega2_dh, constraints2);
//   constraints2.close();

//   const std::vector<types::global_dof_index> block_sizes = {n_u2 , n_lambda };
//   BlockDynamicSparsityPattern                dsp(block_sizes, block_sizes);
//   DoFTools::make_sparsity_pattern(omega2_dh,
//                                   dsp,
//                                   constraints2,
//                                  /*keep_constrained_dofs = */ false);

//   // BlockDynamicSparsityPattern dsp(2, 2);
//   //    dsp.block(0, 0).reinit(n_u2 , n_u2);
//   //    dsp.block(1, 0).reinit(n_lambda, n_u2);
//   //    dsp.block(0, 1).reinit(n_u2, n_lambda);
//   //    dsp.block(1, 1).reinit(n_lambda, n_lambda);
//   //    dsp.collect_sizes();
//   DoFTools::make_sparsity_pattern(omega2_dh, dsp);
 
//   sparsity_pattern_omega2.copy_from(dsp);
//   B_omega2.reinit(sparsity_pattern_omega2);
 
//   u_omega2.reinit(2);
//   u_omega2.block(0).reinit(n_u2);
//   u_omega2.block(1).reinit(n_lambda);
//   u_omega2.collect_sizes();
 
//   rhs_omega2.reinit(2);
//   rhs_omega2.block(0).reinit(n_u2);
//   rhs_omega2.block(1).reinit(n_lambda);
//   rhs_omega2.collect_sizes();

//   error_per_cell_omega2.reinit(triangulation_omega2.n_active_cells());


//   deallog << "Omega2 dofs: " << omega2_dh.n_dofs() << std::endl;
// }

template <int dim>
  void Step6<dim>::setup_coupling()
  {
    // TimerOutput::Scope timer_section(monitor, "Setup coupling");

    QGauss<dim> quad(3);

    DynamicSparsityPattern dsp(omega_dh.n_dofs(), omega2_dh.n_dofs());

    NonMatching::create_coupling_sparsity_pattern(space_grid_tools_cache,
                                                  omega_dh,
                                                  omega2_dh,
                                                  quad,
                                                  dsp,
                                                  constraints,
                                                  ComponentMask(),    // for coupling  u_omega
                                                  ComponentMask(std::vector<bool>{false,true}), //with lambda
                                                  StaticMappingQ1<dim>::mapping,
                                                  constraints2); 
    
    coupling_sparsity.copy_from(dsp);
    coupling_matrix.reinit(coupling_sparsity);
  }

template <int dim>
void Step6<dim>::assemble_system_one_cell_omega(
  const typename DoFHandler<dim>::active_cell_iterator &  cell,
  ScratchData &                                           scratch,
  CopyData &                                              copy)
{
  auto &cell_matrix = copy.matrices[0];
  auto &cell_mass_matrix = copy.matrices[1];
  auto &cell_rhs    = copy.vectors[0];

  cell->get_dof_indices(copy.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  cell_matrix           = 0;
  cell_mass_matrix      = 0;
  cell_rhs              = 0;

  for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
      // const double current_coefficient_omega =
      //   coefficient_omega(fe_values.quadrature_point(q_index));
        // coefficient_omega(fe_values.quadrature_point(q_index));
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          {cell_matrix(i, j) +=
            (coefficient_omega *                            // a(x_q)
              fe_values.shape_grad(i, q_index) *       // grad phi_i(x_q)
              fe_values.shape_grad(j, q_index) *       // grad phi_j(x_q)
              fe_values.JxW(q_index));                 // dx
          cell_mass_matrix(i, j) +=
            ( fe_values.shape_value(i, q_index) *       // phi_i(x_q)
              fe_values.shape_value(j, q_index) *       // phi_j(x_q)
              fe_values.JxW(q_index));                 // dx
              }
      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += (fe_values.shape_value(i, q_index) *   // phi_i(x_q)
                        rhs1 *                                 // f(x)
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
  auto &cell_mass_matrix = copy.matrices[1];
  auto &cell_rhs    = copy.vectors[0];

  cell->get_dof_indices(copy.local_dof_indices[0]);

  const auto &fe_values = scratch.reinit(cell);
  cell_matrix          = 0;
  cell_mass_matrix     = 0;
  cell_rhs             = 0;

  for (const unsigned int q_index : fe_values.quadrature_point_indices())
    {
      // const double current_coefficient_omega =
      //   coefficient_omega(fe_values.quadrature_point(q_index));
        // coefficient_omega(fe_values.quadrature_point(q_index));
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          {cell_matrix(i, j) +=
              (((coefficient_omega2-coefficient_omega) *          // b2(x_q)-b(x_q)
                  fe_values[primal].gradient(i, q_index) *        // grad phi_i(x_q)_omega2
                  fe_values[primal].gradient(j, q_index))         // grad phi_j(x_q)_omega2

                -(fe_values[primal].value(i, q_index) *           // grad phi_i(x_q)_omega2
                  fe_values[multiplier].value(j, q_index) )       // grad phi_j(x_q)_omega

                -(fe_values[multiplier].value(i, q_index) *       // grad phi_i(x_q)_omega
                  fe_values[primal].value(j, q_index) )           // grad phi_j(x_q)_omega2

              )* fe_values.JxW(q_index)   ;                       // dx

          cell_mass_matrix(i,j) +=
              ( (fe_values[primal].value(i, q_index) *            // phi_i(x_q)
                fe_values[primal].value(j, q_index))              // phi_j(x_q)

              +(fe_values[multiplier].value(i, q_index) *         // phi_i(x_q)
                fe_values[multiplier].value(j, q_index))          // phi_j(x_q)

              )*fe_values.JxW(q_index)  ;                         // dx                 
            }


      for (const unsigned int i : fe_values.dof_indices())
        cell_rhs(i) += ((fe_values[primal].value(i, q_index) *   // phi_i(x_q)_onega2
                        (rhs2-rhs1))                             // f2(x)-f1(x)

                +(fe_values[multiplier].value(i, q_index) *     // phi_i(x_q)_onega2
                        (0.0))                                  // zero
         ) * fe_values.JxW(q_index);                            //dx
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
                          update_values| update_quadrature_points | update_hessians |
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
      // ScratchData`
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

    scratch.extract_local_dof_values("u_omega", u_omega);
    const auto &lap_u       = scratch.get_laplacians("u_omega", scalar);
    const auto &JxW         = scratch.get_JxW_values();

    scratch.extract_local_dof_values("u_omega2_prime",   u_omega2_prime);
    const auto &Lambda_prime  = scratch.get_values("u_omega2_prime", scalar);

        // Reset vectors
    float cell_indicator = 0;

        // Now store the values of the residual square in the copy data
    for (const auto q_index : fe_value.quadrature_point_indices())
    {
        const auto res =
          (coefficient_omega * lap_u[q_index]) - Lambda_prime[q_index] + rhs1;

        cell_indicator += H * H * res * res * JxW[q_index]; 
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
      const auto J = - coefficient_omega * jump_grad[q_index] * normals[q_index];

      face_indicator += 0.5 * h * J * J * JxW[q_index]; // dx
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
                              MeshWorker::assemble_own_interior_faces_once,
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
      // const ComponentSelectFunction<dim> pimal_mask(dim, dim + 1);
      // const ComponentSelectFunction<dim> multiplier_mask(dim ,dim + 1);

  
      ScratchData scratch(this->fe2,
                          quadrature_formula,
                          update_values |update_quadrature_points | update_hessians |
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
      // FEValuesExtractors::Scalar scalar(0);
      // FEValuesExtractors::Scalar scalar(1);

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
        scratch.extract_local_dof_values("u_omega2", u_omega2);

        // Get the values of the u_omega at the quadrature points
        const auto &u2          = scratch.get_values("u_omega2", primal);
        const auto &Lambda      = scratch.get_values("u_omega2", multiplier);
        const auto &grad_u2     = scratch.get_gradients("u_omega2", primal);
        const auto &lap_u2      = scratch.get_laplacians("u_omega2", primal);
        const auto &JxW         = scratch.get_JxW_values();


        // scratch.extract_local_dof_values("u_omega_prime", u_omega_prime);
        // const auto u_prime      = scratch.get_values("u_omega_prime", scalar);
        // const auto grad_u_prime = scratch.get_gradients("u_omega_prime", scalar);

        scratch.extract_local_dof_values("u_omega_prime", u_omega_prime);
        const auto u_prime      = scratch.get_values("u_omega_prime", multiplier);
        const auto grad_u_prime = scratch.get_gradients("u_omega_prime", multiplier);
        
        


        // scratch.extract_local_dof_values("u_omega_prime", u_omega_prime);
        // const auto u_prime      = scratch.get_values("u_omega_prime", primal);
        // const auto grad_u_prime = scratch.get_gradients("u_omega_prime", primal);




        // Reset vectors
        float cell_indicator = 0;

        // Now store the values of the residual square in the copy data
        for (const auto q_index : fe_value.quadrature_point_indices())
          {
            const auto res  =
              ((coefficient_omega2-coefficient_omega) * lap_u2[q_index]) +Lambda[q_index] +(rhs2-rhs1);
            const auto res2 = u_prime[q_index] - u2[q_index];
            const auto res3 = grad_u_prime[q_index] - grad_u2[q_index]; 
            cell_indicator += ( (H * H * res * res )+ (res2 * res2) + (res3 * res3) )* JxW[q_index]; 
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
          scratch.get_jumps_in_gradients("u_omega2", primal);

        const auto &JxW     = scratch.get_JxW_values();
        const auto &normals = scratch.get_normal_vectors();
        // const auto &q_points2 = scratch.get_quadrature_points();

        // Now store the values of the gradient jump in the copy data
        float face_indicator = 0;
        for (const auto q_index : fe_ivalue.quadrature_point_indices())
          {
            // const double current_coefficient_omega =
            //    coefficient_omega(q_points2[q_index]);
            const auto J = - (coefficient_omega2 - coefficient_omega) * jump_grad[q_index] * normals[q_index];

            face_indicator += 0.5 * h * J * J * JxW[q_index]; // dx
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
                              MeshWorker::assemble_own_interior_faces_once,
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
  constraints.distribute_local_to_global(copy.matrices[1],
                                          copy.local_dof_indices[0],
                                          M_omega);
}

template <int dim>
void
Step6<dim>::copy_one_cell_omega2(const CopyData &copy)
{
  constraints2.distribute_local_to_global(copy.matrices[0],
                                          copy.vectors[0],
                                          copy.local_dof_indices[0],
                                          B_omega2,
                                          rhs_omega2);
  constraints2.distribute_local_to_global(copy.matrices[1],
                                          copy.local_dof_indices[0],
                                          M_omega2);
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
  M_omega.compress(VectorOperation::add);
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

  B_omega2.compress(VectorOperation::add);
  M_omega2.compress(VectorOperation::add);
  rhs_omega2.compress(VectorOperation::add);
}

template<int dim>
void Step6<dim>::assemble_coupling_system()
{
  // TimerOutput::Scope timer_section(monitor, "Assemble coupling system");

  QGauss<dim> quad(3);
  NonMatching::create_coupling_mass_matrix(space_grid_tools_cache,
                                            omega_dh,
                                            omega2_dh,
                                            quad,
                                            coupling_matrix,
                                            constraints,
                                            ComponentMask(),
                                            ComponentMask(std::vector<bool>{false,true}),
                                            StaticMappingQ1<dim>::mapping,
                                            constraints2);
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
  // SolverControl            solver_control(1000, 1e-12);
  // SolverCG<Vector<double>> solver(solver_control);

  // PreconditionSSOR<SparseMatrix<double>> preconditioner2;
  // preconditioner2.initialize(B_omega2, 1.2);
  SparseDirectUMFPACK solver;
  solver.initialize(B_omega2);
  solver.vmult(u_omega2, rhs_omega2);
  // solver.solve(B_omega2, u_omega2, rhs_omega2, preconditioner2);

  constraints2.distribute(u_omega2);
  deallog << "u_omega2 L_infinity norm = " << u_omega2.linfty_norm() << "(u2 lambda)" << std::endl;
  deallog << "u_omega2 L_2 norm = " << u_omega2.l2_norm() << "(u2 lambda)" << std::endl;

}

template <int dim>
void Step6<dim>::solve()
{
  /// Start by creating the inverse stiffness matrix
    SparseDirectUMFPACK A_omega_inv_umfpack;
    A_omega_inv_umfpack.initialize(A_omega);
    SparseDirectUMFPACK B_omega2_inv_umfpack;
    B_omega2_inv_umfpack.initialize(B_omega2);
    SparseDirectUMFPACK M_omega_inv_umfpack;
    M_omega_inv_umfpack.initialize(M_omega);
    SparseDirectUMFPACK M_omega2_inv_umfpack;
    M_omega2_inv_umfpack.initialize(M_omega2);

    // Initializing the operators, as described in the introduction
    auto A1    = linear_operator(A_omega);
    auto M1    = linear_operator(M_omega);
    auto B     = linear_operator(B_omega2);
    auto M2    = linear_operator(M_omega2);
    auto C1t   = linear_operator(coupling_matrix);
    auto C1    = transpose_operator(C1t);

    using BVec = BlockVector<double>;
    using LinOp = decltype(A1);

    auto AA = block_operator<2, 2, BVec>({{{{A1, C1t}}, {{C1, B}}}});

    auto A1_inv  = linear_operator(A1 , A_omega_inv_umfpack );
    auto B_inv   = linear_operator(B  , B_omega2_inv_umfpack);
    auto M1_inv  = linear_operator(M1 , M_omega_inv_umfpack);
    auto M2_inv  = linear_operator(M2 , M_omega2_inv_umfpack);
    // const auto S     = A1 - ( C1t * B_inv * C1); 
    // // SparseDirectUMFPACK S_inv_umfpack;
    // // S_inv_umfpack.initialize(S);
    // auto S_inv  = inverse_operator(S);


     auto X  = -1.0 * B_inv  *   C1  * A1_inv ;
    //  auto X2 = -1.0 * A1_inv *   C1t  * B_inv ;
     auto low_tri_prec                      = block_operator<2, 2, BVec>({{{{A1_inv, 0.0 * C1t}}, {{X, B_inv}}}});
    //  auto up_tri_prec                       = block_operator<2, 2, BVec>({{{{A1_inv,  X2}}, {{0.0 * C1, B_inv}}}});
    //  auto full_prec                            = block_operator<2, 2, BVec>({{{{A1_inv,  X2}}, {{X, B_inv}}}});
    //  auto sur_prec                         = block_operator<2, 2, BVec>({{{{S_inv,  - 1.0 * S_inv * B_inv * C1t }},{{-1.0 * B_inv * C1 *  S_inv, B_inv +(B_inv * C1 * S_inv * C1t * B_inv)}}}});

    //  std::array<LinOp, 2> diag_ops = {{A1_inv, B_inv}};
    //  auto diagprecAA                        = block_diagonal_operator<2, BVec>(diag_ops);
    //  std::array<LinOp, 2> diag_sur = {{A1_inv, S_inv}};
    //  auto sur_prec                        = block_diagonal_operator<2, BVec>(diag_sur);



    std::array<LinOp, 2> diag_MM = {{M1_inv, M2_inv}};
    auto MM                      = block_diagonal_operator<2, BVec>(diag_MM);
    auto CC                      = block_operator<2, 2, BVec>({{{{0.0* A1, C1t}}, {{C1, 0.0 *B}}}});

    SolverControl            solver_control(2000, 1e-12);
    SolverGMRES<BVec> solver(solver_control);

    BVec system_rhs;
    BVec solution;
    BVec solution_prime;
    AA.reinit_domain_vector(system_rhs, false);
    AA.reinit_range_vector(solution, false);
    AA.reinit_range_vector(solution_prime, false);


    // VectorTools::interpolate(omega2_dh, Functions::CosineFunction<dim>(2), solution.block(1));
    // system_rhs = 1.0;
    // solution.block(0) = -1.0;
    // system_rhs.block(1) = B * solution.block(1);
    // solution.block(1) = B_inv * system_rhs.block(1);
    // solution.block(1) = C1 * system_rhs.block(0);
    
    // solution = 1.0;
    // system_rhs = AA*solution;
    // deallog << "1: " << solution.l2_norm() << std::endl;
    // deallog << "A*1 = " << system_rhs.l2_norm() << std::endl;

    // solution = diagprecAA * system_rhs; 
    // deallog << "diagAinv * (A*1) = " << solution.l2_norm() << std::endl
    // << "A1 norm: " << A_omega.l1_norm() << std::endl
    // deallog << "B norm: " << B_omega2.l1_norm() << std::endl;
    // deallog << "solution2 norm: " << (solution.block(1)).linfty_norm() << std::endl;
    // << "Coupling norm: " << coupling_matrix.l1_norm() << std::endl;

    system_rhs.block(0) = rhs_omega;
    system_rhs.block(1) = rhs_omega2;
    // deallog << "Rhs norm: " << system_rhs.l2_norm() << std::endl;

    // solver.solve(AA, solution, system_rhs, full_prec);
    solver.solve(AA, solution, system_rhs, low_tri_prec);
    // solver.solve(AA, solution, system_rhs, sur_prec);
    // solver.solve(AA, solution, system_rhs, up_tri_prec);
    // solver.solve(AA, solution, system_rhs, diagprecAA);

    u_omega = solution.block(0);
    u_omega2 = solution.block(1);
    // deallog << "sol norm: " << solution.linfty_norm() << std::endl;
    constraints.distribute(u_omega);
    constraints2.distribute(u_omega2);



    solution_prime = MM *  CC * solution;
    u_omega_prime = solution_prime.block(1);
    u_omega2_prime = solution_prime.block(0);

    // deallog << "u_omega L_infinity norm = " << u_omega.linfty_norm() << "(u1)" << std::endl;
    // deallog << "u_omega2 L_infinity norm = " << u_omega2.linfty_norm() << "(u2 lambda)" << std::endl;
    // deallog << "u_omega_prime L_infinity norm = " << u_omega_prime.linfty_norm() << "(0 u1_extended)" << std::endl;
    // deallog << "u_omega2_prime L_infinity norm = " << u_omega2_prime.linfty_norm() << "(lambda extended)" << std::endl;
    // deallog << "u_omega L_2 norm = " << u_omega.l2_norm() << "(u1)"  << std::endl;
    // deallog << "u_omega2 L_2 norm = " << u_omega2.l2_norm() << "(u2 lambda)" << std::endl;
    // deallog << "u_omega_prime L_2 norm = " << u_omega_prime.l2_norm() << "(0 u1_extended)" << std::endl;
    // deallog << "u_omega2_prime L_2 norm = " << u_omega2_prime.l2_norm() << "(lambda extended)" << std::endl;

}

// template <int dim>
// void Step6<dim>::solve()
// {

//   /// Start by creating the inverse stiffness matrix
 
//     // SparseDirectUMFPACK C_inv_umfpack;
//     // C_inv_umfpack.initialize(coupling_matrix);
//     A_omega2 = B_omega2.block(0, 0);
//     C_omega2 = B_omega2.block(0, 1);
//     // Initializing the operators, as described in the introduction
//     auto A1  = linear_operator(A_omega);
//     auto A2  = linear_operator(A_omega2);
//     auto C2t = linear_operator(C_omega2);
//     auto C2  = transpose_operator(C2t);  
//     auto C1t = linear_operator(coupling_matrix);
//     auto C1  = transpose_operator(C1t);


//     // const auto &u_2 = u_omega2.block(0);
//     // const auto &mu = u_omega2.block(1);

//     // auto &U2 = rhs_omega2.block(0);
//     // auto &O = rhs_omega2.block(1);

//     using BVec = BlockVector<double>;
//     using LinOp = decltype(A1);

//     auto AA = block_operator<3, 3, BVec>({{  {{A1   , Zero  ,C1t  }} ,
//                                              {{Zero , A2    ,  C2t}} ,
//                                              {{C1   , C2    , Zero}}  
//                                          }});

//     SparseDirectUMFPACK A_omega_inv_umfpack;
//     A_omega_inv_umfpack.initialize(A_omega);
//     SparseDirectUMFPACK A2_omega2_inv_umfpack;
//     A2_omega2_inv_umfpack.initialize(B_omega2);

//     auto A1_inv = linear_operator(A1, A_omega_inv_umfpack);
//     auto A2_inv   = linear_operator(A2, A2_omega2_inv_umfpack);

//     std::array<LinOp, 2> diag_ops = {{A1_inv, A2_inv}};
//     auto diagprecAA               = block_diagonal_operator<2, BVec>(diag_ops);

//     SolverControl            solver_control(1000, 1e-12, true, true);
//     SolverGMRES<BVec> solver(solver_control);

//     BVec system_rhs;
//     BVec solution;
//     AA.reinit_domain_vector(system_rhs, false);
//     AA.reinit_range_vector(solution, false);

//     // solution = 1.0;
//     // system_rhs = AA*solution;
//     // deallog << "1: " << solution.l2_norm() << std::endl;
//     // deallog << "A*1 = " << system_rhs.l2_norm() << std::endl;

//     // solution = diagprecAA * system_rhs; 
//     // deallog << "diagAinv * (A*1) = " << solution.l2_norm() << std::endl
//     // << "A1 norm: " << A_omega.l1_norm() << std::endl
//     // << "A2 norm: " << B_omega2.l1_norm() << std::endl
//     // << "Coupling norm: " << coupling_matrix.l1_norm() << std::endl;

//     system_rhs.block(0) = rhs_omega;
//     system_rhs.block(1) = rhs_omega2;
//     deallog << "Rhs norm: " << system_rhs.l2_norm() << std::endl;

//     solver.solve(AA, solution, system_rhs, diagprecAA);

//     u_omega = solution.block(0);
//     u_omega2 = solution.block(1);

//     constraints.distribute(u_omega);
//     constraints2.distribute(u_omega2);
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

  
  triangulation_omega2.execute_coarsening_and_refinement();
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
    data2_out.write_vtu(output2);

  }
}

template <int dim>
void Step6<dim>::run()
{
  deallog.depth_console(10);
  deallog.push("RUN");

  std::ofstream outfile("indicator.txt");
  std::ofstream outfile2("indicator2.txt");
  std::ofstream outfile3("total_indicator.txt");

  for (unsigned int cycle = 0; cycle < 9 ; ++cycle)
    {
      deallog << "Cycle " << cycle << std::endl;
      if (cycle == 0)
      {
        make_grid_omega();
        triangulation_omega.refine_global(3);
        make_grid_omega2();
        triangulation_omega2.refine_global(3);
      }
      else
      {
        refine_omega();
        refine_omega2();
      }

      setup_system_omega();
      setup_system_omega2();
      setup_coupling();

      assemble_system_omega();
      assemble_system_omega2();
      assemble_coupling_system();

      // solve_u1();
      // solve_u2();
      solve();

      estimator1();
      estimator2();
      //step 74 for energy norm
      outfile << omega_dh.n_dofs() << " "
             << std::sqrt(error_per_cell_omega.l1_norm()) << " "
             << std::sqrt(error_per_cell_omega.linfty_norm()) << std::endl;
      output_results(cycle);
      outfile2 << omega2_dh.n_dofs() << " "
             << std::sqrt(error_per_cell_omega2.l1_norm()) << " "
             << std::sqrt(error_per_cell_omega2.linfty_norm()) << std::endl;
      output_results(cycle);
      outfile3 << omega2_dh.n_dofs() + omega_dh.n_dofs()<< " "
             << std::sqrt(error_per_cell_omega.l1_norm() + error_per_cell_omega2.l1_norm()) << std::endl;
      output_results(cycle);
      }
    deallog.pop();
    outfile.close();
    outfile2.close();
    outfile3.close();
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
