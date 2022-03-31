// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by Luca Heltai
//
// This file is part of the FSI-suite platform, based on the deal.II library.
//
// The FSI-suite platform is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 3.0 of the License,
// or (at your option) any later version. The full text of the license can be
// found in the file LICENSE at the top level of the FSI-suite platform
// distribution.
//
// ---------------------------------------------------------------------

#ifdef DEAL_II_WITH_CGAL

#  include "create_coupling_sparsity_pattern_with_exact_intersections.h"

#  include "compute_intersections.h"


using namespace dealii;
namespace dealii::NonMatching
{
  template <int dim0,
            int dim1,
            int spacedim,
            typename Sparsity,
            typename number>
  void
  create_coupling_sparsity_pattern_with_exact_intersections(
    const std::vector<std::tuple<
      typename dealii::Triangulation<dim0, spacedim>::active_cell_iterator,
      typename dealii::Triangulation<dim1, spacedim>::active_cell_iterator,
      dealii::Quadrature<spacedim>>> &intersections_info,
    const DoFHandler<dim0, spacedim> &space_dh,
    const DoFHandler<dim1, spacedim> &immersed_dh,
    Sparsity &                        sparsity,
    const AffineConstraints<number> & constraints,
    const ComponentMask &             space_comps,
    const ComponentMask &             immersed_comps,
    const AffineConstraints<number> & immersed_constraints)
  {
    AssertDimension(sparsity.n_rows(), space_dh.n_dofs());
    AssertDimension(sparsity.n_cols(), immersed_dh.n_dofs());
    Assert(dim1 <= dim0,
           ExcMessage("This function can only work if dim1 <= dim0"));
    Assert((dynamic_cast<
              const parallel::distributed::Triangulation<dim1, spacedim> *>(
              &immersed_dh.get_triangulation()) == nullptr),
           ExcNotImplemented());



    const auto &       space_fe                 = space_dh.get_fe();
    const auto &       immersed_fe              = immersed_dh.get_fe();
    const unsigned int n_space_dofs             = space_fe.n_dofs_per_cell();
    const unsigned int n_immersed_dofs          = immersed_fe.n_dofs_per_cell();
    const unsigned int n_space_fe_components    = space_fe.n_components();
    const unsigned int n_immersed_fe_components = immersed_fe.n_components();
    std::vector<types::global_dof_index> space_dofs(n_space_dofs);
    std::vector<types::global_dof_index> immersed_dofs(n_immersed_dofs);


    const ComponentMask space_c =
      (space_comps.size() == 0 ? ComponentMask(n_space_fe_components, true) :
                                 space_comps);
    const ComponentMask immersed_c =
      (immersed_comps.size() == 0 ?
         ComponentMask(n_immersed_fe_components, true) :
         immersed_comps);
    // Global 2 Local indices
    std::vector<unsigned int> space_gtl(n_space_fe_components);
    std::vector<unsigned int> immersed_gtl(n_immersed_fe_components);
    for (unsigned int i = 0, j = 0; i < n_space_fe_components; i++)
      {
        if (space_c[i])
          space_gtl[i] = j++;
      }


    for (unsigned int i = 0, j = 0; i < n_immersed_fe_components; i++)
      {
        if (immersed_c[i])
          immersed_gtl[i] = j++;
      }



    Table<2, bool> dof_mask(n_space_dofs, n_immersed_dofs);
    dof_mask.fill(false); // start off by assuming they don't couple

    for (unsigned int i = 0; i < n_space_dofs; ++i)
      {
        const auto comp_i = space_fe.system_to_component_index(i).first;
        if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
          {
            for (unsigned int j = 0; j < n_immersed_dofs; ++j)
              {
                const auto comp_j =
                  immersed_fe.system_to_component_index(j).first;
                if (immersed_gtl[comp_j] == space_gtl[comp_i])
                  {
                    dof_mask(i, j) = true;
                  }
              }
          }
      }

    // Whenever the BB space_cell intersects the BB of an embedded cell, those
    // DoFs have to be recorded
    for (const auto &it : intersections_info)
      {
        const auto &space_cell    = std::get<0>(it);
        const auto &immersed_cell = std::get<1>(it);
        typename DoFHandler<dim0, spacedim>::cell_iterator space_cell_dh(
          *space_cell, &space_dh);
        typename DoFHandler<dim1, spacedim>::cell_iterator immersed_cell_dh(
          *immersed_cell, &immersed_dh);

        space_cell_dh->get_dof_indices(space_dofs);
        immersed_cell_dh->get_dof_indices(immersed_dofs);


        constraints.add_entries_local_to_global(space_dofs,
                                                immersed_constraints,
                                                immersed_dofs,
                                                sparsity,
                                                true,
                                                dof_mask);
      }
  }

} // namespace dealii::NonMatching


#else

template <int dim0, int dim1, int spacedim, typename Sparsity, typename number>
void
dealii::NonMatching::create_coupling_sparsity_pattern_with_exact_intersections(
  const std::vector<std::tuple<
    typename dealii::Triangulation<dim0, spacedim>::active_cell_iterator,
    typename dealii::Triangulation<dim1, spacedim>::active_cell_iterator,
    dealii::Quadrature<spacedim>>> &intersections_info,
  const DoFHandler<dim0, spacedim> &space_dh,
  const DoFHandler<dim1, spacedim> &immersed_dh,
  Sparsity &                        sparsity,
  const AffineConstraints<number> & constraints,
  const ComponentMask &             space_comps,
  const ComponentMask &             immersed_comps,
  const AffineConstraints<number> & immersed_constraints)
{
  Assert(false,
         ExcMessage("This function needs CGAL to be installed, "
                    "but cmake could not find it."));
  return {};
}

#endif



template void
dealii::NonMatching::create_coupling_sparsity_pattern_with_exact_intersections(
  const std::vector<
    std::tuple<typename dealii::Triangulation<2, 2>::active_cell_iterator,
               typename dealii::Triangulation<1, 2>::active_cell_iterator,
               dealii::Quadrature<2>>> &intersections_info,
  const DoFHandler<2, 2> &              space_dh,
  const DoFHandler<1, 2> &              immersed_dh,
  DynamicSparsityPattern &              sparsity,
  const AffineConstraints<double> &     constraints,
  const ComponentMask &                 space_comps,
  const ComponentMask &                 immersed_comps,
  const AffineConstraints<double> &     immersed_constraint);


template void
dealii::NonMatching::create_coupling_sparsity_pattern_with_exact_intersections(
  const std::vector<
    std::tuple<typename dealii::Triangulation<2, 2>::active_cell_iterator,
               typename dealii::Triangulation<2, 2>::active_cell_iterator,
               dealii::Quadrature<2>>> &intersections_info,
  const DoFHandler<2, 2> &              space_dh,
  const DoFHandler<2, 2> &              immersed_dh,
  DynamicSparsityPattern &              sparsity,
  const AffineConstraints<double> &     constraints,
  const ComponentMask &                 space_comps,
  const ComponentMask &                 immersed_comps,
  const AffineConstraints<double> &     immersed_constraint);

template void
dealii::NonMatching::create_coupling_sparsity_pattern_with_exact_intersections(
  const std::vector<
    std::tuple<typename dealii::Triangulation<3, 3>::active_cell_iterator,
               typename dealii::Triangulation<2, 3>::active_cell_iterator,
               dealii::Quadrature<3>>> &intersections_info,
  const DoFHandler<3, 3> &              space_dh,
  const DoFHandler<2, 3> &              immersed_dh,
  DynamicSparsityPattern &              sparsity,
  const AffineConstraints<double> &     constraints,
  const ComponentMask &                 space_comps,
  const ComponentMask &                 immersed_comps,
  const AffineConstraints<double> &     immersed_constraint);

template void
dealii::NonMatching::create_coupling_sparsity_pattern_with_exact_intersections(
  const std::vector<
    std::tuple<typename dealii::Triangulation<3, 3>::active_cell_iterator,
               typename dealii::Triangulation<3, 3>::active_cell_iterator,
               dealii::Quadrature<3>>> &intersections_info,
  const DoFHandler<3, 3> &              space_dh,
  const DoFHandler<3, 3> &              immersed_dh,
  DynamicSparsityPattern &              sparsity,
  const AffineConstraints<double> &     constraints,
  const ComponentMask &                 space_comps,
  const ComponentMask &                 immersed_comps,
  const AffineConstraints<double> &     immersed_constraint);
