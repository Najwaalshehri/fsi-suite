#include "tools/parsed_grid_generator.h"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

#include "dim_spacedim_tester.h"

using namespace dealii;

TYPED_TEST(DimSpacedimTester, GenerateHyperCube)
{
  Tools::ParsedGridGenerator<this->dim, this->spacedim> pgg("/");
  Triangulation<this->dim, this->spacedim>              tria;

  this->parse(R"(
    set Input name = hyper_cube
    set Arguments = 0: 1: false
    set Transform to simplex grid = false
  )");

  std::string grid_name = "grid_" + std::to_string(this->dim) +
                          std::to_string(this->spacedim) + ".msh";

  this->parse("set Output name = " + grid_name);


  // After this, we should have a file grid.msh
  pgg.generate(tria);
  ASSERT_TRUE(std::ifstream(grid_name));
  std::remove(grid_name.c_str());

  // And the grid should have 1 element
  ASSERT_EQ(tria.n_active_cells(), 1u);
}



TYPED_TEST(DimSpacedimTester, GenerateHyperCubeSimplex)
{
  Tools::ParsedGridGenerator<this->dim, this->spacedim> pgg("/");
  Triangulation<this->dim, this->spacedim>              tria;

  this->parse(R"(
    set Input name = hyper_cube
    set Arguments = 0: 1: false
    set Transform to simplex grid = true
  )");

  std::string grid_name = "grid_" + std::to_string(this->dim) +
                          std::to_string(this->spacedim) + ".msh";

  this->parse("set Output name = " + grid_name);

  // After this, we should have a file grid.msh
  pgg.generate(tria);
  ASSERT_TRUE(std::ifstream(grid_name));
  std::remove(grid_name.c_str());

  // And the grid should have 8 elements in 2d, and 24 in 3d
  const unsigned int dims[] = {0, 1, 8, 24};
  ASSERT_EQ(tria.n_active_cells(), dims[this->dim]);
}