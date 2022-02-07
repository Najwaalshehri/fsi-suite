#include <deal.II/base/config.h>

#include "parsed_tools/enum.h"

#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/fe/fe_values.h>

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

using namespace dealii;

TEST(ParsedEnum, CheckFEValuesFlags)
{
  UpdateFlags flags;

  ParameterAcceptor::prm.add_parameter("Update flags", flags);
  ParameterAcceptor::prm.parse_input_from_string(R"(
    set Update flags = update_values
  )");

  ASSERT_TRUE(flags & update_values)
    << Patterns::Tools::Convert<UpdateFlags>::to_string(flags);

  ParameterAcceptor::prm.parse_input_from_string(R"(
    set Update flags = update_values | update_gradients
  )");

  auto s = Patterns::Tools::Convert<UpdateFlags>::to_string(flags);
  ASSERT_EQ(s, "update_default| update_values| update_gradients");
  ASSERT_TRUE(flags & (update_values | update_gradients));
}