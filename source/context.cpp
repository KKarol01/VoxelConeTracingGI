#include "context.hpp"

static Context ctx{};

Context& get_context() {
    return ctx;
}