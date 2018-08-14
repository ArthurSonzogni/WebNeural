#ifndef LOSSFUNCTION_H
#define LOSSFUNCTION_H

#include "Tensor.hpp"

namespace LossFunction {

using F = void(const Tensor&,  // target
               const Tensor&,  // current
               float*,         // error
               Tensor*);       // derivative

// 
// ⌠        2
// ⌡ (y - z)  ⋅ dy
F SquaredDifference; // S

// ⌠
// ⌡ y ⋅ log(z) ⋅ dy
//
//               ⎛1  ⎞
//               ⎜   ⎟
//               ⎜1  ⎟
// give that z ⋅ ⎜   ⎟ = 1
//               ⎜...⎟
//               ⎜   ⎟
//               ⎝1  ⎠
F CrossEntropy;

F SoftmaxCrossEntropy;

// ⌠
// ⌡ (y > 0.5) z dy- (y<0.5) z dy;
F WasserStein;

} // namespace LossFunction

#endif /* end of inclusde guard: LOSSFUNCTION_H */
