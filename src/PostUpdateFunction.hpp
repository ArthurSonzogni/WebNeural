#ifndef POST_UPDATE_FUNCTION_H
#define POST_UPDATE_FUNCTION_H

#include <functional>

class Model;
class Node;

namespace PostUpdateFunction {

using F = std::function<void(Model*)>;

F None();
F ClipWeight(Node* begin, Node* end);
F GradientPenalty(Node* begin, Node* end, float penalty);

}  // namespace PostUpdateFunction

#endif /* end of include guard: POST_UPDATE_FUNCTION_H */
