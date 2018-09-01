#ifndef POST_UPDATE_FUNCTION_H
#define POST_UPDATE_FUNCTION_H

class Model;

namespace PostUpdateFunction {

using F = void(Model*);

F None;
F ClipWeight;

}  // namespace PostUpdateFunction

#endif /* end of include guard: POST_UPDATE_FUNCTION_H */
