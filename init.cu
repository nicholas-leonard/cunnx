#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <assert.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "SoftMaxTree.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcunnx(lua_State *L);

int luaopen_libcunnx(lua_State *L)
{
  lua_newtable(L);
  
  cunnx_SoftMaxTree_init(L);

  return 1;
}
