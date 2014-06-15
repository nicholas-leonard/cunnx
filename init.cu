#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#define CudaAssert( expression ) \
if ( !(expression)) { \
printf( "Assert failed %d:%d at %s:%d\n", blockIdx.x, threadIdx.x,  __FILE__, __LINE__ ); \
}

#include "SoftMaxTree.cu"
#include "BlockSparse.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libcunnx(lua_State *L);

int luaopen_libcunnx(lua_State *L)
{
  lua_newtable(L);
  
  cunnx_SoftMaxTree_init(L);
  cunnx_BlockSparse_init(L);

  return 1;
}
