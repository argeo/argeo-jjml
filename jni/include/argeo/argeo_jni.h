
#ifndef argeo_jni_h
#define argeo_jni_h

#include <jni_md.h>
#include <type_traits>

namespace argeo::jni {

/** Cast a jlong to a pointer. */
template<typename T>
inline T getPointer(jlong pointer) {
	static_assert(std::is_pointer<T>::value);
	// Check the (unlikely) case where casting pointers as jlong would fail,
	// since it is relied upon in order to map Java and native structures
	// TODO provide alternative mechanism, such as a registry?
	static_assert(sizeof(T) <= sizeof(jlong));
	return reinterpret_cast<T>(pointer);
}


}  // namespace argeo::jni

#endif
