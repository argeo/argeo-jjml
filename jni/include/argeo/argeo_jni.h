#ifndef argeo_jni_h
#define argeo_jni_h

#include <jni_md.h>
#include <fstream>
#include <type_traits>

namespace argeo::jni {
/*
 * ENCODING
 */
/**
 * Required in order to override destructor in order to use encoding conversions.
 *
 * Usage:
 * <code></code>
 */
template<class I, class E, class S>
struct codecvt: std::codecvt<I, E, S> {
	~codecvt() {
	}
};

/** Utility wrapper to adapt locale-bound facets for wstring/wbuffer convert
 *
 * @see https://en.cppreference.com/w/cpp/locale/codecvt
 */
template<class Facet>
struct deletable_facet: Facet {
	template<class ... Args>
	deletable_facet(Args &&... args) :
			Facet(std::forward<Args>(args)...) {
	}
	~deletable_facet() {
	}
};

/*
 * POINTERS
 */

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
