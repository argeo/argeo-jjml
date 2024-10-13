#ifndef argeo_jni_h
#define argeo_jni_h

#include <bits/types/mbstate_t.h>
#include <jni.h>
#include <jni_md.h>
#include <fstream>
#include <locale>
#include <string>
#include <type_traits>

namespace argeo::jni {
// TYPES
/** Utility wrapper to adapt locale-bound facets for wstring/wbuffer convert
 *
 * @see https://en.cppreference.com/w/cpp/locale/codecvt
 */
/*
 * ENCODING
 */
// TYPES
// We need to use this wrapper in order to have an UTF-16 converter which can be instantiated.
// see https://en.cppreference.com/w/cpp/locale/codecvt
// TODO Make sure that there is no better way.
template<class Facet>
struct deletable_facet: Facet {
	template<class ... Args>
	deletable_facet(Args &&... args) :
			Facet(std::forward<Args>(args)...) {
	}
	~deletable_facet() {
	}
};

/** A converter from std::string to std::u16string.
 *
 * Usage:
 * <code>
 * </code>*/
typedef std::wstring_convert<
		argeo::jni::deletable_facet<std::codecvt<char16_t, char, std::mbstate_t>>,
		char16_t> utf16_convert;

/** Convenience method to make casting more readable in code.*/
inline std::u16string jcharsToUtf16(const jchar *jchars) {
	// sanity check
	static_assert(sizeof(char16_t) == sizeof(jchar));

	const char16_t *u16chars = reinterpret_cast<const char16_t*>(jchars);
	std::u16string u16text(u16chars);
	return u16text;
}

// METHODS
/** Convenience method to make casting more readable in code.*/
inline const jchar* utf16ToJchars(std::u16string u16text) {
	// sanity check
	static_assert(sizeof(char16_t) == sizeof(jchar));

	return reinterpret_cast<const jchar*>(u16text.data());
}

/** UTF-16 string to Java string. No conversion is needed, as this is the default format.*/
inline jstring utf16ToJstring(JNIEnv *env, std::u16string u16text) {
	return env->NewString(utf16ToJchars(u16text), u16text.size());
}

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
