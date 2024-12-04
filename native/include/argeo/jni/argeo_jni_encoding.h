#ifndef argeo_jni_encoding_h
#define argeo_jni_encoding_h

#include <locale>
#include <string>
#include <type_traits>

#include <jni.h>

namespace argeo::jni {
// TYPES
/** Utility wrapper to adapt locale-bound facets for wstring/wbuffer convert
 *
 * @see https://en.cppreference.com/w/cpp/locale/codecvt
 */
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
 *	std::string text = ...
 *	std::u16string u16text = utf16_converter.from_bytes(text);
 * </code>
 */
typedef std::wstring_convert<
		argeo::jni::deletable_facet<std::codecvt<char16_t, char, std::mbstate_t>>,
		char16_t> utf16_convert;

/** Convenience method to make casting more readable in code.*/
inline std::u16string jchars_to_utf16(const jchar *jchars, const jsize length) {
	// sanity check
	static_assert(sizeof(char16_t) == sizeof(jchar));

	const char16_t *u16chars = reinterpret_cast<const char16_t*>(jchars);
	std::u16string u16text(u16chars, static_cast<size_t>(length));
	return u16text;
}

/** Convert a jstring to an UTF-16 string.
 * Note that the characters are copied,
 *  use jchars_to_utf16 in order to work directly
 *  on the characters returned by GetString / GetStringCritical.
 */
inline std::u16string jstring_to_utf16(JNIEnv *env, jstring str) {
	jsize length = env->GetStringLength(str);
	jchar buf[length];
	env->GetStringRegion(str, 0, length, buf);
	return argeo::jni::jchars_to_utf16(buf, length);
}

/** Convert a jstring to a string, via UTF-16.
 * Note that the characters are copied.
 */
inline std::string to_string(JNIEnv *env, jstring str,
		utf16_convert *utf16_converter) {
	std::u16string u16text = jstring_to_utf16(env, str);
	return utf16_converter->to_bytes(u16text);
}

/** Convenience method to make casting more readable in code.*/
inline const jchar* utf16_to_jchars(std::u16string u16text) {
	// sanity check
	static_assert(sizeof(char16_t) == sizeof(jchar));

	return reinterpret_cast<const jchar*>(u16text.data());
}

/** UTF-16 string to Java string. No conversion is needed, as this is the default format.*/
inline jstring utf16_to_jstring(JNIEnv *env, std::u16string u16text) {
	return env->NewString(utf16_to_jchars(u16text),
			static_cast<jsize>(u16text.size()));
}

/** Converts a string to a jstring. */
inline jstring to_jstring(JNIEnv *env, std::string str,
		utf16_convert *utf16_converter) {
	std::u16string u16text = utf16_converter->from_bytes(str);
	return utf16_to_jstring(env, u16text);
}

}  // namespace argeo::jni
#endif
