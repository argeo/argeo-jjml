#ifndef argeo_jni_h
#define argeo_jni_h

#include <jni.h>
#include <jni_md.h>
#include <cstdarg>
#include <fstream>
#include <locale>
#include <string>
#include <type_traits>

/*
 * PRE-PROCESSING
 */
//#define ARGEO_PERF 1
#ifdef ARGEO_PERF
#include <chrono>
#include <iomanip>
#include <iostream>

#define PERF_DURATION(_begin_) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - _begin_).count()
#define PERF_NS(_begin_, msg) "## ARGEO_PERF ## " << std::setfill(' ') << std::setw(16) << PERF_DURATION(_begin_) << " ns " << msg << std::endl
#define PERF_BEGIN() auto _begin_ = std::chrono::high_resolution_clock::now()
#define PERF_END(msg) std::cout << PERF_NS(_begin_, msg)
#else
#define PERF_DURATION(_begin_)
#define PERF_NS(_begin_, msg)
#define PERF_BEGIN()
#define PERF_END(msg)
#endif

/*
 * NAMESPACE INDEPENDENT
 */
#define LongSupplier$getAsLong(env) env->GetMethodID( \
		env->FindClass("java/util/function/LongSupplier"), "getAsLong", "()J")
extern jmethodID LongSupplier$getAsLong;

namespace argeo::jni {
/*
 * EXCEPTION HANDLING
 */
#define RuntimeException(env) env->FindClass("java/lang/RuntimeException")
#define IllegalArgumentException(env) env->FindClass("java/lang/IllegalArgumentException")
#define IndexOutOfBoundException(env) env->FindClass("java/lang/IndexOutOfBoundException")

/** Catches a standard C++ exception and throws an unchecked Java exception.
 * Does nothing if a Java execution is already pending
 * (that is, the already thrown java exception has priority.
 * A best effort is made to use an appropriate standard Java exception type.
 * Returns a <code>nullptr</code> as a convenience,
 * so that a call to it can be directly returned as the result of a (failed) non-void C++ function.
 * */
inline nullptr_t throw_to_java(JNIEnv *env, const std::exception &ex) {
	if (env->ExceptionCheck())
		return nullptr;
	if (typeid(std::invalid_argument) == typeid(ex)) {
		env->ThrowNew(IllegalArgumentException(env), ex.what());
	} else if (typeid(std::range_error) == typeid(ex)) {
		env->ThrowNew(IndexOutOfBoundException(env), ex.what());
	} else {
		env->ThrowNew(RuntimeException(env), ex.what());
	}
	return nullptr;
}
/*
 * SAFE FIELDS AN METHODS ACCESSORS
 */
inline jfieldID GetFieldID(JNIEnv *env, jclass clazz, const char *name,
		const char *sig) {
	return env->GetFieldID(clazz, name, sig);
}

/*
 * CALLBACKS
 */
/**
 * A structure holding the reference required for executing callback in the Java code.
 */
typedef struct java_callback {
	jobject callback = nullptr;
	jmethodID method = nullptr;
	JavaVM *jvm = nullptr;
} java_callback;

/**
 * Return the JNI environment for the current thread or attach it if it was detached.
 * @return true if the current thread was detached before this call.
 */
inline bool load_thread_jnienv(JavaVM *jvm, void **penv) {
	jint status = jvm->GetEnv(penv, JNI_VERSION_10);
	bool wasDetached = true;
	if (JNI_EVERSION == status) {
		// JNI version unsupported
		// TODO throw exception?
		wasDetached = true;
	} else if (JNI_EDETACHED == status) {
		wasDetached = true;
		jvm->AttachCurrentThreadAsDaemon(penv, nullptr);
	} else {
		//std::assert(threadEnv != nullptr,"");
		wasDetached = false;
	}
	return wasDetached;
}

/** Calls the callback's method on the provided Java object.*/
inline jboolean exec_boolean_callback(java_callback *cb, ...) {
	JNIEnv *threadEnv;
	bool wasDetached = load_thread_jnienv(cb->jvm, (void**) &threadEnv);
	jobject obj = threadEnv->NewLocalRef(cb->callback);

	// process variable arguments
	va_list args;
	jboolean result;
	va_start(args, cb);
	result = threadEnv->CallBooleanMethodV(obj, cb->method, args);
	va_end(args);

	if (wasDetached)
		cb->jvm->DetachCurrentThread();
	return result;
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

/** Cast a java.util.function.LongSupplier to a native pointer.
 *
 * @param env the JNI environment
 * @param reference the Java object implementing java.util.function.LongSupplier
 * @return the native pointer
 */
template<typename T>
inline T getPointer(JNIEnv *env, jobject reference) {
	jlong pointer = env->CallLongMethod(reference, LongSupplier$getAsLong);
	return getPointer<T>(pointer);
}

/*
 * ENCODING
 */
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
 * </code>*/
typedef std::wstring_convert<
		argeo::jni::deletable_facet<std::codecvt<char16_t, char, std::mbstate_t>>,
		char16_t> utf16_convert;

/** Convenience method to make casting more readable in code.*/
inline std::u16string jcharsToUtf16(const jchar *jchars, const jsize length) {
	// sanity check
	static_assert(sizeof(char16_t) == sizeof(jchar));

	const char16_t *u16chars = reinterpret_cast<const char16_t*>(jchars);
	std::u16string u16text(u16chars, static_cast<size_t>(length));
	return u16text;
}

inline std::u16string jstringToUtf16(JNIEnv *env, jstring str) {
	return argeo::jni::jcharsToUtf16(env->GetStringChars(str, nullptr),
			env->GetStringLength(str));
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
	return env->NewString(utf16ToJchars(u16text),
			static_cast<jsize>(u16text.size()));
}

}  // namespace argeo::jni
#endif
