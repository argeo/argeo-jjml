#include <string>

#include <ggml-backend.h>

#include <argeo/jni/argeo_jni_encoding.h>

#include "org_argeo_jjml_ggml_GgmlBackend.h" // IWYU pragma: keep

/** UTF-16 converter. */
static argeo::jni::utf16_convert utf16_conv;

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doLoadBackend(
		JNIEnv*, jclass, jstring) {
	return 0;
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_ggml_GgmlBackend_doLoadAllBackends(
		JNIEnv *env, jclass, jstring basePath) {
	std::string search_path = argeo::jni::to_string(env, basePath, &utf16_conv);
	ggml_backend_load_all_from_path(search_path.c_str());
}
