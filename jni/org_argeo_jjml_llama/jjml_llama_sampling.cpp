#include <llama.h>

#include <argeo/jni/argeo_jni.h>
#include <argeo/jni/argeo_jni_encoding.h>

#include "org_argeo_jjml_llama_.h"
#include "org_argeo_jjml_llama_LlamaCppNativeSampler.h" // IWYU pragma: keep
#include "org_argeo_jjml_llama_LlamaCppSamplerChain.h" // IWYU pragma: keep
#include "org_argeo_jjml_llama_LlamaCppSamplers.h" // IWYU pragma: keep

/*
 * STANDARD SAMPLERS
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitGreedy(
		JNIEnv*, jclass) {
	llama_sampler *smpl = llama_sampler_init_greedy();
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitPenalties(
		JNIEnv *env, jclass, jobject modelObj, jint penalty_last_n,
		jfloat penalty_repeat, jfloat penalty_freq, jfloat penalty_present,
		jboolean penalize_nl, jboolean ignore_eos) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, modelObj);
	llama_sampler *smpl = llama_sampler_init_penalties(llama_n_vocab(model),
			llama_token_eos(model), llama_token_nl(model), penalty_last_n,
			penalty_repeat, penalty_freq, penalty_present, penalize_nl,
			ignore_eos);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTopK(
		JNIEnv*, jclass, jint top_k) {
	return reinterpret_cast<jlong>(llama_sampler_init_top_k(top_k));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTopP(
		JNIEnv*, jclass, jfloat top_p, jlong min_keep) {
	return reinterpret_cast<jlong>(llama_sampler_init_top_p(top_p, min_keep));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitMinP(
		JNIEnv*, jclass, jfloat min_p, jlong min_keep) {
	return reinterpret_cast<jlong>(llama_sampler_init_min_p(min_p, min_keep));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTailFree(
		JNIEnv*, jclass, jfloat tfs_z, jlong min_keep) {
	return reinterpret_cast<jlong>(llama_sampler_init_tail_free(tfs_z, min_keep));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTypicalP(
		JNIEnv*, jclass, jfloat typ_p, jlong min_keep) {
	return reinterpret_cast<jlong>(llama_sampler_init_typical(typ_p, min_keep));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTempExt(
		JNIEnv*, jclass, jfloat temp, jfloat dynatemp_ext,
		jfloat dynatemp_exponent) {
	return reinterpret_cast<jlong>(llama_sampler_init_temp_ext(temp,
			dynatemp_ext, dynatemp_exponent));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTemp(
		JNIEnv*, jclass, jfloat temp) {
	return reinterpret_cast<jlong>(llama_sampler_init_temp(temp));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitSoftMax(
		JNIEnv*, jclass) {
	return reinterpret_cast<jlong>(llama_sampler_init_softmax());
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitDist__(
		JNIEnv*, jclass) {
	return reinterpret_cast<jlong>(llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitDist__I(
		JNIEnv*, jclass, jint seed) {
	return reinterpret_cast<jlong>(llama_sampler_init_dist(seed));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitGrammar(
		JNIEnv *env, jclass, jobject modelObj, jstring grammarStr,
		jstring rootStr) {
	auto *model = argeo::jni::as_pointer<llama_model*>(env, modelObj);

	const char *grammar_str = env->GetStringUTFChars(grammarStr, nullptr);
	const char *grammar_root = env->GetStringUTFChars(rootStr, nullptr);

	llama_sampler *smpl = llama_sampler_init_grammar(model, grammar_str,
			grammar_root);

	// clean up
	env->ReleaseStringUTFChars(grammarStr, grammar_str);
	env->ReleaseStringUTFChars(rootStr, grammar_root);

	return reinterpret_cast<jlong>(smpl);
}

/*
 * JAVA SAMPLER
 */
struct jjml_llama_sampler_java {
	const jobject obj; //
	const jmethodID applyMethod; //
	const jmethodID acceptMethod; //
	const jmethodID resetMethod; //
	const char *name; //
	JavaVM *jvm; //
};

void jjml_llama_sampler_java_apply(struct llama_sampler *smpl,
		llama_token_data_array *cur_p) {
	const auto *ctx = static_cast<jjml_llama_sampler_java*>(smpl->ctx);
	JNIEnv *env;
	argeo::jni::load_thread_jnienv(ctx->jvm, (void**) &env);

	jobject obj = env->NewLocalRef(ctx->obj);
	jobject buf = env->NewDirectByteBuffer(cur_p->data,
			cur_p->size * sizeof(llama_token_data));
	jlong selected = env->CallLongMethod(obj, ctx->applyMethod, buf,
			cur_p->size, cur_p->selected, cur_p->sorted);

	if (selected >= 0) {
		cur_p->selected = static_cast<int64_t>(selected);
	}
}

void jjml_llama_sampler_java_accept(struct llama_sampler *smpl,
		llama_token token) {
	const auto *ctx = static_cast<jjml_llama_sampler_java*>(smpl->ctx);
	JNIEnv *env;
	argeo::jni::load_thread_jnienv(ctx->jvm, (void**) &env);

	jobject obj = env->NewLocalRef(ctx->obj);
	env->CallVoidMethod(obj, ctx->acceptMethod, token);
}

void jjml_llama_sampler_java_reset(struct llama_sampler *smpl) {
	const auto *ctx = static_cast<jjml_llama_sampler_java*>(smpl->ctx);
	JNIEnv *env;
	argeo::jni::load_thread_jnienv(ctx->jvm, (void**) &env);

	jobject obj = env->NewLocalRef(ctx->obj);
	env->CallVoidMethod(obj, ctx->resetMethod);
}

void jjml_llama_sampler_java_free(struct llama_sampler *smpl) {
	const auto *ctx = static_cast<jjml_llama_sampler_java*>(smpl->ctx);
	JNIEnv *env;
	argeo::jni::load_thread_jnienv(ctx->jvm, (void**) &env);

	// TODO call a close method on the Java object?
	env->DeleteGlobalRef(ctx->obj);
	delete ctx;
}

static const char* jjml_llama_sampler_java_name(
		const struct llama_sampler *smpl) {
	//return "java";
	const auto *ctx = static_cast<jjml_llama_sampler_java*>(smpl->ctx);
	return ctx->name;
}

static struct llama_sampler_i jjml_llama_sampler_java_i = {
/* .name   = */jjml_llama_sampler_java_name,
/* .accept = */jjml_llama_sampler_java_accept,
/* .apply  = */jjml_llama_sampler_java_apply,
/* .reset  = */jjml_llama_sampler_java_reset,
/* .clone  = */nullptr,
/* .free   = */jjml_llama_sampler_java_free, };

struct llama_sampler* jjml_llama_sampler_init_java(JNIEnv *env, jobject obj) {
//	jclass clss = argeo::jni::find_jclass(env, JCLASS_JAVA_SAMPLER);
	jjml_llama_sampler_java *ctx = new jjml_llama_sampler_java {
			env->NewGlobalRef(obj), //
			LlamaCppJavaSampler$apply, //
			LlamaCppJavaSampler$accept, //
			LlamaCppJavaSampler$reset, //
			"java" //
			};
	env->GetJavaVM(&ctx->jvm);
	return new llama_sampler {
	/* .iface = */&jjml_llama_sampler_java_i,
	/* .ctx   = */ctx, //
	};
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitJavaSampler(
		JNIEnv *env, jclass, jobject javaSamplerObj) {
	return reinterpret_cast<jlong>(jjml_llama_sampler_init_java(env,
			javaSamplerObj));
}
/*
 * SAMPLER CHAIN
 */
JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplerChain_doInit(
		JNIEnv*, jclass) {
	auto sparams = llama_sampler_chain_default_params();
	llama_sampler *smpl = llama_sampler_chain_init(sparams);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplerChain_doAddSampler(
		JNIEnv *env, jobject obj, jobject child) {
	auto *chain = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(env, child);
	llama_sampler_chain_add(chain, smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplerChain_doRemoveSampler(
		JNIEnv *env, jobject obj, jint index) {
	auto *chain = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	auto *smpl = llama_sampler_chain_remove(chain, index);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplerChain_doGetSampler(
		JNIEnv *env, jobject obj, jint index) {
	auto *chain = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	auto *smpl = llama_sampler_chain_get(chain, index);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplerChain_doGetSize(
		JNIEnv *env, jobject obj) {
	auto *chain = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	return llama_sampler_chain_n(chain);
}

/*
 * GENERIC SAMPLER
 */
JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppNativeSampler_doReset(
		JNIEnv *env, jobject obj) {
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	llama_sampler_reset(smpl);
}

JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppNativeSampler_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	llama_sampler_free(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppNativeSampler_doClone(
		JNIEnv *env, jobject obj) {
	auto *smpl = argeo::jni::as_pointer<llama_sampler*>(env, obj);
	auto *cloned = llama_sampler_clone(smpl);
	return reinterpret_cast<jlong>(cloned);
}
