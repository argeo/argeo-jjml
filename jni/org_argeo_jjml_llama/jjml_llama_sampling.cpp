#include <llama.h>

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
	auto *model = argeo::jni::getPointer<llama_model*>(env, modelObj);
	auto *smpl = llama_sampler_init_penalties(llama_n_vocab(model),
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
		JNIEnv*, jclass, jfloat top_p, jfloat min_p) {
	return reinterpret_cast<jlong>(llama_sampler_init_top_p(top_p, min_p));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitTemp(
		JNIEnv*, jclass, jfloat temp) {
	return reinterpret_cast<jlong>(llama_sampler_init_temp(temp));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitDist__(
		JNIEnv*, jclass) {
	return reinterpret_cast<jlong>(llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplers_doInitDist__I(
		JNIEnv*, jclass, jint seed) {
	return reinterpret_cast<jlong>(llama_sampler_init_dist(seed));
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
	auto *chain = argeo::jni::getPointer<llama_sampler*>(env, obj);
	auto *smpl = argeo::jni::getPointer<llama_sampler*>(env, child);
	llama_sampler_chain_add(chain, smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplerChain_doRemoveSampler(
		JNIEnv *env, jobject obj, jint index) {
	auto *chain = argeo::jni::getPointer<llama_sampler*>(env, obj);
	auto *smpl = llama_sampler_chain_remove(chain, index);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplerChain_doGetSampler(
		JNIEnv *env, jobject obj, jint index) {
	auto *chain = argeo::jni::getPointer<llama_sampler*>(env, obj);
	auto *smpl = llama_sampler_chain_get(chain, index);
	return reinterpret_cast<jlong>(smpl);
}

JNIEXPORT jint JNICALL Java_org_argeo_jjml_llama_LlamaCppSamplerChain_doGetSize(
		JNIEnv *env, jobject obj) {
	auto *chain = argeo::jni::getPointer<llama_sampler*>(env, obj);
	return llama_sampler_chain_n(chain);
}

/*
 * GENERIC SAMPLER
 */
JNIEXPORT void JNICALL Java_org_argeo_jjml_llama_LlamaCppNativeSampler_doDestroy(
		JNIEnv *env, jobject obj) {
	auto *smpl = argeo::jni::getPointer<llama_sampler*>(env, obj);
	llama_sampler_free(smpl);
}

JNIEXPORT jlong JNICALL Java_org_argeo_jjml_llama_LlamaCppNativeSampler_doClone(
		JNIEnv *env, jobject obj) {
	auto *smpl = argeo::jni::getPointer<llama_sampler*>(env, obj);
	auto *cloned = llama_sampler_clone(smpl);
	return reinterpret_cast<jlong>(cloned);
}
