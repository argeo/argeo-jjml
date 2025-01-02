package org.argeo.jjml.llama;

import org.argeo.jjml.llama.params.DefaultSamplerChainParams;

/**
 * Access to the native standard samplers.
 * 
 * @see llama.h - llama_sampler_init_*
 */
public class LlamaCppSamplers {

	/*
	 * NATIVE
	 */
	private static native long doInitGreedy();

	private static native long doInitPenalties( //
			LlamaCppModel model, //
			int penalty_last_n, //
			float penalty_repeat, //
			float penalty_freq, //
			float penalty_present, //
			boolean penalize_nl, //
			boolean ignore_eos //
	);

	private static native long doInitTopK(int top_k);

	private static native long doInitTopP(float top_p, long min_keep);

	private static native long doInitMinP(float min_p, long min_keep);

	//private static native long doInitTailFree(float tfs_z, long min_keep);

	private static native long doInitTypicalP(float typ_p, long min_keep);

	private static native long doInitTempExt(float temp, float dynatemp_range, float dynatemp_exponent);

	private static native long doInitTemp(float temp);

	//private static native long doInitSoftMax();

	private static native long doInitDist();

	private static native long doInitDist(int seed);

	private static native long doInitGrammar(LlamaCppModel model, String grammar, String root);

	private static native long doInitJavaSampler(LlamaCppJavaSampler javaSampler);

	/*
	 * DEFAULT CHAINS
	 */
	public static LlamaCppSamplerChain newDefaultSampler(LlamaCppModel model) {
		return newDefaultSampler(model, false);
	}

	public static LlamaCppSamplerChain newDefaultSampler(LlamaCppModel model, boolean withTemp) {
		return newDefaultSampler(model, withTemp ? new DefaultSamplerChainParams() : new DefaultSamplerChainParams(0));
	}

	public static LlamaCppSamplerChain newDefaultSampler(LlamaCppModel model, DefaultSamplerChainParams params) {
		// see gpt_sampler_init in sampling.cpp

		LlamaCppSamplerChain chain = new LlamaCppSamplerChain();
		chain.addSampler(LlamaCppSamplers.newSamplerPenalties(model, params));
		if (params.temp() > 0) {
			chain.addSampler(LlamaCppSamplers.newSamplerTopK(params.top_k()));
			long min_keep = params.min_keep();
			//chain.addSampler(LlamaCppSamplers.newSamplerTailFree(params.tfs_z(), min_keep));
			chain.addSampler(LlamaCppSamplers.newSamplerTypicalP(params.typ_p(), min_keep));
			chain.addSampler(LlamaCppSamplers.newSamplerTopP(params.top_p(), min_keep));
			chain.addSampler(LlamaCppSamplers.newSamplerMinP(params.min_p(), min_keep));
			chain.addSampler(LlamaCppSamplers.newSamplerTempExt(params.temp(), params.dynatemp_range(),
					params.dynatemp_exponent()));

			// final sampler
			//chain.addSampler(LlamaCppSamplers.newSamplerSoftMax());
			chain.addSampler(LlamaCppSamplers.newSamplerDist());
		} else {
			if (params.n_probs() > 0) {
				chain.addSampler(LlamaCppSamplers.newSamplerTopK(params.n_probs()));
				//chain.addSampler(LlamaCppSamplers.newSamplerSoftMax());
			}
			chain.addSampler(LlamaCppSamplers.newSamplerGreedy());
//			chain.addSampler(LlamaCppSamplers.newJavaSampler(new LlamaCppJavaSampler.SimpleGreedy()));
		}
		return chain;
	}

	/*
	 * FACTORY
	 */
	public static LlamaCppNativeSampler newSamplerGreedy() {
		return new LlamaCppNativeSampler(doInitGreedy());
	}

	public static LlamaCppNativeSampler newSamplerPenalties(LlamaCppModel model, //
			int penalty_last_n, // last n tokens to penalize (0 = disable penalty, -1 = context size)
			float penalty_repeat, // 1.0 = disabled
			float penalty_freq, // 0.0 = disabled
			float penalty_present, // 0.0 = disabled
			boolean penalize_nl, // consider newlines as a repeatable token
			boolean ignore_eos // ignore the end-of-sequence token
	) {
		return new LlamaCppNativeSampler(doInitPenalties(model, penalty_last_n, penalty_repeat, penalty_freq,
				penalty_present, penalize_nl, ignore_eos));

	}

	public static LlamaCppNativeSampler newSamplerPenalties(LlamaCppModel model, DefaultSamplerChainParams params) {
		return newSamplerPenalties(model, params.penalty_last_n(), params.penalty_repeat(), params.penalty_freq(),
				params.penalty_freq(), params.penalize_nl(), params.ignore_eos());

	}

	public static LlamaCppNativeSampler newSamplerTopK(int top_k) {
		return new LlamaCppNativeSampler(doInitTopK(top_k));
	}

	public static LlamaCppNativeSampler newSamplerTopP(float top_p, long min_keep) {
		return new LlamaCppNativeSampler(doInitTopP(top_p, min_keep));
	}

	public static LlamaCppNativeSampler newSamplerMinP(float min_p, long min_keep) {
		return new LlamaCppNativeSampler(doInitMinP(min_p, min_keep));
	}

//	public static LlamaCppNativeSampler newSamplerTailFree(float tfs_z, long min_keep) {
//		return new LlamaCppNativeSampler(doInitTailFree(tfs_z, min_keep));
//	}

	public static LlamaCppNativeSampler newSamplerTypicalP(float typ_p, long min_keep) {
		return new LlamaCppNativeSampler(doInitTypicalP(typ_p, min_keep));
	}

	public static LlamaCppNativeSampler newSamplerTempExt(float temp, float dynatemp_range, float dynatemp_exponent) {
		return new LlamaCppNativeSampler(doInitTempExt(temp, dynatemp_range, dynatemp_exponent));
	}

	public static LlamaCppNativeSampler newSamplerTemp(float temp) {
		return new LlamaCppNativeSampler(doInitTemp(temp));
	}

//	public static LlamaCppNativeSampler newSamplerSoftMax() {
//		return new LlamaCppNativeSampler(doInitSoftMax());
//	}

	public static LlamaCppNativeSampler newSamplerDist(int seed) {
		return new LlamaCppNativeSampler(doInitDist(seed));
	}

	public static LlamaCppNativeSampler newSamplerDist() {
		return new LlamaCppNativeSampler(doInitDist());
	}

	public static LlamaCppNativeSampler newSamplerGrammar(LlamaCppModel model, String grammar, String root) {
		return new LlamaCppNativeSampler(doInitGrammar(model, grammar, root));
	}

	public static LlamaCppNativeSampler newJavaSampler(LlamaCppJavaSampler javaSampler) {
		return new LlamaCppNativeSampler(doInitJavaSampler(javaSampler));
	}

	/** singleton */
	private LlamaCppSamplers() {
	}
}
