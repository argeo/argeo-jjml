package org.argeo.jjml.llama;

/** Access to the native standard samplers. */
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

	private static native long doInitTopP(float top_p, float min_p);

	private static native long doInitTemp(float temp);

	private static native long doInitDist();

	private static native long doInitDist(int seed);

	/*
	 * DEFAULT CHAINS
	 */
	public static LlamaCppSamplerChain newDefaultSampler(LlamaCppModel model) {
		return newDefaultSampler(model, false);
	}

	public static LlamaCppSamplerChain newDefaultSampler(LlamaCppModel model, boolean withTemp) {
		return newDefaultSampler(model, withTemp ? 0.8f : 0.0f);
	}

	public static LlamaCppSamplerChain newDefaultSampler(LlamaCppModel model, float temp) {
		LlamaCppSamplerChain chain = new LlamaCppSamplerChain();
		chain.addSampler(LlamaCppSamplers.newSamplerPenalties(model));
		if (temp > 0) {
			chain.addSampler(LlamaCppSamplers.newSamplerTopK(40));
			chain.addSampler(LlamaCppSamplers.newSamplerTopP(0.95f, 0.05f));
			chain.addSampler(LlamaCppSamplers.newSamplerTemp(temp));

			// final sampler
			chain.addSampler(LlamaCppSamplers.newSamplerDist());
		} else {
			chain.addSampler(LlamaCppSamplers.newSamplerGreedy());
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

	public static LlamaCppNativeSampler newSamplerPenalties(LlamaCppModel model) {
		return newSamplerPenalties(model, 64, 1, 0, 0, false, false);

	}

	public static LlamaCppNativeSampler newSamplerTopK(int top_k) {
		return new LlamaCppNativeSampler(doInitTopK(top_k));
	}

	public static LlamaCppNativeSampler newSamplerTopP(float top_p, float min_p) {
		return new LlamaCppNativeSampler(doInitTopP(top_p, min_p));
	}

	public static LlamaCppNativeSampler newSamplerTemp(float temp) {
		return new LlamaCppNativeSampler(doInitTemp(temp));
	}

	public static LlamaCppNativeSampler newSamplerDist(int seed) {
		return new LlamaCppNativeSampler(doInitDist(seed));
	}

	public static LlamaCppNativeSampler newSamplerDist() {
		return new LlamaCppNativeSampler(doInitDist());
	}

	/** singleton */
	private LlamaCppSamplers() {
	}
}
