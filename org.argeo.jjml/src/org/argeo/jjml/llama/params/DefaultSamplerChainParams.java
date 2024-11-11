package org.argeo.jjml.llama.params;

/** Parameters for the default sampler chain. */
public class DefaultSamplerChainParams {//
// TODO distinct params records per sampler
	// see gpt_sampler_params in common.h
	float temp; // temperature
	int n_probs; // if greater than 0, output the probabilities of top n_probs tokens
	long min_keep; // minimal number of tokens to keep
	int top_k; // <= 0 to use vocab size
	float top_p; // 1.0 = disabled
	float min_p; // 0.0 = disabled
	float tfs_z; // 1.0 = disabled
	float typ_p; // typical_p, 1.0 = disabled
	float dynatemp_range; // 0.0 = disabled
	float dynatemp_exponent; // controls how entropy maps to temperature in dynamic temperature sampler
	int penalty_last_n; // last n tokens to penalize (0 = disable penalty, -1 = context size)
	float penalty_repeat; // 1.0 = disabled
	float penalty_freq; // 0.0 = disabled
	float penalty_present; // 0.0 = disabled
	boolean penalize_nl; // consider newlines as a repeatable token
	boolean ignore_eos; //

	/** Record-like full constructor */
	public DefaultSamplerChainParams(//
			float temp, //
			int n_probs, //
			long min_keep, //
			int top_k, //
			float top_p, //
			float min_p, //
			float tfs_z, //
			float typ_p, //
			float dynatemp_range, //
			float dynatemp_exponent, //
			int penalty_last_n, //
			float penalty_repeat, //
			float penalty_freq, //
			float penalty_present, //
			boolean penalize_nl, //
			boolean ignore_eos //
	) {
		this.temp = temp;
		this.n_probs = n_probs;
		this.min_keep = min_keep;
		this.top_k = top_k;
		this.top_p = top_p;
		this.min_p = min_p;
		this.tfs_z = tfs_z;
		this.typ_p = typ_p;
		this.dynatemp_range = dynatemp_range;
		this.dynatemp_exponent = dynatemp_exponent;
		this.penalty_last_n = penalty_last_n;
		this.penalty_repeat = penalty_repeat;
		this.penalty_freq = penalty_freq;
		this.penalty_present = penalty_present;
		this.penalize_nl = penalize_nl;
		this.ignore_eos = ignore_eos;
	}

	public DefaultSamplerChainParams() {
		this(0.80f);
	}

	/** Default non-deterministic. */
	public DefaultSamplerChainParams(float temp) {
		this(temp, 0);
	}

	/** Default deterministic. */
	public DefaultSamplerChainParams(int n_probs) {
		this(0, n_probs);
	}

	/** Defaults. */
	public DefaultSamplerChainParams(float temp, int n_probs) {
		this(temp, n_probs, 0l, 40, 0.95f, 0.05f, 1.00f, 1.00f, 0.00f, 1.00f, 64, 1.00f, 0.00f, 0.00f, false, false);
	}

	/*
	 * GETTERS
	 */

	public float temp() {
		return temp;
	}

	public int n_probs() {
		return n_probs;
	}

	public long min_keep() {
		return min_keep;
	}

	public int top_k() {
		return top_k;
	}

	public float top_p() {
		return top_p;
	}

	public float min_p() {
		return min_p;
	}

	public float tfs_z() {
		return tfs_z;
	}

	public float typ_p() {
		return typ_p;
	}

	public float dynatemp_range() {
		return dynatemp_range;
	}

	public float dynatemp_exponent() {
		return dynatemp_exponent;
	}

	public int penalty_last_n() {
		return penalty_last_n;
	}

	public float penalty_repeat() {
		return penalty_repeat;
	}

	public float penalty_freq() {
		return penalty_freq;
	}

	public float penalty_present() {
		return penalty_present;
	}

	public boolean penalize_nl() {
		return penalize_nl;
	}

	public boolean ignore_eos() {
		return ignore_eos;
	}

}
