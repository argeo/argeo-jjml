package org.argeo.jjml.llama;

public class LlamaCppSamplerChain extends LlamaCppNativeSampler {
	private static native long doInit();

	private native void doAddSampler(LlamaCppNativeSampler sampler);

	private native long doRemoveSampler(int index);

	private native long doGetSampler(int index);

	private native int doGetSize();

	public LlamaCppSamplerChain() {
		super(doInit());
	}

	public void addSampler(LlamaCppNativeSampler sampler) {
		// we cannot have a sampler shared between chain (or added twice), as it would
		// cause problems when the chain is closed.
		if (sampler.getSamplerChain() != null) {
			throw new IllegalStateException(
					"A sampler cannot be used by two chains or added twice, it should be removed first.");
		}

		doAddSampler(sampler);
		sampler.setSamplerChain(this);
	}

}
